#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

import ptq_opd as G


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DivergenceTests(unittest.TestCase):
    def setUp(self):
        self.ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.teacher_lp = torch.log(torch.tensor([[0.45, 0.30], [0.50, 0.25]], dtype=torch.float64))
        self.teacher_tail_lp = torch.log(torch.tensor([0.25, 0.25], dtype=torch.float64))
        self.student = torch.tensor(
            [[0.30, 0.10, -0.20, -1.10], [0.20, 0.35, -0.10, -0.90]],
            dtype=torch.float64,
            requires_grad=True,
        )

    @staticmethod
    def reference(student, ids, teacher_lp, teacher_tail_lp, objective, beta):
        q = torch.softmax(student.double(), -1)
        q_top = q.gather(1, ids)
        q_tail = 1.0 - q_top.sum(-1, keepdim=True)
        p_top = teacher_lp.double().exp()
        p_tail = teacher_tail_lp.double().exp().unsqueeze(-1)
        p = torch.cat([p_top, p_tail], -1)
        q = torch.cat([q_top, q_tail], -1)
        if objective == "reverse_kl":
            return (q * (q.log() - p.log())).sum(-1).mean()
        m = beta * p + (1.0 - beta) * q
        return (beta * (p * (p.log() - m.log())).sum(-1) +
                (1.0 - beta) * (q * (q.log() - m.log())).sum(-1)).mean()

    def test_expected_values_jsd_beta_nonhalf_and_reverse_kl(self):
        for objective, beta in (("jsd", 0.3), ("jsd", 0.5), ("reverse_kl", 0.5)):
            got = G.bucketed_divergence(self.student, self.ids, self.teacher_lp, self.teacher_tail_lp, objective=objective, beta=beta)
            want = self.reference(self.student, self.ids, self.teacher_lp, self.teacher_tail_lp, objective, beta)
            self.assertAlmostEqual(float(got), float(want), places=7)

    def test_low_precision_parity_and_nonzero_gradient(self):
        base = torch.tensor([[0.40, 0.39, 0.05, -0.3]], dtype=torch.float32)
        ids = torch.tensor([[0, 1]], dtype=torch.long)
        teacher_lp = torch.log(torch.tensor([[0.410, 0.385]], dtype=torch.float32))
        ref = G.bucketed_divergence(base.clone().requires_grad_(True), ids, teacher_lp, torch.log(torch.tensor([0.205])))
        self.assertGreater(float(ref), 0.0)
        for dtype in (torch.bfloat16, torch.float16):
            x = base.to(dtype).requires_grad_(True)
            with torch.autocast(device_type="cpu", dtype=dtype):
                # Bank teacher tensors remain FP32; only the student forward is
                # low precision before the loss enters disabled autocast.
                loss = G.bucketed_divergence(x, ids, teacher_lp, torch.log(torch.tensor([0.205])))
            loss.backward()
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.float().abs().max()), 0.0)
            self.assertLess(abs(float(loss) - float(ref)), 2e-4)

    def test_large_support_fp32_partition_rounding_is_normalized(self):
        # Separately exponentiating a full-vocabulary normalizer for 8K gathered
        # buckets and one aggregate tail can sum to 1.0000057 in FP32 even
        # though both pieces partition the same finite logits exactly.
        vocab, topk = 129_280, 8_192
        generator = torch.Generator().manual_seed(19)
        base = (torch.randn((1, vocab), generator=generator) * 16).to(torch.bfloat16).float()
        ids = torch.randperm(vocab, generator=generator)[:topk].view(1, -1)
        teacher_lp = torch.full((1, topk), math.log(0.5 / topk), dtype=torch.float32)
        teacher_tail_lp = torch.tensor([math.log(0.5)], dtype=torch.float32)

        with torch.no_grad():
            log_z = torch.logsumexp(base, dim=-1, keepdim=True)
            q_top = torch.exp(base.gather(1, ids) - log_z)
            non_top = base.clone()
            non_top.scatter_(1, ids, -torch.inf)
            q_tail = torch.exp(torch.logsumexp(non_top, dim=-1, keepdim=True) - log_z)
            raw_mass_excess = float((q_top.sum(-1, keepdim=True) + q_tail - 1.0).item())
        self.assertGreater(raw_mass_excess, 4e-6)

        for objective in ("jsd", "reverse_kl"):
            logits = base.clone().requires_grad_(True)
            loss = G.bucketed_divergence(
                logits, ids, teacher_lp, teacher_tail_lp, objective=objective
            )
            loss.backward()

            ref_logits = base.double().requires_grad_(True)
            ref_loss = self.reference(
                ref_logits, ids, teacher_lp.double(), teacher_tail_lp.double(), objective, 0.5
            )
            ref_loss.backward()
            relative_grad_error = float(
                (logits.grad.double() - ref_logits.grad).norm()
                / ref_logits.grad.norm().clamp_min(torch.finfo(torch.float64).tiny)
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertLess(abs(float(loss.detach()) - float(ref_loss.detach())), 2e-7)
            self.assertLess(relative_grad_error, 5e-6)

    def test_rejects_gross_student_partition_drift_before_normalizing(self):
        logits = torch.zeros((1, 10), dtype=torch.float32, requires_grad=True)
        duplicate_ids = torch.tensor([[0, 0]], dtype=torch.long)
        teacher_lp = torch.log(torch.tensor([[0.2, 0.2]], dtype=torch.float32))
        teacher_tail_lp = torch.log(torch.tensor([0.6], dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "student bucket partition drift"):
            G.bucketed_divergence(logits, duplicate_ids, teacher_lp, teacher_tail_lp)

    def test_rejects_impossible_teacher_mass_without_tail_clamp(self):
        lp = torch.log(torch.tensor([[0.6000008, 0.4000008]], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "top-k plus exact tail"):
            G.bucketed_divergence(self.student[:1], self.ids[:1], lp, torch.log(torch.tensor([0.01])))

    def test_saturated_student_topk_keeps_finite_nonzero_tail_gradient(self):
        # softmax([30, 0, -1, -2])[0] rounds to exactly one in FP32.  A
        # subtractive tail therefore becomes zero even though the true aggregate
        # non-top-k mass is positive, and x*log(x) produces NaN gradients.
        logits = torch.tensor([[30.0, 0.0, -1.0, -2.0]], requires_grad=True)
        ids = torch.tensor([[0]], dtype=torch.long)
        teacher_lp = torch.log(torch.tensor([[0.8]], dtype=torch.float32))
        loss = G.bucketed_divergence(logits, ids, teacher_lp, torch.log(torch.tensor([0.2])), objective="jsd")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().max()), 0.0)


class BankContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.score_path = self.root / "scores" / "p0.pt"
        self.score_path.parent.mkdir()
        self.tensors = {
            "teacher_topk_ids": torch.tensor([[11, 8, 7], [12, 4, 3]], dtype=torch.long),
            "teacher_topk_logprobs": torch.log(torch.tensor([[0.5, 0.2, 0.1], [0.4, 0.3, 0.1]], dtype=torch.float32)),
            "teacher_target_logprobs": torch.log(torch.tensor([0.5, 0.4], dtype=torch.float32)),
            "teacher_tail_logmass": torch.log(torch.tensor([0.2, 0.2], dtype=torch.float32)),
        }
        torch.save(self.tensors, self.score_path)
        digest = G.sha256_file(self.score_path)
        self.row = {
            "schema": G.ROW_SCHEMA,
            "prompt_id": "train/000",
            "sample_id": "train/000/greedy/0",
            "sampling_role": "greedy",
            "corpus_track": "benchmark_distribution_mechanism",
            "shippable": False,
            "generation_config": {
                "phase": "greedy", "temperature": 0.0, "top_p": 1.0,
                "max_tokens": 4096, "n": 1, "seed_base": 7192026000,
                "seed": 7192026000,
            },
            "generation_config_sha256": hashlib.sha256(json.dumps({
                "phase": "greedy", "temperature": 0.0, "top_p": 1.0,
                "max_tokens": 4096, "n": 1, "seed_base": 7192026000,
                "seed": 7192026000,
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "student_checkpoint_sha256": G.STUDENT_SHA256,
            "serve_fingerprint": G.SERVE_FINGERPRINT,
            "prompt_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "token_ids": [1, 2, 11, 12],
            "score_start": 2,
            "score_alignment": G.SCORE_ALIGNMENT,
            "score_payload_schema": G.SCORE_PAYLOAD_SCHEMA,
            "teacher_tail_logmass_preserved": True,
            "completion_tokens": 2,
            "fp_reference_completion_tokens": 3,
            "fp_reference_sha256": "e" * 64,
            "teacher_model": "DeepSeek-V4-FP8",
            "teacher_model_sha256": "f" * 64,
            "teacher_scorer_sha256": "1" * 64,
            "teacher_topk": 3,
            "teacher_mean_nll": -float(self.tensors["teacher_target_logprobs"].mean()),
            "scores_file": "scores/p0.pt",
            "scores_sha256": digest,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_row_and_exact_alignment(self):
        loaded = G.validate_and_load_row(self.row, self.root, min_topk=3)
        self.assertEqual(loaded.n_score_tokens, 2)
        self.assertEqual(loaded.target_ids.tolist(), [11, 12])

    def test_rejects_missing_or_inconsistent_exact_teacher_tail(self):
        missing = dict(self.tensors)
        missing.pop("teacher_tail_logmass")
        torch.save(missing, self.score_path)
        self.row["scores_sha256"] = G.sha256_file(self.score_path)
        with self.assertRaisesRegex(ValueError, "payload fields"):
            G.validate_and_load_row(self.row, self.root, min_topk=3)

        bad = dict(self.tensors)
        bad["teacher_tail_logmass"] = torch.log(torch.tensor([0.3, 0.2]))
        torch.save(bad, self.score_path)
        self.row["scores_sha256"] = G.sha256_file(self.score_path)
        with self.assertRaisesRegex(ValueError, "top-k plus exact tail"):
            G.validate_and_load_row(self.row, self.root, min_topk=3)

    def test_rejects_exact_json_type_coercions_and_bools(self):
        for key, bad in (("score_start", 2.0), ("completion_tokens", True), ("teacher_topk", "3")):
            row = dict(self.row)
            row[key] = bad
            with self.subTest(key=key), self.assertRaises((TypeError, ValueError)):
                G.validate_and_load_row(row, self.root, min_topk=3)
        row = dict(self.row)
        row["token_ids"] = [1, 2, True, 12]
        with self.assertRaises(TypeError):
            G.validate_and_load_row(row, self.root, min_topk=3)

    def test_rejects_target_alignment_present_and_absent_corruption(self):
        bad = dict(self.tensors)
        bad["teacher_target_logprobs"] = bad["teacher_target_logprobs"].clone()
        bad["teacher_target_logprobs"][0] -= 0.1
        torch.save(bad, self.score_path)
        self.row["scores_sha256"] = G.sha256_file(self.score_path)
        self.row["teacher_mean_nll"] = -float(bad["teacher_target_logprobs"].mean())
        with self.assertRaisesRegex(ValueError, "target/top-k logprob mismatch"):
            G.validate_and_load_row(self.row, self.root, min_topk=3)

        absent = dict(self.tensors)
        absent["teacher_topk_ids"] = absent["teacher_topk_ids"].clone()
        absent["teacher_topk_ids"][0] = torch.tensor([8, 7, 6])
        absent["teacher_target_logprobs"] = absent["teacher_target_logprobs"].clone()
        absent["teacher_target_logprobs"][0] = absent["teacher_topk_logprobs"][0, -1] + 0.01
        torch.save(absent, self.score_path)
        self.row["scores_sha256"] = G.sha256_file(self.score_path)
        self.row["teacher_mean_nll"] = -float(absent["teacher_target_logprobs"].mean())
        with self.assertRaisesRegex(ValueError, "omitted target exceeds kth"):
            G.validate_and_load_row(self.row, self.root, min_topk=3)

    def test_completion_length_is_derived_from_scored_suffix(self):
        row = dict(self.row)
        row["completion_tokens"] = 99
        with self.assertRaisesRegex(ValueError, "completion length"):
            G.validate_and_load_row(row, self.root, min_topk=3)

    def test_nonempty_prompt_does_not_consume_completion_cap(self):
        row = dict(self.row)
        row["token_ids"] = [1] * 4095 + [11, 12]
        row["score_start"] = 4095
        row["completion_tokens"] = 2
        loaded = G.validate_and_load_row(row, self.root, min_topk=3)
        self.assertEqual(loaded.row["completion_tokens"], 2)

    def test_completion_cap_is_fail_closed(self):
        row = dict(self.row)
        row["token_ids"] = [1] * 4098
        row["score_start"] = 1
        row["completion_tokens"] = 4097
        with self.assertRaisesRegex(ValueError, "completion exceeds"):
            G.validate_and_load_row(row, self.root, min_topk=3)

    def test_paths_relative_contained_nonsymlinked_hash_same_bytes(self):
        for bad in (str(self.score_path), "../escape.pt", "scores/../../escape.pt"):
            row = dict(self.row)
            row["scores_file"] = bad
            with self.subTest(path=bad), self.assertRaises(ValueError):
                G.validate_and_load_row(row, self.root, min_topk=3)
        link = self.root / "scores" / "link.pt"
        link.symlink_to(self.score_path)
        row = dict(self.row)
        row["scores_file"] = "scores/link.pt"
        row["scores_sha256"] = G.sha256_file(self.score_path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            G.validate_and_load_row(row, self.root, min_topk=3)

    def test_bank_group_reference_identity_and_rolling_full_composition(self):
        row2 = dict(self.row)
        row2["sample_id"] = "train/000/temp0.7/1"
        row2["sampling_role"] = "temp0.7"
        row2["generation_config"] = {
            "phase": "temp0.7", "temperature": 0.7, "top_p": 1.0,
            "max_tokens": 4096, "n": 1, "seed_base": 7192026000,
            "seed": 7192026001,
        }
        row2["generation_config_sha256"] = hashlib.sha256(json.dumps(row2["generation_config"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        row2["fp_reference_completion_tokens"] = 4
        with self.assertRaisesRegex(ValueError, "FP reference"):
            G.validate_bank_rows([self.row, row2], self.root, mode="rolling", min_rows=2, min_prompts=1, min_topk=3)

        receipt = G.validate_bank_rows([self.row], self.root, mode="rolling", min_rows=1, min_prompts=1, min_topk=3)
        self.assertEqual(receipt["role_counts"], {"greedy": 1})
        with self.assertRaisesRegex(ValueError, "composition"):
            G.validate_bank_rows([self.row], self.root, mode="sealed", min_rows=1, min_prompts=1, min_topk=3)

    def test_accepts_exact_upstream_greedy_generation_provenance(self):
        row = dict(self.row)
        row["generation_config"] = {
            "phase": "greedy",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
            "n": 1,
            "seed_base": 7192026000,
            "seed": 7192026013,
        }
        row["generation_config_sha256"] = G.canonical_sha256(row["generation_config"])
        loaded = G.validate_and_load_row(row, self.root, min_topk=3)
        self.assertEqual(loaded.row["generation_config"], row["generation_config"])

    def test_track_specific_greedy_seed_base_is_exact(self):
        shippable = dict(self.row)
        shippable["corpus_track"] = "tailfix_general_shippable"
        shippable["shippable"] = True
        shippable["generation_config"] = {
            "phase": "greedy", "temperature": 0.0, "top_p": 1.0,
            "max_tokens": 4096, "n": 1, "seed_base": 7192028000,
            "seed": 7192028141,
        }
        shippable["generation_config_sha256"] = G.canonical_sha256(shippable["generation_config"])
        loaded = G.validate_and_load_row(shippable, self.root, min_topk=3)
        self.assertEqual(loaded.row["generation_config"]["seed_base"], 7192028000)

        wrong_track = dict(shippable)
        wrong_track["corpus_track"] = "benchmark_distribution_mechanism"
        wrong_track["shippable"] = False
        with self.assertRaisesRegex(ValueError, "seed_base mismatch"):
            G.validate_and_load_row(wrong_track, self.root, min_topk=3)

    def test_shippable_arm_a_allows_explicit_missing_arm_b_anchor_only(self):
        row = dict(self.row)
        row.update({
            "corpus_track": "tailfix_general_shippable",
            "shippable": True,
            "arm_a_eligible": True,
            "arm_b_eligible": False,
            "shippable_training_eligible": True,
            "benchmark_distribution_only": False,
            "fp_reference_completion_tokens": None,
            "fp_reference_sha256": None,
            "fp_reference_completion_sha256": None,
            "fp_reference_identity": None,
        })
        row["generation_config"] = {
            "phase": "greedy", "temperature": 0.0, "top_p": 1.0,
            "max_tokens": 4096, "n": 1, "seed_base": 7192028000,
            "seed": 7192028141,
        }
        row["generation_config_sha256"] = G.canonical_sha256(row["generation_config"])
        loaded = G.validate_and_load_row(row, self.root, min_topk=3)
        self.assertFalse(loaded.row["arm_b_eligible"])

        falsely_arm_b = dict(row)
        falsely_arm_b["arm_b_eligible"] = True
        with self.assertRaisesRegex(ValueError, "Arm-B anchor"):
            G.validate_and_load_row(falsely_arm_b, self.root, min_topk=3)

    def test_rejects_track_label_drift_and_false_greedy_provenance(self):
        for track, shippable in (("benchmark_distribution_mechanism", True),
                                 ("tailfix_general_shippable", False),
                                 ("unknown", False)):
            row = dict(self.row)
            row["corpus_track"] = track
            row["shippable"] = shippable
            with self.subTest(track=track, shippable=shippable), self.assertRaises((TypeError, ValueError)):
                G.validate_and_load_row(row, self.root, min_topk=3)
        row = dict(self.row)
        row["generation_config"] = {
            "phase": "greedy", "temperature": 0.0, "top_p": 1.0,
            "max_tokens": 4096, "n": 1, "seed_base": 7192026000,
            "seed": 0,
        }
        row["generation_config_sha256"] = G.canonical_sha256(row["generation_config"])
        with self.assertRaisesRegex(ValueError, "greedy generation seed"):
            G.validate_and_load_row(row, self.root, min_topk=3)

    def test_rejects_unbound_provenance_and_generation_settings(self):
        for key in ("prompt_sha256", "split_sha256", "tokenizer_sha256", "teacher_scorer_sha256"):
            row = dict(self.row)
            row[key] = "not-a-sha"
            with self.subTest(key=key), self.assertRaises(ValueError):
                G.validate_and_load_row(row, self.root, min_topk=3)
        row = dict(self.row)
        row["generation_config"] = dict(row["generation_config"], max_tokens=999)
        row["generation_config_sha256"] = G.canonical_sha256(row["generation_config"])
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            G.validate_and_load_row(row, self.root, min_topk=3)


class GateAndRankingTests(unittest.TestCase):
    def test_static_gate_requires_global_all_classes_nonnegative_finite(self):
        required = ("global", "agentic", "chat", "code", "prose", "reasoning", "multilingual")
        base = {key: 1.0 for key in required}
        cand = {key: 1.005 for key in required}
        self.assertTrue(G.static_kld_gate(base, cand, required_classes=required)["passed"])
        for mutation, match in (({"global": -1.0}, "negative"), ({"code": math.nan}, "finite")):
            broken = dict(cand); broken.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, match):
                G.static_kld_gate(base, broken, required_classes=required)
        missing = dict(cand); missing.pop("global")
        with self.assertRaisesRegex(ValueError, "class set"):
            G.static_kld_gate(base, missing, required_classes=required)

    def test_arm_b_formula_and_fixed_reference(self):
        rows = [
            {"prompt_id": "p", "sample_id": "a", "teacher_mean_nll": 0.2, "completion_tokens": 90, "fp_reference_completion_tokens": 100, "fp_reference_sha256": "a" * 64},
            {"prompt_id": "p", "sample_id": "b", "teacher_mean_nll": 0.1, "completion_tokens": 140, "fp_reference_completion_tokens": 100, "fp_reference_sha256": "a" * 64},
        ]
        ranked = G.rank_arm_b(rows)
        self.assertEqual(ranked["p"][0]["sample_id"], "a")
        rows[1]["fp_reference_completion_tokens"] = 101
        with self.assertRaisesRegex(ValueError, "FP reference"):
            G.rank_arm_b(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
