#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import train_contracts as C


class LadderTests(unittest.TestCase):
    def test_exact_transition_matrix(self):
        self.assertEqual(C.validate_transition(None, 4), "train")
        for step in (1, 2, 3):
            state = {"next_update": step, "last_passed_milestone": 0, "gate": None}
            self.assertEqual(C.validate_transition(state, 4), "train")
            with self.assertRaises(ValueError):
                C.validate_transition(state, 8)

        at4 = {"next_update": 4, "last_passed_milestone": 0, "gate": None}
        self.assertEqual(C.validate_transition(at4, 4), "gate")
        self.assertEqual(C.validate_transition(C.gated_state(4), 8), "train")

    def test_warmup_never_resets(self):
        self.assertEqual(
            [C.lr_for_update(i) for i in range(1, 7)],
            [2.5e-4, 2.5e-4, 5e-4, 5e-4, 5e-4, 5e-4],
        )
        self.assertEqual(C.lr_for_update(9), 5e-4)

    def test_step4_to_step8_consumes_exact_disjoint_bank_once(self):
        rows = list(range(17, 33))
        selected = [
            C.continuation_rows(rows, update, start_update=4, target_update=8)
            for update in range(5, 9)
        ]
        self.assertEqual([len(part) for part in selected], [4, 4, 4, 4])
        flat = [value for part in selected for value in part]
        self.assertEqual(sorted(flat), rows)
        self.assertEqual(len(flat), len(set(flat)))

    def test_deep_dose_reuses_each_bank_once_per_block(self):
        rows = list(range(17, 33))
        selected = [C.deep_dose_rows(rows, update) for update in range(9, 17)]
        for block in (selected[:4], selected[4:]):
            flat = [value for part in block for value in part]
            self.assertEqual(sorted(flat), rows)
            self.assertEqual(len(flat), len(set(flat)))

    def test_campaign_transfer_continuation_consumes_shard_once(self):
        rows = list(range(17, 33))
        selected = [
            C.campaign_continuation_rows(
                rows, update, start_update=4, target_update=8
            )
            for update in range(5, 9)
        ]
        flat = [value for part in selected for value in part]
        self.assertEqual(sorted(flat), rows)
        self.assertEqual(len(flat), len(set(flat)))
        with self.assertRaisesRegex(ValueError, "4→8 or 8→16"):
            C.campaign_continuation_rows(
                rows, 5, start_update=4, target_update=16
            )

    def test_committed_source_seal_verifies(self):
        root = Path(__file__).resolve().parents[1]
        seal = json.loads((root / "SOURCE_SEAL.json").read_text())
        C.verify_source_seal(root, seal)


class DurabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("PTQ_OPD_BANK", str(Path(__file__).resolve()))
        spec = importlib.util.spec_from_file_location("test_ptq_opd_trainer", root / "train_ptq_opd.py")
        assert spec is not None and spec.loader is not None
        cls.trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.trainer)

    def test_checkpoint_save_fsyncs_file_then_rename_then_directory(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "LATEST.pt"
            events = []
            real_fsync = os.fsync
            real_replace = os.replace

            def recording_fsync(fd):
                events.append("fsync_dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync_file")
                return real_fsync(fd)

            def recording_replace(source, destination):
                events.append("replace")
                return real_replace(source, destination)

            with mock.patch.object(self.trainer.os, "fsync", side_effect=recording_fsync), \
                    mock.patch.object(self.trainer.os, "replace", side_effect=recording_replace):
                self.trainer.save_torch_atomic(path, {"next_update": 5})

            self.assertEqual(events, ["fsync_file", "replace", "fsync_dir"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["next_update"], 5)
            self.assertFalse(Path(str(path) + ".tmp").exists())

    def test_step4_parent_lineage_is_accepted_without_adam_reset(self):
        candidate_sha = "c" * 64
        bank_sha = "b" * 64
        gate = {
            "format": "ptq-opd-static-gate-receipt-v2",
            "milestone": 4,
            "candidate_sha256": candidate_sha,
            "passed": True,
            "decision": {"passed": True},
        }
        gate_sha = C.canonical_sha256(gate)
        checkpoint = {
            "format": "ptq-opd-train-v2",
            "next_update": 4,
            "last_passed_milestone": 4,
            "objective": "reverse_kl",
            "beta": 0.5,
            "anchor_weight": 0.5,
            "trainable_params": self.trainer.EXPECTED_TRAINABLE_PARAMS,
            "promotable": False,
            "paired_gate_required": True,
            "identity": {"bank_manifest_sha256": bank_sha},
            "state": {"one": torch.tensor(1)},
            "optimizer": {
                "state": {
                    index: {"step": torch.tensor(4)}
                    for index in range(self.trainer.EXPECTED_OPTIMIZER_ENTRIES)
                }
            },
            "candidate_sha256": candidate_sha,
            "gate_receipt_sha256": gate_sha,
            "lineage": {"parent": "step0"},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint_path = root / "LATEST.pt"
            gate_path = root / "STATIC_GATE_STEP4.json"
            torch.save(checkpoint, checkpoint_path)
            gate_path.write_text(json.dumps(gate))
            with mock.patch.multiple(
                self.trainer,
                START_UPDATE=4,
                TARGET=8,
                START_CHECKPOINT=checkpoint_path,
                START_CHECKPOINT_SHA256=self.trainer.sha256_file(checkpoint_path),
                START_CANDIDATE_SHA256=candidate_sha,
                START_GATE=gate_path,
                START_GATE_CANONICAL_SHA256=gate_sha,
                START_BANK_SHA256=bank_sha,
                OBJECTIVE="reverse_kl",
                BETA=0.5,
                ANCHOR_WEIGHT=0.5,
            ):
                loaded, metadata = self.trainer.load_start_lineage()
        self.assertEqual(loaded["next_update"], 4)
        self.assertEqual(metadata["adam_readback"]["optimizer_step_min"], 4)
        self.assertTrue(metadata["lineage"]["state_optimizer_loaded_without_reset"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
