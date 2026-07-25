#!/usr/bin/env python3
"""Fail-closed contracts and FP32 objectives for PTQ-OPD own-rollout GKD.

Autoregressive alignment is explicit: for ``score_start=s`` the n score rows
are teacher predictions at positions ``s-1 .. len(token_ids)-2`` and their
exact targets are ``token_ids[s:]``. Completion length is therefore the scored
suffix length; EOS, when present, remains part of that suffix.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ROW_SCHEMA = "ptq-opd-row-v2"
SCORE_ALIGNMENT = "next_token_logits_at_score_start_minus_1_v1"
SCORE_PAYLOAD_SCHEMA = "ptq-opd-consumer-topk-target-tail-v1"
STUDENT_SHA256 = "fae41d519193269aec4b2221c97a1dc00e0b00d3d66074d917a78489fac2149c"
SERVE_FINGERPRINT = "vllm-0.24.0-3f34bf12"
MAX_COMPLETION_TOKENS = 4096
MASS_TOLERANCE = 1e-6
LOGPROB_TOLERANCE = 2e-5
MAX_SCORE_FILE_BYTES = 512 << 20
MAX_MANIFEST_BYTES = 32 << 20
DEFAULT_REQUIRED_KLD_CLASSES = (
    "global", "agentic", "chat", "code", "prose", "reasoning", "multilingual"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_ROLES = {"greedy", "temp0.7", "temp0.8"}
_UPSTREAM_SEED_BASE_BY_TRACK = {
    "benchmark_distribution_mechanism": 7192026000,
    "tailfix_general_shippable": 7192028000,
}
_UPSTREAM_GENERATION_FIELDS = {
    "phase", "temperature", "top_p", "max_tokens", "n", "seed_base", "seed",
}
_REQUIRED_SCORE_KEYS = {
    "teacher_topk_ids",
    "teacher_topk_logprobs",
    "teacher_target_logprobs",
    "teacher_tail_logmass",
}


@dataclass
class LoadedRow:
    row: Dict[str, Any]
    teacher_topk_ids: torch.Tensor
    teacher_topk_logprobs: torch.Tensor
    teacher_target_logprobs: torch.Tensor
    teacher_tail_logmass: torch.Tensor
    target_ids: torch.Tensor
    n_score_tokens: int


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _exact_int(value: object, name: str, minimum: Optional[int] = None) -> int:
    if type(value) is not int:  # bool is deliberately excluded
        raise TypeError(f"{name} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _exact_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact JSON number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or any(ord(ch) < 32 for ch in value):
        raise TypeError(f"{name} must be a nonempty printable string")
    return value


def _sha(value: object, name: str) -> str:
    text = _nonempty_string(value, name)
    if not _SHA_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _validated_tail(mass: torch.Tensor, *, label: str, tolerance: float = MASS_TOLERANCE) -> torch.Tensor:
    if not torch.isfinite(mass).all():
        raise ValueError(f"{label} mass is nonfinite")
    if bool((mass < -tolerance).any()):
        raise ValueError(f"{label} mass is negative")
    excess = mass - 1.0
    if bool((excess > tolerance).any()):
        raise ValueError(f"{label} top-k mass exceeds one")
    tail = 1.0 - mass
    # Only an explicitly validated <= tolerance rounding overshoot reaches this.
    if bool((tail < 0).any()):
        tail = torch.where(tail < 0, torch.zeros_like(tail), tail)
    return tail


def _xlog_ratio(x: torch.Tensor, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    # x*log(numerator/denominator), defining the zero-mass bucket as zero.
    # Merely masking the final x*log(x) with torch.where is not gradient-safe:
    # autograd still encounters log(0) in the unselected branch and 0*NaN can
    # poison every upstream gradient.  Replace both log arguments on zero-mass
    # rows before evaluating either logarithm.
    positive = x > 0
    one = torch.ones((), dtype=x.dtype, device=x.device)
    safe_numerator = torch.where(positive, numerator, one)
    safe_denominator = torch.where(positive, denominator, one)
    return x * (torch.log(safe_numerator) - torch.log(safe_denominator))


def bucketed_divergence(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_tail_logmass: torch.Tensor,
    *,
    objective: str = "jsd",
    beta: float = 0.5,
) -> torch.Tensor:
    """Top-k+aggregate-tail JSD or reverse-KL, entirely in FP32.

    Teacher logprobs must be absolute full-softmax values, not top-k
    renormalized values. Student probabilities use the full vocabulary before
    gathering teacher top-k IDs, so both tail buckets have their honest mass.
    """
    if objective not in {"jsd", "reverse_kl"}:
        raise ValueError("objective must be jsd or reverse_kl")
    if objective == "jsd" and not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must be in (0,1)")
    if student_logits.ndim != 2 or teacher_topk_ids.ndim != 2 or teacher_topk_logprobs.ndim != 2:
        raise ValueError("divergence tensors must be rank two")
    if teacher_tail_logmass.ndim != 1:
        raise ValueError("teacher_tail_logmass must be rank one")
    if (teacher_topk_ids.shape != teacher_topk_logprobs.shape
            or student_logits.shape[0] != teacher_topk_ids.shape[0]
            or teacher_tail_logmass.shape[0] != student_logits.shape[0]):
        raise ValueError("divergence tensor shapes mismatch")
    if teacher_topk_ids.dtype != torch.long:
        raise TypeError("teacher_topk_ids must be int64")
    if teacher_topk_ids.numel() == 0:
        raise ValueError("empty top-k support")
    if int(teacher_topk_ids.min()) < 0 or int(teacher_topk_ids.max()) >= student_logits.shape[1]:
        raise ValueError("top-k token ID outside student vocabulary")

    # A disabled autocast region is necessary even when the caller wraps the
    # complete training step in BF16/FP16 autocast.
    with torch.autocast(device_type=student_logits.device.type, enabled=False):
        logits = student_logits.float()
        teacher_lp = teacher_topk_logprobs.float()
        teacher_tail_lp = teacher_tail_logmass.float()
        if (not torch.isfinite(logits).all() or not torch.isfinite(teacher_lp).all()
                or not torch.isfinite(teacher_tail_lp).all()):
            raise ValueError("nonfinite divergence input")
        p_top = teacher_lp.exp()
        p_tail = teacher_tail_lp.exp().unsqueeze(-1)
        teacher_mass = p_top.double().sum(-1, keepdim=True) + p_tail.double()
        if bool((torch.abs(teacher_mass - 1.0) > MASS_TOLERANCE).any()):
            raise ValueError("teacher top-k plus exact tail mass is inconsistent")

        # Do not derive the student tail as 1-sum(top-k).  For a peaked
        # distribution the FP32 top-k sum legitimately rounds to one while the
        # non-top-k mass is still positive; subtraction then creates an exact
        # zero bucket and NaN x*log(x) gradients.  Compute both support pieces
        # from the common full-vocabulary log normalizer instead.
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_logits = logits.gather(1, teacher_topk_ids)
        q_top = torch.exp(top_logits - log_z)
        non_top_logits = logits.clone()
        non_top_logits.scatter_(1, teacher_topk_ids, -torch.inf)
        q_tail = torch.exp(torch.logsumexp(non_top_logits, dim=-1, keepdim=True) - log_z)
        if not torch.isfinite(q_top).all() or not torch.isfinite(q_tail).all():
            raise ValueError("student bucket probabilities are nonfinite")
        # q_top and q_tail are separately exponentiated views of the same FP32
        # partition.  With 8192 gathered buckets their rounded sum can drift by
        # several e-6 even though the supports are exhaustive and disjoint.
        # Renormalize the derived student buckets instead of treating this
        # expected reduction error as malformed external probability data.
        q = torch.cat([q_top, q_tail], dim=-1)
        q_mass = q.sum(-1, keepdim=True)
        if not torch.isfinite(q_mass).all() or (q_mass <= 0).any():
            raise ValueError("student bucket mass must be finite and positive")
        if ((q_mass - 1.0).abs() > 1e-3).any():
            raise ValueError("student bucket partition drift exceeds sanity bound")
        q = q / q_mass
        p = torch.cat((p_top, p_tail), -1)
        if objective == "reverse_kl":
            per_token = _xlog_ratio(q, q, p).sum(-1)
        else:
            b = float(beta)
            mixture = b * p + (1.0 - b) * q
            per_token = b * _xlog_ratio(p, p, mixture).sum(-1)
            per_token = per_token + (1.0 - b) * _xlog_ratio(q, q, mixture).sum(-1)
        return per_token.mean(dtype=torch.float32)


def _validate_generation(row: Mapping[str, Any]) -> None:
    role = _nonempty_string(row.get("sampling_role"), "sampling_role")
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"unsupported sampling_role: {role}")
    config = row.get("generation_config")
    if type(config) is not dict:
        raise TypeError("generation_config must be an object")
    if set(config) != _UPSTREAM_GENERATION_FIELDS:
        raise ValueError("generation_config fields drift")
    phase = _nonempty_string(config["phase"], "generation_config.phase")
    temperature = _exact_number(config["temperature"], "generation_config.temperature")
    top_p = _exact_number(config["top_p"], "generation_config.top_p")
    max_tokens = _exact_int(config["max_tokens"], "generation_config.max_tokens", 1)
    n = _exact_int(config["n"], "generation_config.n", 1)
    seed_base = _exact_int(config["seed_base"], "generation_config.seed_base", 0)
    seed = _exact_int(config["seed"], "generation_config.seed", 0)
    if max_tokens != MAX_COMPLETION_TOKENS:
        raise ValueError("generation max_tokens must equal frozen 4096")
    if n != 1:
        raise ValueError("generation n must equal frozen one")
    expected_seed_base = _UPSTREAM_SEED_BASE_BY_TRACK.get(row.get("corpus_track"))
    if seed_base != expected_seed_base:
        raise ValueError("generation seed_base mismatch")
    if seed < seed_base:
        raise ValueError("greedy generation seed must be seed_base plus prompt ordinal")
    if abs(top_p - 1.0) > 1e-12:
        raise ValueError("generation top_p must equal frozen 1.0")
    expected_phase = "greedy" if role == "greedy" else role
    if phase != expected_phase:
        raise ValueError("sampling_role/generation phase mismatch")
    expected_temp = 0.0 if role == "greedy" else float(role.removeprefix("temp"))
    if abs(temperature - expected_temp) > 1e-12:
        raise ValueError("sampling_role/temperature mismatch")
    digest = _sha(row.get("generation_config_sha256"), "generation_config_sha256")
    if digest != canonical_sha256(config):
        raise ValueError("generation_config_sha256 mismatch")


def _safe_read_file(root: Path, relative: object, expected_sha: str, *, max_bytes: int) -> Tuple[Path, bytes]:
    rel_text = _nonempty_string(relative, "scores_file")
    rel = Path(rel_text)
    if rel.is_absolute() or ".." in rel.parts or rel == Path("."):
        raise ValueError("scores_file must be relative and traversal-free")
    root = root.resolve(strict=True)
    candidate = root / rel
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        raise ValueError(f"scores_file does not exist: {rel_text}")
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("scores_file symlink is forbidden")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("scores_file must be regular")
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise ValueError("scores_file size outside bounds")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("scores_file escapes bank root")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(candidate, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ValueError("scores_file changed during safe open")
        parts = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(8 << 20, remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(parts)
        if len(raw) != opened.st_size or len(raw) > max_bytes:
            raise ValueError("scores_file changed or exceeded size bound during read")
    finally:
        os.close(fd)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("scores_sha256 mismatch")
    return resolved, raw


def _load_score_tensors(raw: bytes) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except TypeError:  # older torch used only by local contract tests
        payload = torch.load(io.BytesIO(raw), map_location="cpu")
    if type(payload) is not dict or set(payload) != _REQUIRED_SCORE_KEYS:
        raise ValueError("score tensor payload fields drift")
    if any(type(value) is not torch.Tensor for value in payload.values()):
        raise TypeError("score payload values must be tensors")
    return payload


def validate_and_load_row(
    row: Mapping[str, Any],
    root: Path,
    *,
    min_topk: int = 32,
    max_score_file_bytes: int = MAX_SCORE_FILE_BYTES,
) -> LoadedRow:
    if type(row) is not dict:
        raise TypeError("bank row must be a JSON object")
    if row.get("schema") != ROW_SCHEMA:
        raise ValueError("bank row schema mismatch")
    for key in ("prompt_id", "sample_id", "teacher_model"):
        _nonempty_string(row.get(key), key)
    corpus_track = _nonempty_string(row.get("corpus_track"), "corpus_track")
    if corpus_track not in {"benchmark_distribution_mechanism", "tailfix_general_shippable"}:
        raise ValueError("unsupported corpus_track")
    if type(row.get("shippable")) is not bool:
        raise TypeError("shippable must be an exact JSON boolean")
    expected_shippable = corpus_track == "tailfix_general_shippable"
    if row["shippable"] is not expected_shippable:
        raise ValueError("corpus_track/shippable mismatch")
    _validate_generation(row)
    if _sha(row.get("student_checkpoint_sha256"), "student_checkpoint_sha256") != STUDENT_SHA256:
        raise ValueError("student checkpoint identity mismatch")
    if row.get("serve_fingerprint") != SERVE_FINGERPRINT:
        raise ValueError("serve fingerprint mismatch")
    for key in (
        "prompt_sha256", "split_sha256", "tokenizer_sha256",
        "teacher_model_sha256", "teacher_scorer_sha256", "scores_sha256",
    ):
        _sha(row.get(key), key)
    missing_arm_b_anchor = corpus_track == "tailfix_general_shippable" and row.get("arm_b_eligible") is False
    if missing_arm_b_anchor:
        if row.get("arm_a_eligible") is not True or row.get("shippable_training_eligible") is not True:
            raise ValueError("shippable missing-anchor row must remain Arm-A eligible")
        if row.get("benchmark_distribution_only") is not False:
            raise ValueError("shippable missing-anchor row cannot be benchmark-distribution-only")
        anchor_fields = (
            "fp_reference_completion_tokens", "fp_reference_sha256",
            "fp_reference_completion_sha256", "fp_reference_identity",
        )
        if any(row.get(key) is not None for key in anchor_fields):
            raise ValueError("missing Arm-B anchor fields must be explicit nulls")
    else:
        if corpus_track == "tailfix_general_shippable" and row.get("arm_b_eligible") is True:
            raise ValueError("Arm-B anchor is required when Arm-B eligibility is true")
        _sha(row.get("fp_reference_sha256"), "fp_reference_sha256")
    if row.get("score_alignment") != SCORE_ALIGNMENT:
        raise ValueError("score alignment contract mismatch")
    if row.get("score_payload_schema") != SCORE_PAYLOAD_SCHEMA:
        raise ValueError("score payload schema mismatch")
    if row.get("teacher_tail_logmass_preserved") is not True:
        raise ValueError("exact teacher tail-logmass preservation is required")

    token_ids = row.get("token_ids")
    if type(token_ids) is not list or not token_ids:
        raise TypeError("token_ids must be a nonempty JSON array")
    for index, token in enumerate(token_ids):
        _exact_int(token, f"token_ids[{index}]", 0)
    score_start = _exact_int(row.get("score_start"), "score_start", 1)
    if score_start >= len(token_ids):
        raise ValueError("score_start leaves empty scored suffix")
    n_score = len(token_ids) - score_start
    completion = _exact_int(row.get("completion_tokens"), "completion_tokens", 1)
    if completion != n_score:
        raise ValueError("completion length must equal scored suffix length")
    if completion > MAX_COMPLETION_TOKENS:
        raise ValueError("completion exceeds frozen 4096-token cap")
    if not missing_arm_b_anchor:
        _exact_int(row.get("fp_reference_completion_tokens"), "fp_reference_completion_tokens", 1)
    teacher_topk = _exact_int(row.get("teacher_topk"), "teacher_topk", min_topk)
    teacher_mean_nll = _exact_number(row.get("teacher_mean_nll"), "teacher_mean_nll")
    if teacher_mean_nll < 0:
        raise ValueError("teacher_mean_nll must be nonnegative")

    scores_sha = str(row["scores_sha256"])
    _path, raw = _safe_read_file(Path(root), row.get("scores_file"), scores_sha, max_bytes=max_score_file_bytes)
    payload = _load_score_tensors(raw)
    ids = payload["teacher_topk_ids"]
    topk_lp = payload["teacher_topk_logprobs"]
    target_lp = payload["teacher_target_logprobs"]
    tail_lp = payload["teacher_tail_logmass"]
    if ids.dtype != torch.long:
        raise TypeError("teacher_topk_ids must be int64")
    if ids.ndim != 2 or tuple(ids.shape) != (n_score, teacher_topk):
        raise ValueError("teacher_topk_ids shape mismatch")
    if (topk_lp.shape != ids.shape or target_lp.ndim != 1 or target_lp.shape[0] != n_score
            or tail_lp.ndim != 1 or tail_lp.shape[0] != n_score):
        raise ValueError("teacher score tensor shapes mismatch")
    topk_lp = topk_lp.float()
    target_lp = target_lp.float()
    tail_lp = tail_lp.float()
    if (not torch.isfinite(topk_lp).all() or not torch.isfinite(target_lp).all()
            or not torch.isfinite(tail_lp).all()):
        raise ValueError("teacher logprobs are nonfinite")
    if (bool((topk_lp > LOGPROB_TOLERANCE).any()) or bool((target_lp > LOGPROB_TOLERANCE).any())
            or bool((tail_lp > LOGPROB_TOLERANCE).any())):
        raise ValueError("full-softmax logprob cannot be positive")
    if bool((topk_lp[:, 1:] > topk_lp[:, :-1] + LOGPROB_TOLERANCE).any()):
        raise ValueError("teacher top-k logprobs must be descending")
    if any(torch.unique(ids[i]).numel() != teacher_topk for i in range(n_score)):
        raise ValueError("teacher top-k token IDs must be unique per position")
    teacher_mass = topk_lp.double().exp().sum(-1) + tail_lp.double().exp()
    if bool((torch.abs(teacher_mass - 1.0) > MASS_TOLERANCE).any()):
        raise ValueError("teacher top-k plus exact tail mass is inconsistent")

    targets = torch.tensor(token_ids[score_start:], dtype=torch.long)
    matches = ids.eq(targets[:, None])
    present = matches.any(-1)
    if bool(present.any()):
        gathered = topk_lp[present].gather(1, matches[present].long().argmax(-1, keepdim=True)).squeeze(1)
        if not torch.allclose(gathered, target_lp[present], atol=LOGPROB_TOLERANCE, rtol=0.0):
            raise ValueError("target/top-k logprob mismatch")
    absent = ~present
    if bool(absent.any()):
        kth = topk_lp[absent, -1]
        if bool((target_lp[absent] > kth + LOGPROB_TOLERANCE).any()):
            raise ValueError("omitted target exceeds kth top-k logprob")
    actual_nll = -float(target_lp.double().mean())
    if abs(actual_nll - teacher_mean_nll) > 2e-5:
        raise ValueError("teacher_mean_nll does not match exact target logprobs")

    return LoadedRow(dict(row), ids, topk_lp, target_lp, tail_lp, targets, n_score)


def validate_bank_rows(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    mode: str,
    min_rows: int,
    min_prompts: int,
    min_topk: int = 32,
) -> Dict[str, Any]:
    if mode not in {"rolling", "sealed"}:
        raise ValueError("bank mode must be rolling or sealed")
    _exact_int(min_rows, "min_rows", 1)
    _exact_int(min_prompts, "min_prompts", 1)
    if len(rows) < min_rows:
        raise ValueError("bank row count below gate")
    sample_ids = set()
    grouped = defaultdict(list)
    global_identity = None
    for row in rows:
        loaded = validate_and_load_row(row, root, min_topk=min_topk)
        item = loaded.row
        sample_id = item["sample_id"]
        if sample_id in sample_ids:
            raise ValueError("duplicate sample_id")
        sample_ids.add(sample_id)
        grouped[item["prompt_id"]].append(item)
        identity = (
            item["corpus_track"], item["shippable"], item["split_sha256"],
            item["tokenizer_sha256"], item["student_checkpoint_sha256"],
            item["serve_fingerprint"], item["teacher_model"], item["teacher_model_sha256"],
            item["teacher_scorer_sha256"], item["teacher_topk"], item["score_alignment"],
            item["score_payload_schema"], item["teacher_tail_logmass_preserved"],
        )
        if global_identity is None:
            global_identity = identity
        elif identity != global_identity:
            raise ValueError("bank global provenance/settings drift")
    if len(grouped) < min_prompts:
        raise ValueError("bank prompt count below gate")

    all_roles = Counter()
    sealed_temp_role = None
    sealed_temp_seed_set = None
    for prompt_id, prompt_rows in grouped.items():
        fp_refs = {(r["fp_reference_completion_tokens"], r["fp_reference_sha256"]) for r in prompt_rows}
        prompt_hashes = {r["prompt_sha256"] for r in prompt_rows}
        if len(fp_refs) != 1:
            raise ValueError(f"one fixed FP reference identity/length required for prompt {prompt_id}")
        if len(prompt_hashes) != 1:
            raise ValueError(f"prompt hash drift for prompt {prompt_id}")
        roles = Counter(r["sampling_role"] for r in prompt_rows)
        all_roles.update(roles)
        if mode == "sealed":
            temp_roles = set(roles) - {"greedy"}
            if roles.get("greedy") != 1 or len(temp_roles) != 1 or sum(roles[r] for r in temp_roles) != 3:
                raise ValueError(f"sealed bank composition requires one greedy plus three temp rows per prompt: {prompt_id}")
            temp_role = next(iter(temp_roles))
            seeds = {r["generation_config"]["seed"] for r in prompt_rows if r["sampling_role"] == temp_role}
            if len(seeds) != 3:
                raise ValueError(f"sealed temp seeds must be distinct for prompt {prompt_id}")
            if sealed_temp_role is None:
                sealed_temp_role, sealed_temp_seed_set = temp_role, seeds
            elif (temp_role, seeds) != (sealed_temp_role, sealed_temp_seed_set):
                raise ValueError("sealed sampling wave settings drift across prompts")
    if mode == "rolling" and set(all_roles) != {"greedy"}:
        raise ValueError("rolling first16 snapshot must contain greedy rows only")
    return {
        "format": "ptq-opd-bank-validation-v2",
        "mode": mode,
        "rows": len(rows),
        "prompts": len(grouped),
        "role_counts": dict(sorted(all_roles.items())),
        "corpus_track": rows[0]["corpus_track"],
        "shippable": rows[0]["shippable"],
        "student_checkpoint_sha256": STUDENT_SHA256,
        "serve_fingerprint": SERVE_FINGERPRINT,
        "score_alignment": SCORE_ALIGNMENT,
        "score_payload_schema": SCORE_PAYLOAD_SCHEMA,
        "teacher_tail_logmass_preserved": True,
        "validated": True,
    }


def load_bank(
    manifest_path: Path,
    *,
    mode: str,
    min_rows: int,
    min_prompts: int,
    min_topk: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = Path(manifest_path)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("manifest must be a nonsymlinked regular file")
    if info.st_size <= 0 or info.st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest size outside bounds")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("manifest changed during read")
    rows = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}: {exc}")
        rows.append(row)
    receipt = validate_bank_rows(rows, path.parent, mode=mode, min_rows=min_rows, min_prompts=min_prompts, min_topk=min_topk)
    receipt["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["manifest_bytes"] = len(raw)
    return rows, receipt


def rank_arm_b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        prompt_id = _nonempty_string(row.get("prompt_id"), "prompt_id")
        _nonempty_string(row.get("sample_id"), "sample_id")
        nll = _exact_number(row.get("teacher_mean_nll"), "teacher_mean_nll")
        completion = _exact_int(row.get("completion_tokens"), "completion_tokens", 1)
        fp_length = _exact_int(row.get("fp_reference_completion_tokens"), "fp_reference_completion_tokens", 1)
        fp_sha = _sha(row.get("fp_reference_sha256"), "fp_reference_sha256")
        if nll < 0:
            raise ValueError("teacher_mean_nll must be nonnegative")
        row["arm_b_score"] = nll + abs(completion / fp_length - 1.0)
        grouped[prompt_id].append(row)
    result = {}
    for prompt_id, candidates in grouped.items():
        refs = {(r["fp_reference_completion_tokens"], r["fp_reference_sha256"]) for r in candidates}
        if len(refs) != 1:
            raise ValueError(f"one fixed FP reference identity/length required for prompt {prompt_id}")
        result[prompt_id] = sorted(candidates, key=lambda r: (r["arm_b_score"], r["sample_id"]))
    return result


def static_kld_gate(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    required_classes: Sequence[str] = DEFAULT_REQUIRED_KLD_CLASSES,
    max_relative_regression: float = 0.01,
) -> Dict[str, Any]:
    required = tuple(required_classes)
    if len(required) != len(set(required)) or "global" not in required:
        raise ValueError("required_classes must be unique and include global")
    if set(baseline) != set(required) or set(candidate) != set(required):
        raise ValueError("static KLD class set must exactly match required global+all classes")
    if not 0.0 <= max_relative_regression <= 1.0:
        raise ValueError("max_relative_regression outside [0,1]")
    deltas = {}
    violations = {}
    for key in required:
        base = _exact_number(baseline[key], f"baseline[{key}]")
        cand = _exact_number(candidate[key], f"candidate[{key}]")
        if base < 0 or cand < 0:
            raise ValueError(f"negative static KLD for {key}")
        if base == 0:
            passed = cand == 0
            delta = 0.0 if passed else math.inf
        else:
            delta = cand / base - 1.0
            passed = delta <= max_relative_regression + 1e-12
        deltas[key] = delta
        if not passed:
            violations[key] = delta
    return {
        "format": "ptq-opd-static-kld-gate-v2",
        "passed": not violations,
        "max_relative_regression": max_relative_regression,
        "relative_delta_by_class": deltas,
        "violations": violations,
    }
