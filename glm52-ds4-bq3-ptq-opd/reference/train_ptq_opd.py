#!/usr/bin/env python3
"""Full GPU PTQ-OPD trainer for fresh and source-sealed continuation doses.

The script deliberately never emits PROMOTABLE.json. A static-pass checkpoint is
only a candidate for the separately instrumented same-fingerprint paired panel.
"""
from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import ptq_opd as G
import train_contracts as C

ROOT = Path(__file__).resolve().parent
OUTDIR = Path(os.path.expanduser(os.environ.get("PTQ_OPD_OUTDIR", str(ROOT / "out"))))
BANK_PATH = Path(os.path.expanduser(os.environ["PTQ_OPD_BANK"])).resolve()
BANK_MODE = os.environ.get("PTQ_OPD_BANK_MODE", "rolling")
MIN_BANK_ROWS = int(os.environ.get("PTQ_OPD_MIN_BANK_ROWS", "16"))
MIN_PROMPTS = int(os.environ.get("PTQ_OPD_MIN_PROMPTS", "16"))
TARGET = int(os.environ.get("PTQ_OPD_TARGET", "4"))
START_UPDATE = int(os.environ.get("PTQ_OPD_START_UPDATE", "0"))
START_CHECKPOINT = Path(os.path.expanduser(os.environ["PTQ_OPD_START_CHECKPOINT"])).resolve() if os.environ.get("PTQ_OPD_START_CHECKPOINT") else None
START_CHECKPOINT_SHA256 = os.environ.get("PTQ_OPD_START_CHECKPOINT_SHA256", "")
START_CANDIDATE_SHA256 = os.environ.get("PTQ_OPD_START_CANDIDATE_SHA256", "")
START_GATE = Path(os.path.expanduser(os.environ["PTQ_OPD_START_GATE"])).resolve() if os.environ.get("PTQ_OPD_START_GATE") else None
START_GATE_CANONICAL_SHA256 = os.environ.get("PTQ_OPD_START_GATE_CANONICAL_SHA256", "")
START_BANK_SHA256 = os.environ.get("PTQ_OPD_START_BANK_SHA256", "")
OBJECTIVE = os.environ.get("PTQ_OPD_OBJECTIVE", "jsd")
BETA = float(os.environ.get("PTQ_OPD_BETA", "0.5"))
ANCHOR_WEIGHT = float(os.environ.get("PTQ_OPD_ANCHOR_WEIGHT", "0.5"))
LOGIT_CHUNK = int(os.environ.get("PTQ_OPD_LOGIT_CHUNK", "512"))
ROLLOUT_SEGMENT_ROWS = int(os.environ.get("PTQ_OPD_ROLLOUT_SEGMENT_ROWS", "2"))
OWN_MICROBATCH = int(os.environ.get("PTQ_OPD_OWN_MICROBATCH", "2"))
SEED = int(os.environ.get("PTQ_OPD_SEED", "75382"))
SOURCE_SEAL_PATH = Path(os.path.expanduser(os.environ.get("PTQ_OPD_SOURCE_SEAL", str(ROOT / "SOURCE_SEAL.json"))))
LATEST = OUTDIR / "LATEST.pt"
STATUS = OUTDIR / "STATUS.json"
EVENTS = OUTDIR / "events.jsonl"
DOSE_LEDGER = OUTDIR / "DOSE_LEDGER.jsonl"
BASELINE_RECEIPT = OUTDIR / "STATIC_BASELINE.json"
GRADCHECK_RECEIPT = OUTDIR / "GRADCHECK.json"
LOCK_PATH = OUTDIR / "TRAIN.lock"
OOM_SENTINEL = OUTDIR / "OOM_SENTINEL.json"
EXPECTED_TRAINABLE_PARAMS = 1_855_147
EXPECTED_OPTIMIZER_ENTRIES = 343
DOSE_MILESTONES = (8, 12, 16)
STOP_REQUESTED = False
STOP_SIGNAL = 0


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(event: str, **fields: object) -> None:
    row = {"event": event, "ts": time.time(), **fields}
    with EVENTS.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(row, sort_keys=True), flush=True)


def status(state: str, **fields: object) -> None:
    value: Dict[str, object] = {}
    if STATUS.exists():
        try:
            value.update(json.loads(STATUS.read_text()))
        except Exception:
            pass
    value.update({"state": state, "updated_ts": time.time(), "host": os.uname().nodename, **fields})
    atomic_json(STATUS, value)


def _signal_handler(signum, _frame) -> None:
    # Signal context must remain flag-only: no locks, allocation, printing, or
    # file I/O. The main flow seals the receipt at an exact durable boundary.
    global STOP_REQUESTED, STOP_SIGNAL
    STOP_REQUESTED = True
    STOP_SIGNAL = signum


def stop_at_safe_boundary(next_update: int, phase: str) -> bool:
    if not STOP_REQUESTED:
        return False
    emit("signal_requested", signum=STOP_SIGNAL, phase=phase, next_update=next_update)
    status("stopped_at_safe_boundary", next_update=next_update, phase=phase, promotable=False)
    return True


def acquire_lock():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another Arm-A trainer holds the exclusive task lock")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} host={os.uname().nodename} started={time.time()}\n")
    handle.flush()
    return handle


def preflight_gpu_empty() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real PTQ-OPD trainer")
    command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi preflight failed: {result.stderr.strip()}")
    foreign = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if foreign:
        raise RuntimeError(f"foreign GPU processes present before model assembly: {foreign}")
    emit("gpu_preflight", gpu_empty=True, cuda_device=torch.cuda.get_device_name(0))


def mem_available_bytes() -> int:
    values = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0]) * 1024
    return values["MemAvailable"]


def start_memory_watchdog() -> threading.Thread:
    minimum = int(float(os.environ.get("PTQ_OPD_MIN_MEM_AVAILABLE_GB", "6")) * (1024 ** 3))

    def worker() -> None:
        consecutive = 0
        while True:
            try:
                available = mem_available_bytes()
                consecutive = consecutive + 1 if available < minimum else 0
                if consecutive >= 2:
                    receipt = {
                        "format": "ptq-opd-unified-memory-sentinel-v1",
                        "available_bytes": available,
                        "minimum_bytes": minimum,
                        "pid": os.getpid(),
                        "ts": time.time(),
                    }
                    atomic_json(OOM_SENTINEL, receipt)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as exc:
                atomic_json(OOM_SENTINEL, {"format": "ptq-opd-unified-memory-sentinel-v1", "watchdog_error": str(exc), "ts": time.time()})
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(5)

    thread = threading.Thread(target=worker, name="ptq-opd-mem-watchdog", daemon=True)
    thread.start()
    return thread


def import_integration():
    integration_dir = ROOT / "adapter"
    sys.path.insert(0, str(integration_dir))
    os.environ.setdefault("COMBO_BINREPAIR_BASE", str(integration_dir / "binrepair_e2e.py"))
    os.environ.setdefault("BR_BASE_HARNESS", str(integration_dir / "binrepair_base_e2e.py"))
    # Model checkpoints are intentionally not distributed in this repository.
    # Supply TAILFIX_START_CKPT from the prior BQ3 build.
    if not os.environ.get("TAILFIX_START_CKPT"):
        raise RuntimeError("TAILFIX_START_CKPT must point to the external BQ3 step0 checkpoint")
    os.environ.setdefault("TAILFIX_PLAN", str(ROOT / "plans" / "static_anchor_control.json"))
    os.environ.setdefault("TAILFIX_ARM", "control")
    spec = importlib.util.spec_from_file_location("ptq_opd_bq3_adapter", integration_dir / "tailfix_repair_e2e.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import sealed tailfix integration")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_identity() -> Dict[str, object]:
    seal = json.loads(SOURCE_SEAL_PATH.read_text())
    C.verify_source_seal(ROOT, seal)
    return {
        "source_seal_sha256": seal["seal_sha256"],
        "source_seal_path": str(SOURCE_SEAL_PATH),
        "student_checkpoint_sha256": G.STUDENT_SHA256,
    }


def optimizer_step_readback(
    checkpoint: Mapping[str, Any], expected_update: int, expected_entries: int
) -> Dict[str, int]:
    state = checkpoint.get("optimizer", {}).get("state", {})
    if len(state) != expected_entries:
        raise RuntimeError(
            f"optimizer entry count mismatch: {len(state)} != {expected_entries}"
        )
    steps = []
    for entry in state.values():
        value = entry.get("step")
        if isinstance(value, torch.Tensor):
            value = value.item()
        if type(value) not in (int, float) or int(value) != value:
            raise RuntimeError("optimizer step is not an exact integer")
        steps.append(int(value))
    if min(steps) != expected_update or max(steps) != expected_update:
        raise RuntimeError(
            f"optimizer step mismatch: min={min(steps)} max={max(steps)} expected={expected_update}"
        )
    return {
        "optimizer_state_entries": len(state),
        "optimizer_step_min": min(steps),
        "optimizer_step_max": max(steps),
    }


def load_start_lineage() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """CPU-read and bind an exact gate-sealed parent state and Adam."""
    if START_UPDATE == 0:
        if any((START_CHECKPOINT, START_CHECKPOINT_SHA256, START_CANDIDATE_SHA256,
                START_GATE, START_GATE_CANONICAL_SHA256, START_BANK_SHA256)):
            raise RuntimeError("start-lineage fields are forbidden for a step0 run")
        return None, None
    if (START_UPDATE, TARGET) not in {(4, 8), (8, 16)}:
        raise RuntimeError("sealed continuation must be exactly step4→step8 or step8→step16")
    if START_CHECKPOINT is None or START_GATE is None:
        raise RuntimeError("continuation requires checkpoint and static-gate paths")
    if not START_CHECKPOINT.is_file() or not START_GATE.is_file():
        raise RuntimeError("continuation lineage file is missing")
    actual_checkpoint_sha = sha256_file(START_CHECKPOINT)
    if actual_checkpoint_sha != START_CHECKPOINT_SHA256:
        raise RuntimeError("gate-bound parent LATEST byte SHA mismatch")
    checkpoint = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    required = {
        "format": "ptq-opd-train-v2",
        "next_update": START_UPDATE,
        "last_passed_milestone": START_UPDATE,
        "objective": OBJECTIVE,
        "beta": BETA,
        "anchor_weight": ANCHOR_WEIGHT,
        "trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "promotable": False,
        "paired_gate_required": True,
    }
    bad = {key: (checkpoint.get(key), expected) for key, expected in required.items()
           if checkpoint.get(key) != expected}
    if bad:
        raise RuntimeError(f"parent checkpoint header mismatch: {sorted(bad)}")
    checkpoint_bank_sha = checkpoint.get("identity", {}).get("bank_manifest_sha256")
    if checkpoint_bank_sha != START_BANK_SHA256:
        raise RuntimeError("parent continuation bank SHA mismatch")
    if not checkpoint.get("state") or not checkpoint.get("optimizer", {}).get("state"):
        raise RuntimeError("parent checkpoint lacks trained state or Adam lineage")
    adam = optimizer_step_readback(checkpoint, START_UPDATE, EXPECTED_OPTIMIZER_ENTRIES)
    gate = json.loads(START_GATE.read_text())
    if C.canonical_sha256(gate) != START_GATE_CANONICAL_SHA256:
        raise RuntimeError("parent static-gate canonical SHA mismatch")
    if (gate.get("format") != "ptq-opd-static-gate-receipt-v2" or gate.get("milestone") != START_UPDATE
            or gate.get("candidate_sha256") != START_CANDIDATE_SHA256
            or gate.get("passed") is not True or gate.get("decision", {}).get("passed") is not True):
        raise RuntimeError("parent static gate does not pass and bind the immutable candidate")
    if checkpoint.get("candidate_sha256") != START_CANDIDATE_SHA256:
        raise RuntimeError("gate-bound parent LATEST candidate binding mismatch")
    if checkpoint.get("gate_receipt_sha256") != START_GATE_CANONICAL_SHA256:
        raise RuntimeError("gate-bound parent LATEST gate binding mismatch")
    lineage = {
        "format": f"ptq-opd-step{START_UPDATE}-to-step{TARGET}-lineage-v1",
        "parent_latest": str(START_CHECKPOINT),
        "parent_latest_sha256": actual_checkpoint_sha,
        "parent_candidate_sha256": START_CANDIDATE_SHA256,
        "parent_static_gate": str(START_GATE),
        "parent_static_gate_canonical_sha256": START_GATE_CANONICAL_SHA256,
        "parent_bank_sha256": START_BANK_SHA256,
        "parent_distinct_rows": 16,
        "parent_optimizer_updates": START_UPDATE,
        **adam,
        "parent_lineage": checkpoint.get("lineage", {}),
        "state_optimizer_loaded_without_reset": True,
    }
    return checkpoint, {"gate": gate, "lineage": lineage, "adam_readback": adam}


def assemble(I):
    I.B.T.TrainableExperts = I.B.K4096Experts
    I.B.T.PILOT = tuple(I.EXPECTED_LAYERS)
    student = I.B.T.Student()
    norms, outputs = I.expose_combo_parameters(student)
    codebooks = I.codebook_params(student)
    _baseline, seed_info = I.validate_and_load_seeds(student, norms, outputs)
    norm_params = [parameter for _name, _module, parameter in norms]
    output_params = [parameter for _name, _module, parameter in outputs]
    groups = {"codebooks": codebooks, "norms": norm_params, "outputs": output_params}
    count = sum(parameter.numel() for values in groups.values() for parameter in values)
    if count != EXPECTED_TRAINABLE_PARAMS:
        raise RuntimeError(f"trainable surface mismatch: {count}")
    emit(
        "assembled",
        trainable_params=count,
        codebook_params=sum(p.numel() for p in codebooks),
        norm_params=sum(p.numel() for p in norm_params),
        output_params=sum(p.numel() for p in output_params),
        seed_info=seed_info,
    )
    return student, norms, outputs, groups


def group_grad_norms(groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> Dict[str, float]:
    result = {}
    for name, parameters in groups.items():
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                value = float(parameter.grad.detach().float().norm())
                total += value * value
        result[name] = math.sqrt(total)
    return result


def clear_gradients(groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> None:
    for parameters in groups.values():
        for parameter in parameters:
            parameter.grad = None


def evict_actcache(acache, wins: Optional[Sequence[int]] = None, *, reason: str) -> int:
    """Drop completed activation-cache tensors while retaining disk cache files."""
    if type(reason) is not str or not reason:
        raise ValueError("activation-cache eviction reason must be nonempty")
    if acache is None or not hasattr(acache, "mem"):
        return 0
    keys = list(acache.mem) if wins is None else list(wins)
    released = 0
    for key in keys:
        if key in acache.mem:
            value = acache.mem.pop(key)
            del value
            released += 1
    if released:
        torch.cuda.empty_cache()
    return released


def _right_padded_own_ids(I, loaded_rows: Sequence[G.LoadedRow]) -> torch.Tensor:
    """Build a causal batch: zero padding is strictly after each real sequence."""
    if not loaded_rows:
        raise RuntimeError("cannot pad an empty own-rollout microbatch")
    max_len = max(len(loaded.row["token_ids"]) for loaded in loaded_rows)
    ids = torch.zeros((len(loaded_rows), max_len), dtype=torch.long, device=I.B.DEV)
    for batch_index, loaded in enumerate(loaded_rows):
        values = torch.tensor(loaded.row["token_ids"], dtype=torch.long, device=I.B.DEV)
        ids[batch_index, : values.numel()] = values
    return ids


def _own_microbatch_backward(
    I, student, loaded_rows: Sequence[G.LoadedRow], *, scale: float, total_rows: int
) -> List[Dict[str, object]]:
    """One right-padded suffix walk and one summed backward for 2 or 4 rows."""
    ids = _right_padded_own_ids(I, loaded_rows)
    embeds = student.model.model.embed_tokens(ids)
    hidden = embeds.unsqueeze(2).expand(-1, -1, student.config.hc_mult, -1).contiguous()
    del embeds
    hidden = I.B.fast_forward(student, hidden, ids, True)
    weighted_losses = []
    per_row = []
    for batch_index, loaded in enumerate(loaded_rows):
        token_ids = loaded.row["token_ids"]
        score_start = loaded.row["score_start"]
        score_hidden = hidden[batch_index, score_start - 1 : len(token_ids) - 1]
        weighted_value = 0.0
        n = loaded.n_score_tokens
        for lo in range(0, n, LOGIT_CHUNK):
            hi = min(n, lo + LOGIT_CHUNK)
            logits = student.model.lm_head(score_hidden[lo:hi].to(torch.bfloat16))
            loss = G.bucketed_divergence(
                logits,
                loaded.teacher_topk_ids[lo:hi].to(I.B.DEV),
                loaded.teacher_topk_logprobs[lo:hi].to(I.B.DEV),
                loaded.teacher_tail_logmass[lo:hi].to(I.B.DEV),
                objective=OBJECTIVE,
                beta=BETA,
            )
            weighted = loss * (scale * ((hi - lo) / n) / total_rows)
            weighted_value += float(weighted.detach())
            weighted_losses.append(weighted)
            del logits, loss
        per_row.append({
            "prompt_id": loaded.row["prompt_id"],
            "sample_id": loaded.row["sample_id"],
            "score_tokens": n,
            "weighted_divergence": weighted_value,
        })
    if not weighted_losses:
        raise RuntimeError("own-rollout microbatch produced no scored loss")
    torch.stack(weighted_losses).sum().backward()
    del weighted_losses, hidden, ids
    return per_row


def own_divergence_backward(
    I, student, rows: Sequence[Mapping[str, Any]], *, scale: float, update_number: int
) -> Dict[str, object]:
    if not rows:
        raise RuntimeError("empty own-rollout update batch")
    if OWN_MICROBATCH not in {2, 4}:
        raise RuntimeError("PTQ_OPD_OWN_MICROBATCH must be 4, or 2 for long-row memory safety")
    loaded_rows = [G.validate_and_load_row(dict(row), BANK_PATH.parent) for row in rows]
    per_row = []
    segment_receipts = []
    segments = C.segment_rows(loaded_rows, OWN_MICROBATCH)
    for segment_index, segment in enumerate(segments, 1):
        segment_started = time.time()
        emit(
            "rollout_segment_start", update=update_number, segment=segment_index,
            segments=len(segments), rows=len(segment), rows_completed=len(per_row),
            rows_total=len(rows), max_segment_rows=OWN_MICROBATCH,
            own_microbatch=OWN_MICROBATCH,
        )
        for offset, loaded in enumerate(segment, 1):
            emit(
                "own_row_start", update=update_number, ordinal=len(per_row) + offset,
                rows_total=len(rows), segment=segment_index,
                prompt_id=loaded.row.get("prompt_id"), sample_id=loaded.row.get("sample_id"),
                score_tokens=loaded.n_score_tokens, batched=True,
            )
        batch_rows = _own_microbatch_backward(
            I, student, segment, scale=scale, total_rows=len(loaded_rows)
        )
        batch_seconds = time.time() - segment_started
        for row_receipt in batch_rows:
            per_row.append({**row_receipt, "seconds": batch_seconds})
            emit(
                "own_row_complete", update=update_number, ordinal=len(per_row),
                rows_total=len(rows), segment=segment_index, max_segment_rows=OWN_MICROBATCH,
                own_microbatch=OWN_MICROBATCH, batched=True, seconds=batch_seconds,
                cuda_memory_allocated=int(torch.cuda.memory_allocated()),
                cuda_max_memory_allocated=int(torch.cuda.max_memory_allocated()),
                **row_receipt,
            )
        segment_receipt = {
            "update": update_number, "segment": segment_index, "segments": len(segments),
            "rows": len(segment), "rows_completed": len(per_row), "rows_total": len(rows),
            "max_segment_rows": OWN_MICROBATCH, "own_microbatch": OWN_MICROBATCH,
            "seconds": batch_seconds,
        }
        segment_receipts.append(segment_receipt)
        emit("rollout_segment_complete", **segment_receipt)
        gc.collect()
        torch.cuda.empty_cache()
    del loaded_rows
    return {
        "rows": per_row, "segments": segment_receipts,
        "weighted_mean_sum": sum(r["weighted_divergence"] for r in per_row),
        "suffix_walks": len(segments), "microbatch": OWN_MICROBATCH,
    }


def anchor_backward(I, student, corpus, acache, wins: Sequence[int]) -> Dict[str, object]:
    consumed_wins = list(wins)
    loss, stats = I.tailfix_batch_loss(student, corpus, acache, consumed_wins, True)
    weighted = loss * ANCHOR_WEIGHT
    value = float(loss.detach())
    weighted.backward()
    del loss, weighted
    released = evict_actcache(acache, consumed_wins, reason="anchor_backward_complete")
    return {
        "anchor_mean_kld": value,
        "anchor_weight": ANCHOR_WEIGHT,
        "wins": consumed_wins,
        "actcache_windows_released": released,
        "stats": stats,
    }


def run_gradcheck(I, student, corpus, acache, groups, rows, identity) -> Dict[str, object]:
    if GRADCHECK_RECEIPT.exists():
        receipt = json.loads(GRADCHECK_RECEIPT.read_text())
        if receipt.get("bank_manifest_sha256") != identity["bank_manifest_sha256"] or receipt.get("source_seal_sha256") != identity["source_seal_sha256"]:
            raise RuntimeError("existing gradcheck receipt identity mismatch")
        return receipt
    if LATEST.exists():
        raise RuntimeError("LATEST exists before required no-optimizer gradcheck receipt")
    clear_gradients(groups)
    anchor = anchor_backward(I, student, corpus, acache, I.TRAINING_SCHEDULE[0])
    own = own_divergence_backward(I, student, rows[:1], scale=1.0, update_number=START_UPDATE)
    norms = group_grad_norms(groups)
    if any(not math.isfinite(value) or value <= 0.0 for value in norms.values()):
        raise RuntimeError(f"real gradcheck missing/nonfinite group gradient: {norms}")
    receipt = {
        "format": "ptq-opd-real-no-optimizer-gradcheck-v2",
        "passed": True,
        "optimizer_created": False,
        "promotable_checkpoint_created": False,
        "trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "group_grad_norms": norms,
        "anchor": anchor,
        "own_rollout": own,
        **identity,
        "ts": time.time(),
    }
    atomic_json(GRADCHECK_RECEIPT, receipt)
    clear_gradients(groups)
    torch.cuda.empty_cache()
    emit("gradcheck_pass", receipt=str(GRADCHECK_RECEIPT), group_grad_norms=norms)
    return receipt


def static_summary(I, student, corpus, acache) -> Dict[str, object]:
    by_class = defaultdict(list)
    all_values = []
    per_window = {}
    for win in I.B.PROBE_WINS:
        try:
            values = I.heldout_token_values(student, corpus, acache, win).double().cpu()
        finally:
            evict_actcache(acache, [win], reason="static_probe_copied_to_cpu")
        if not torch.isfinite(values).all():
            raise RuntimeError(f"nonfinite static KLD values for win {win}")
        cls = I.PROBE_CLASS_BY_WIN[int(win)]
        by_class[cls].append(values)
        all_values.append(values)
        per_window[str(win)] = {
            "class": cls,
            "mean": float(values.mean()),
            "min": float(values.min()),
            "negative_positions": int((values < 0).sum()),
            "positions": int(values.numel()),
        }
        del values
        torch.cuda.empty_cache()
    classes = {cls: float(torch.cat(parts).mean()) for cls, parts in sorted(by_class.items())}
    metrics = {"global": float(torch.cat(all_values).mean()), **classes}
    bad_metrics = {key: value for key, value in metrics.items() if not math.isfinite(value) or value < 0.0}
    if bad_metrics:
        raise RuntimeError(f"nonphysical static KLD metric: {bad_metrics}")
    return {
        "instrument": "exact heldout spot16 scorer KLD on teacher top-8192 support",
        "n_windows": len(per_window),
        "metrics": metrics,
        "per_window": per_window,
    }


def save_torch_atomic(path: Path, payload: object) -> None:
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(payload, tmp)
        # torch.save closes its path-owned descriptor before returning. Reopen
        # the completed temporary file and force its bytes to stable storage
        # before making the checkpoint name visible.
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # The rename itself is metadata. Persist the parent directory so a
        # power loss cannot roll LATEST.pt back to the previous optimizer
        # boundary after an optimizer_update event has been emitted.
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp.exists():
            tmp.unlink()


def append_dose_ledger(value: Mapping[str, Any]) -> None:
    with DOSE_LEDGER.open("a") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def static_dose_receipt(
    I, student, corpus, acache, baseline_receipt: Mapping[str, Any], baseline_sha: str,
    milestone: int, candidate_path: Path, candidate_sha: str,
) -> Dict[str, Any]:
    path = OUTDIR / f"STATIC_DOSE_STEP{milestone}.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        if (receipt.get("milestone") != milestone
                or receipt.get("candidate_sha256") != candidate_sha
                or receipt.get("baseline_receipt_sha256") != baseline_sha):
            raise RuntimeError(f"existing static dose receipt drift at step{milestone}")
        return receipt
    summary = static_summary(I, student, corpus, acache)
    required_classes = tuple(["global"] + sorted(I.TF.CLASSES))
    decision = G.static_kld_gate(
        baseline_receipt["metrics"], summary["metrics"], required_classes=required_classes
    )
    receipt = {
        "format": "ptq-opd-static-dose-receipt-v1",
        "milestone": milestone,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "baseline_receipt_sha256": baseline_sha,
        "required_classes": list(required_classes),
        "baseline_metrics": baseline_receipt["metrics"],
        "candidate_metrics": summary["metrics"],
        "decision": decision,
        "passed": decision["passed"],
        "static": summary,
        "ts": time.time(),
    }
    atomic_json(path, receipt)
    emit(
        "static_dose_sealed", milestone=milestone, receipt=str(path),
        receipt_sha256=sha256_file(path), candidate_sha256=candidate_sha,
        metrics=summary["metrics"], passed=decision["passed"],
    )
    return receipt


def checkpoint_payload(
    I, student, norms, outputs, optimizer, identity, lineage, baseline_sha,
    next_update, last_passed, gate, candidate_sha,
) -> Dict[str, object]:
    return {
        "format": "ptq-opd-train-v2",
        "identity": identity,
        "lineage": lineage,
        "baseline_receipt_sha256": baseline_sha,
        "objective": OBJECTIVE,
        "beta": BETA,
        "anchor_weight": ANCHOR_WEIGHT,
        "trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "next_update": next_update,
        "last_passed_milestone": last_passed,
        "gate": gate,
        "gate_receipt_sha256": None if gate is None else C.canonical_sha256(gate),
        "candidate_sha256": candidate_sha,
        "state": I.state_named(student, norms, outputs),
        "optimizer": optimizer.state_dict(),
        "promotable": False,
        "paired_gate_required": True,
        "saved_ts": time.time(),
        "host": os.uname().nodename,
    }


def validate_resume(
    checkpoint: Mapping[str, Any], identity: Mapping[str, Any], lineage: Mapping[str, Any], baseline_sha: str
) -> None:
    expected = {
        "format": "ptq-opd-train-v2",
        "identity": dict(identity),
        "lineage": dict(lineage),
        "baseline_receipt_sha256": baseline_sha,
        "objective": OBJECTIVE,
        "beta": BETA,
        "anchor_weight": ANCHOR_WEIGHT,
        "trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "promotable": False,
        "paired_gate_required": True,
    }
    bad = {key: (checkpoint.get(key), value) for key, value in expected.items() if checkpoint.get(key) != value}
    if bad:
        raise RuntimeError(f"resume identity mismatch: {sorted(bad)}")


def select_rows(rows: Sequence[Mapping[str, Any]], update_number: int, target: int) -> List[Mapping[str, Any]]:
    selected = [row for index, row in enumerate(rows) if index % target == update_number - 1]
    if not selected:
        selected = [rows[(update_number - 1) % len(rows)]]
    return selected


def main() -> None:
    if OBJECTIVE not in {"jsd", "reverse_kl"} or BETA != 0.5 or ANCHOR_WEIGHT != 0.5:
        raise ValueError("binding Arm-A contract requires JSD/reverse-KL beta=.5 and anchor weight=.5")
    if LOGIT_CHUNK < 16 or LOGIT_CHUNK > 2048:
        raise ValueError("PTQ_OPD_LOGIT_CHUNK outside [16,2048]")
    if ROLLOUT_SEGMENT_ROWS not in (1, 2):
        raise ValueError("PTQ_OPD_ROLLOUT_SEGMENT_ROWS must be exactly 1 or 2")
    if OWN_MICROBATCH not in (2, 4):
        raise ValueError("PTQ_OPD_OWN_MICROBATCH must be exactly 2 or 4")
    random.seed(SEED)
    torch.manual_seed(SEED)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    lock_handle = acquire_lock()
    source = source_identity()
    start_checkpoint, start_meta = load_start_lineage()
    lineage = {} if start_meta is None else dict(start_meta["lineage"])
    if start_meta is not None:
        emit(
            "parent_step8_cpu_readback", checkpoint_sha256=START_CHECKPOINT_SHA256,
            candidate_sha256=START_CANDIDATE_SHA256, **start_meta["adam_readback"],
        )
    rows, bank_receipt = G.load_bank(
        BANK_PATH, mode=BANK_MODE, min_rows=MIN_BANK_ROWS, min_prompts=MIN_PROMPTS
    )
    expected_track = os.environ.get("PTQ_OPD_EXPECT_CORPUS_TRACK", "benchmark_distribution_mechanism")
    if bank_receipt["corpus_track"] != expected_track:
        raise RuntimeError("bank corpus track does not match this sealed run")
    if expected_track == "benchmark_distribution_mechanism" and bank_receipt["shippable"] is not False:
        raise RuntimeError("mechanism bank must be explicitly non-shippable")
    if START_UPDATE:
        if len(rows) != 16 or len({row["sample_id"] for row in rows}) != 16:
            raise RuntimeError("continuation requires exactly 16 distinct shard rows")
        update_numbers = list(range(START_UPDATE + 1, TARGET + 1))
        schedule = [
            C.campaign_continuation_rows(
                rows, update, start_update=START_UPDATE, target_update=TARGET
            )
            for update in update_numbers
        ]
        block_schedules = [schedule] if START_UPDATE == 4 else [schedule[:4], schedule[4:]]
        block_coverage = []
        expected_ids = sorted(row["sample_id"] for row in rows)
        for block_index, block in enumerate(block_schedules, 1):
            scheduled_ids = [row["sample_id"] for part in block for row in part]
            if sorted(scheduled_ids) != expected_ids or len(scheduled_ids) != len(set(scheduled_ids)):
                raise RuntimeError(f"deep-dose block {block_index} does not cover shard exactly once")
            block_coverage.append({
                "block": block_index,
                "updates": update_numbers[(block_index - 1) * 4:block_index * 4],
                "sample_ids": scheduled_ids,
                "distinct_rows": len(set(scheduled_ids)),
            })
        lineage.update({
            "continuation_bank_sha256": bank_receipt["manifest_sha256"],
            "continuation_distinct_rows": len(rows),
            "continuation_sample_ids": [row["sample_id"] for row in rows],
            "continuation_row_sha256": [row["row_sha256"] for row in rows],
            "continuation_score_sha256": [row["scores_sha256"] for row in rows],
            "continuation_score_tokens": [row["n_score_tokens"] for row in rows],
            "dose_block_coverage": block_coverage,
            "target_optimizer_updates": TARGET,
            "target_total_row_uses": 16 if START_UPDATE == 4 else 32,
        })
    identity = {
        **source,
        "bank_manifest": str(BANK_PATH),
        "bank_manifest_sha256": bank_receipt["manifest_sha256"],
        "bank_mode": BANK_MODE,
        "bank_rows": bank_receipt["rows"],
        "bank_prompts": bank_receipt["prompts"],
        "serve_fingerprint": G.SERVE_FINGERPRINT,
        "continuation_start_update": START_UPDATE,
        "continuation_target_update": TARGET,
    }
    status("validated_bank", target=TARGET, start_update=START_UPDATE, promotable=False,
           identity=identity, lineage=lineage, bank_receipt=bank_receipt)
    emit("bank_validated", receipt=bank_receipt, lineage=lineage)
    preflight_gpu_empty()
    start_memory_watchdog()
    I = import_integration()
    student, norms, outputs, groups = assemble(I)
    corpus = I.B.T.load_corpus()
    acache = I.B.ActCache(student)

    checkpoint = torch.load(LATEST, map_location="cpu", weights_only=False) if LATEST.exists() else None
    if checkpoint is None and start_checkpoint is not None:
        I.load_named(student, norms, outputs, start_checkpoint["state"])
        emit("parent_state_loaded", parent_update=START_UPDATE, lineage=lineage)

    # No optimizer exists before this real full-model receipt is sealed.
    run_gradcheck(I, student, corpus, acache, groups, rows, identity)
    if stop_at_safe_boundary(START_UPDATE, "no_optimizer_gradcheck"):
        return
    optimizer = torch.optim.Adam(
        [p for values in groups.values() for p in values],
        lr=C.lr_for_update(max(1, START_UPDATE + 1)),
    )

    if checkpoint is not None:
        if not BASELINE_RECEIPT.exists():
            raise RuntimeError("resume checkpoint is missing frozen static baseline receipt")
        baseline_sha = sha256_file(BASELINE_RECEIPT)
        validate_resume(checkpoint, identity, lineage, baseline_sha)
        I.load_named(student, norms, outputs, checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        emit("resumed", next_update=checkpoint["next_update"],
             last_passed_milestone=checkpoint["last_passed_milestone"], lineage=lineage)
    else:
        if BASELINE_RECEIPT.exists():
            raise RuntimeError("static baseline exists without a bound checkpoint; refusing ambiguous restart")
        if start_checkpoint is not None:
            optimizer.load_state_dict(start_checkpoint["optimizer"])
            loaded_adam = optimizer_step_readback(
                {"optimizer": optimizer.state_dict()}, START_UPDATE, EXPECTED_OPTIMIZER_ENTRIES
            )
            emit("parent_optimizer_loaded_without_reset", parent_update=START_UPDATE, **loaded_adam)
            parent_gate = dict(start_meta["gate"])
            candidate_sha = START_CANDIDATE_SHA256
            last_passed = START_UPDATE
        else:
            parent_gate = None
            candidate_sha = None
            last_passed = 0
        baseline = static_summary(I, student, corpus, acache)
        baseline_receipt = {
            "format": "ptq-opd-static-baseline-v2",
            "student_checkpoint_sha256": G.STUDENT_SHA256,
            "evaluated_checkpoint_sha256": START_CHECKPOINT_SHA256 or G.STUDENT_SHA256,
            "bank_manifest_sha256": identity["bank_manifest_sha256"],
            "source_seal_sha256": identity["source_seal_sha256"],
            "lineage": lineage,
            **baseline,
            "ts": time.time(),
        }
        atomic_json(BASELINE_RECEIPT, baseline_receipt)
        baseline_sha = sha256_file(BASELINE_RECEIPT)
        checkpoint = checkpoint_payload(
            I, student, norms, outputs, optimizer, identity, lineage, baseline_sha,
            START_UPDATE, last_passed, parent_gate, candidate_sha,
        )
        save_torch_atomic(LATEST, checkpoint)
        emit("static_baseline_sealed", receipt=str(BASELINE_RECEIPT), sha256=baseline_sha,
             metrics=baseline["metrics"], lineage=lineage)

    action = C.validate_transition(checkpoint, TARGET)
    if action == "done":
        status("static_pass_waiting_paired_panel", next_update=checkpoint["next_update"], promotable=False)
        return

    if action == "train":
        start = int(checkpoint["next_update"])
        last_passed = int(checkpoint["last_passed_milestone"])
        gate = checkpoint.get("gate")
        candidate_sha = checkpoint.get("candidate_sha256")
        baseline_receipt = json.loads(BASELINE_RECEIPT.read_text())
        if start >= 12 and not (OUTDIR / "STATIC_DOSE_STEP12.json").exists():
            if start != 12 or not (OUTDIR / "CANDIDATE_STEP12.pt").exists():
                raise RuntimeError("cannot reconstruct missing step12 static receipt after later updates")
            step12_path = OUTDIR / "CANDIDATE_STEP12.pt"
            static_dose_receipt(
                I, student, corpus, acache, baseline_receipt, baseline_sha,
                12, step12_path, sha256_file(step12_path),
            )
        for update_index in range(start, TARGET):
            update_started = time.time()
            if stop_at_safe_boundary(update_index, "optimizer_safe_boundary"):
                return
            update_number = update_index + 1
            lr = C.lr_for_update(update_number)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            anchor = anchor_backward(I, student, corpus, acache, I.TRAINING_SCHEDULE[update_index % len(I.TRAINING_SCHEDULE)])
            selected = (
                C.campaign_continuation_rows(
                    rows, update_number, start_update=START_UPDATE, target_update=TARGET
                )
                if START_UPDATE else select_rows(rows, update_number, TARGET)
            )
            own = own_divergence_backward(
                I, student, selected, scale=1.0, update_number=update_number
            )
            grad_norms = group_grad_norms(groups)
            if any(not math.isfinite(value) or value <= 0 for value in grad_norms.values()):
                raise RuntimeError(f"nonfinite/zero update gradients: {grad_norms}")
            if stop_at_safe_boundary(update_index, "pre_optimizer_step"):
                clear_gradients(groups)
                return
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            next_update = update_number
            payload = checkpoint_payload(
                I, student, norms, outputs, optimizer, identity, lineage, baseline_sha,
                next_update, last_passed, gate, candidate_sha,
            )
            snapshot_path = OUTDIR / f"CHECKPOINT_STEP{next_update}.pt"
            if snapshot_path.exists():
                raise RuntimeError(f"immutable step{next_update} snapshot already exists")
            save_torch_atomic(snapshot_path, payload)
            snapshot_sha = sha256_file(snapshot_path)
            save_torch_atomic(LATEST, payload)
            checkpoint = payload
            latest_sha = sha256_file(LATEST)
            milestone_candidate_path = None
            milestone_candidate_sha = None
            if next_update in DOSE_MILESTONES:
                milestone_candidate_path = OUTDIR / f"CANDIDATE_STEP{next_update}.pt"
                if milestone_candidate_path.exists():
                    raise RuntimeError(f"immutable step{next_update} candidate already exists")
                save_torch_atomic(milestone_candidate_path, payload)
                milestone_candidate_sha = sha256_file(milestone_candidate_path)
                emit(
                    "candidate_checkpoint", step=next_update,
                    path=str(milestone_candidate_path), sha256=milestone_candidate_sha,
                    promotable=False,
                )
            ledger_row = {
                "format": "ptq-opd-deep-dose-ledger-v1",
                "update": next_update,
                "dose_block": 1 if next_update <= 12 else 2,
                "checkpoint": str(LATEST),
                "checkpoint_sha256": latest_sha,
                "snapshot": str(snapshot_path),
                "snapshot_sha256": snapshot_sha,
                "candidate": None if milestone_candidate_path is None else str(milestone_candidate_path),
                "candidate_sha256": milestone_candidate_sha,
                "row_ids": [row["sample_id"] for row in selected],
                "row_sha256": [row["row_sha256"] for row in selected],
                "seconds": time.time() - update_started,
                "lr": lr,
                "anchor_mean_kld": anchor["anchor_mean_kld"],
                "own_weighted_mean_sum": own["weighted_mean_sum"],
                "group_grad_norms": grad_norms,
                "ts": time.time(),
            }
            append_dose_ledger(ledger_row)
            emit("optimizer_update", update=next_update, checkpoint_sha256=latest_sha,
                 lr=lr, anchor=anchor, own=own, group_grad_norms=grad_norms)
            status(
                "training", next_update=next_update, target=TARGET, lr=lr,
                checkpoint_sha256=latest_sha, group_grad_norms=grad_norms, promotable=False,
            )
            if next_update in DOSE_MILESTONES:
                assert milestone_candidate_path is not None and milestone_candidate_sha is not None
                static_dose_receipt(
                    I, student, corpus, acache, baseline_receipt, baseline_sha,
                    next_update, milestone_candidate_path, milestone_candidate_sha,
                )
        action = "gate"

    if action == "gate":
        candidate_path = OUTDIR / f"CANDIDATE_STEP{TARGET}.pt"
        if not candidate_path.exists():
            # Interrupted after target weights were saved in LATEST but before the
            # immutable candidate copy: reconstruct the copy without training.
            save_torch_atomic(candidate_path, checkpoint)
        candidate_sha = sha256_file(candidate_path)
        baseline_receipt = json.loads(BASELINE_RECEIPT.read_text())
        dose_receipt = static_dose_receipt(
            I, student, corpus, acache, baseline_receipt, baseline_sha,
            TARGET, candidate_path, candidate_sha,
        )
        candidate_static = dose_receipt["static"]
        required_classes = tuple(dose_receipt["required_classes"])
        decision = dose_receipt["decision"]
        gate = {
            "format": "ptq-opd-static-gate-receipt-v2",
            "milestone": TARGET,
            "candidate": str(candidate_path),
            "candidate_sha256": candidate_sha,
            "baseline_receipt_sha256": baseline_sha,
            "required_classes": list(required_classes),
            "baseline_metrics": baseline_receipt["metrics"],
            "candidate_metrics": candidate_static["metrics"],
            "decision": decision,
            "passed": decision["passed"],
            "ts": time.time(),
        }
        gate_path = OUTDIR / f"STATIC_GATE_STEP{TARGET}.json"
        atomic_json(gate_path, gate)
        gate_sha = C.canonical_sha256(gate)
        if decision["passed"]:
            passed_payload = checkpoint_payload(
                I, student, norms, outputs, optimizer, identity, lineage, baseline_sha,
                TARGET, TARGET, gate, candidate_sha,
            )
            save_torch_atomic(LATEST, passed_payload)
            status(
                "static_pass_waiting_paired_panel",
                next_update=TARGET,
                last_passed_milestone=TARGET,
                candidate=str(candidate_path),
                candidate_sha256=candidate_sha,
                gate_receipt=str(gate_path),
                gate_receipt_sha256=gate_sha,
                promotable=False,
                paired_gate_required=True,
            )
            emit("static_gate_pass", milestone=TARGET, candidate_sha256=candidate_sha, gate_receipt_sha256=gate_sha, promotable=False)
        else:
            status(
                "static_failed_stop_arm_a",
                next_update=TARGET,
                candidate=str(candidate_path),
                candidate_sha256=candidate_sha,
                gate_receipt=str(gate_path),
                gate_receipt_sha256=gate_sha,
                decision=decision,
                promotable=False,
                arm_b_priority=True,
            )
            emit("static_gate_fail", milestone=TARGET, candidate_sha256=candidate_sha, decision=decision, arm_b_priority=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        try:
            status("failed", error=f"{type(exc).__name__}: {exc}", promotable=False)
            emit("failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
