#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

TASK = "PUBLIC_TASK"
LABEL = "current-GENESIS-WITHOUT-QTIP2"
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
SOURCE_HOST = "203.0.113.9"
SOURCE_ROOT = Path("$HOME/run-bundles/P537_TERMINAL_PUBLIC_TASK_s8")
SOURCE_SCORE_REL = Path("receipts/SCORE_U030_283aa34e65912a9023c8157e42505b1bd75dff96d5577bacaff879eb7c8e1d9c_full512.json")
SOURCE_OUT_REL = Path("out/U030_283aa34e65912a9023c8157e42505b1bd75dff96d5577bacaff879eb7c8e1d9c_full512")
CHECKPOINT_REL = Path("checkpoints/UPDATE_030_283aa34e65912a9023c8157e42505b1bd75dff96d5577bacaff879eb7c8e1d9c.pt")
CLASS_MAP_REL = Path("inputs/BQ3_STEP0_PER_CLASS.json")
WINDOW_CONTRACT_REL = Path("inputs/WINDOW_CONTRACT.json")
EXPECTED = {
    "source_score_sha256": "c3ba83fddf8f39d4b300c2baf8ad242bfdef21d3a90ac758b005fd01b078d3d5",
    "checkpoint_sha256": "283aa34e65912a9023c8157e42505b1bd75dff96d5577bacaff879eb7c8e1d9c",
    "assignment_sha256": "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d",
    "wire_manifest_sha256": "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755",
    "compact_manifest_sha256": "d9421f1f6d0e696608bb0ce9b09131e63790c18e9cd536e440b1884b727db00d",
    "class_map_sha256": "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88",
    "window_contract_sha256": "91a33069d7d2f5648d63ef10b4a11eb122dbce740eec2ac9acd0bc202325fbad",
    "window_ids_sha256": "036f25d3e52783865ddc4915bf4d0cb05839ca214b36af36cfda74b16c16b99c",
    "teacher_done_sha256": "6338af84f907a26dfdf0f784edc322aa672738542ed884b70e4d9b6e96aa33b0",
    "teacher_manifest_sha256": "3c3b99d1a8ee1c20d92f41dbe0578f3fe095a099595211f7f6f4d1f60329a130",
    "teacher_selected_sha256_set": "d02a5836a6463cf5c6883bb370b1d6e559039b7fed7864ac1ee58c9848d93d0e",
    "model_index_sha256": "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
    "instrument_id_sha256": "7f7167886e7df3b622412cfaa9741fb2d2108c3b8fa0bd8cdd1e772cea74ad6d",
    "canonical_builder_sha256": "d56677ed63711aac24181463d7ef8ac45499c4b507919b3ad4d5dcb63da205bb",
    "canonical_reader_sha256": "bc0920b8865376463e58d11686e888524122b9bc995668fca23fa1ec24312b42",
    "canonical_delta_source_sha256": "2aeed7527631050ad440a52fe796502ff01dcd98096f86dd20e8ca9e9187625f",
}
SEALED_GLOBAL = 0.08394998423027422
SEALED_BY_CLASS = {
    "agentic": 0.10077935296474695,
    "chat": 0.038348293176555956,
    "code": 0.0417040064907229,
    "multilingual": 0.14647959260940158,
    "prose": 0.11731817815086114,
    "reasoning": 0.026519590746997834,
}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def stage_one(remote_rel: Path, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run([
        "rsync", "-a", "--partial", "--timeout=120",
        "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3",
        f"{SOURCE_HOST}:{SOURCE_ROOT / remote_rel}", str(local),
    ])


def exact_float(actual: float, expected: float, name: str, atol: float = 5e-15) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > atol:
        raise RuntimeError(f"{name} mismatch: expected={expected!r} observed={actual!r}")


def validate_claim(mission: Path) -> str:
    claim_path = Path("$HOME/HOST_CLAIM.json")
    raw = claim_path.read_bytes()
    claim = json.loads(raw)
    if claim.get("owner") != TASK or claim.get("task_id") != TASK or claim.get("host") != "compute-node-6" or claim.get("mission") != str(mission):
        raise RuntimeError(f"host claim drift: owner={claim.get('owner')} task_id={claim.get('task_id')} host={claim.get('host')} mission={claim.get('mission')}")
    return hashlib.sha256(raw).hexdigest()


def write_progress(mission: Path, *, state: str, completed: int, started: float, detail: dict | None = None) -> None:
    payload = {
        "schema": "p623-current-genesis-baseline-progress-v1",
        "task_id": TASK,
        "label": LABEL,
        "host": "compute-node-6",
        "state": state,
        "scorer_pid": os.getpid(),
        "completed_windows": completed,
        "target_windows": 512,
        "elapsed_seconds": time.monotonic() - started,
        "updated_unix": time.time(),
    }
    if detail:
        payload["detail"] = detail
    atomic_json(mission / "run/PROGRESS.json", payload)


def identity_canary(mission: Path, source: dict, source_score_path: Path, checkpoint_path: Path,
                    class_map_path: Path, window_contract_path: Path, started: float) -> dict:
    first_diff = None
    checks: list[tuple[str, object, object]] = [
        ("source_score_sha256", sha256(source_score_path), EXPECTED["source_score_sha256"]),
        ("source.status", source.get("status"), "PASS_VALIDATED_RECEIPT"),
        ("source.schema", source.get("schema"), "genesis-repair-terminal-full512-v1"),
        ("source.direction", source.get("direction"), "KL(teacher||candidate)"),
        ("source.windows", source.get("windows"), 512),
        ("source.support", source.get("support"), 8192),
        ("source.cutoff", source.get("cutoff"), 1024),
        ("source.window_ids_sha256", source.get("window_ids_sha256"), EXPECTED["window_ids_sha256"]),
        ("checkpoint_sha256", sha256(checkpoint_path), EXPECTED["checkpoint_sha256"]),
        ("class_map_sha256", sha256(class_map_path), EXPECTED["class_map_sha256"]),
        ("window_contract_sha256", sha256(window_contract_path), EXPECTED["window_contract_sha256"]),
        ("source.checkpoint_sha256", source.get("checkpoint_sha256"), EXPECTED["checkpoint_sha256"]),
        ("source.instrument_id_sha256", source.get("instrument_id_sha256"), EXPECTED["instrument_id_sha256"]),
    ]
    instrument = source.get("instrument", {})
    artifact_validation = instrument.get("artifact_validation", {})
    checks.extend([
        ("instrument.assignment_sha256", instrument.get("assignment_sha256"), EXPECTED["assignment_sha256"]),
        ("instrument.wire_manifest_sha256", instrument.get("wire_manifest_sha256"), EXPECTED["wire_manifest_sha256"]),
        ("instrument.compact_manifest_sha256", instrument.get("compact_manifest_sha256"), EXPECTED["compact_manifest_sha256"]),
        ("instrument.window_contract_sha256", instrument.get("window_contract_sha256"), EXPECTED["window_contract_sha256"]),
        ("instrument.teacher_done_sha256", instrument.get("teacher_done_sha256"), EXPECTED["teacher_done_sha256"]),
        ("instrument.model_index_sha256", instrument.get("model_index_sha256"), EXPECTED["model_index_sha256"]),
        ("instrument.canonical_builder_sha256", instrument.get("canonical_builder_sha256"), EXPECTED["canonical_builder_sha256"]),
        ("instrument.canonical_reader_sha256", instrument.get("canonical_reader_sha256"), EXPECTED["canonical_reader_sha256"]),
        ("instrument.canonical_delta_source_sha256", instrument.get("canonical_delta_source_sha256"), EXPECTED["canonical_delta_source_sha256"]),
        ("instrument.microbatch", instrument.get("microbatch"), 2),
        ("instrument.attention", instrument.get("attention"), "eager"),
        ("instrument.teacher_manifest_sha256", artifact_validation.get("teacher_manifest_sha256"), EXPECTED["teacher_manifest_sha256"]),
        ("instrument.teacher_selected_sha256_set", artifact_validation.get("teacher_selected_sha256_set"), EXPECTED["teacher_selected_sha256_set"]),
        ("instrument.train_selected_file_count", artifact_validation.get("train_selected_file_count"), 0),
    ])
    for name, observed, expected in checks:
        if observed != expected:
            first_diff = {"json_path": name, "expected": expected, "observed": observed}
            break
    if first_diff is None:
        try:
            exact_float(float(source["global"]["mean"]), SEALED_GLOBAL, "sealed global")
            for cls in CLASSES:
                exact_float(float(source["by_class"][cls]["mean"]), SEALED_BY_CLASS[cls], f"sealed class {cls}")
        except Exception as exc:
            first_diff = {"json_path": "sealed_baseline_numeric", "expected": {"global": SEALED_GLOBAL, "by_class": SEALED_BY_CLASS}, "observed": str(exc)}
    canary = {
        "schema": "p623-current-genesis-instrument-canary-v1",
        "status": "PASS_IDENTITY_BASELINE_CANARY" if first_diff is None else "FAIL_IDENTITY_BASELINE_CANARY",
        "task_id": TASK,
        "label": LABEL,
        "host": "compute-node-6",
        "direction": "KL(teacher||candidate)",
        "assignment_sha256": EXPECTED["assignment_sha256"],
        "checkpoint_sha256": EXPECTED["checkpoint_sha256"],
        "model_index_sha256": EXPECTED["model_index_sha256"],
        "wire_manifest_sha256": EXPECTED["wire_manifest_sha256"],
        "window_ids_sha256": EXPECTED["window_ids_sha256"],
        "class_map_sha256": EXPECTED["class_map_sha256"],
        "teacher_done_sha256": EXPECTED["teacher_done_sha256"],
        "teacher_manifest_sha256": EXPECTED["teacher_manifest_sha256"],
        "instrument_id_sha256": EXPECTED["instrument_id_sha256"],
        "source_score_sha256": EXPECTED["source_score_sha256"],
        "microbatch": 2,
        "attention": "eager",
        "support": 8192,
        "cutoff": 1024,
        "mask_policy": "TWOBIN_REUSE_MASK=1; paired identical ordered positions; no imputation",
        "aggregation": "float64 position mean globally and within frozen source_class",
        "baseline_expected": {"global": SEALED_GLOBAL, "by_class": SEALED_BY_CLASS},
        "first_diff": first_diff,
        "elapsed_seconds": time.monotonic() - started,
        "created_unix": time.time(),
    }
    atomic_json(mission / "receipts/CANARY.json", canary)
    if first_diff is not None:
        atomic_json(mission / "receipts/MISMATCH.json", canary)
        raise RuntimeError(f"instrument canary failed: {first_diff}")
    return canary


def load_and_verify(path: Path, expected: dict) -> tuple[torch.Tensor, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha = sha256(path)
    if observed_sha != expected["sha256"]:
        raise RuntimeError(f"raw artifact SHA mismatch win{expected['win']}: expected={expected['sha256']} observed={observed_sha}")
    if path.stat().st_size != int(expected["bytes"]):
        raise RuntimeError(f"raw artifact byte mismatch win{expected['win']}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensor = payload.get("kld")
    if payload.get("win") != expected["win"] or payload.get("support") != 8192 or payload.get("cutoff") != 1024:
        raise RuntimeError(f"raw payload identity mismatch win{expected['win']}")
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (1024,) or not bool(torch.isfinite(tensor).all()) or bool((tensor < -1e-6).any()):
        raise RuntimeError(f"raw KLD tensor invalid win{expected['win']}")
    tensor = tensor.double()
    observed_mean = float(tensor.mean())
    exact_float(observed_mean, float(expected["mean"]), f"raw window mean win{expected['win']}")
    return tensor, observed_mean


def make_table(mission: Path, source: dict, rows: list[dict], tensors: list[torch.Tensor], count: int, started: float) -> dict:
    selected_rows = rows[:count]
    selected_tensors = tensors[:count]
    elapsed = time.monotonic() - started
    joined = torch.cat(selected_tensors)
    global_mean = float(joined.mean())
    by_class = {}
    for cls in CLASSES:
        indices = [i for i, row in enumerate(selected_rows) if row["source_class"] == cls]
        if indices:
            combined = torch.cat([selected_tensors[i] for i in indices])
            by_class[cls] = {
                "mean_kld": float(combined.mean()),
                "window_count": len(indices),
                "position_count": int(combined.numel()),
            }
        else:
            by_class[cls] = {"mean_kld": None, "window_count": 0, "position_count": 0}
    subset_manifest = [
        {"win": row["win"], "source_class": row["source_class"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in selected_rows
    ]
    subset_manifest_sha = hashlib.sha256(json.dumps(subset_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    scorer_sha = sha256(Path(__file__))
    payload = {
        "schema": "p623-current-genesis-matched-baseline-v1",
        "status": "PASS_MEASURED_RAW_ARTIFACT_REDUCTION",
        "task_id": TASK,
        "label": LABEL,
        "measurement_label": "MEASURED",
        "host": "compute-node-6",
        "source_host": "compute-node-8 read-only sealed measurement bank over QSFP",
        "window_count": count,
        "position_count": count * 1024,
        "support": 8192,
        "cutoff": 1024,
        "direction": "KL(teacher||candidate)",
        "microbatch": 2,
        "attention": "eager",
        "loader_mode": "torch-mmap",
        "mask_policy": "TWOBIN_REUSE_MASK=1; paired identical ordered positions; no imputation",
        "aggregation": "float64 position mean globally and within frozen source_class",
        "global": {"mean_kld": global_mean, "window_count": count, "position_count": count * 1024},
        "by_class": by_class,
        "ordered_windows": [row["win"] for row in selected_rows],
        "ordered_window_ids_sha256": EXPECTED["window_ids_sha256"],
        "subset_window_manifest_sha256": subset_manifest_sha,
        "source_full_window_output_set_sha256": source["outputs"]["window_output_set_sha256"],
        "model_index_sha256": EXPECTED["model_index_sha256"],
        "assignment_sha256": EXPECTED["assignment_sha256"],
        "checkpoint_sha256": EXPECTED["checkpoint_sha256"],
        "wire_manifest_sha256": EXPECTED["wire_manifest_sha256"],
        "compact_manifest_sha256": EXPECTED["compact_manifest_sha256"],
        "class_map_sha256": EXPECTED["class_map_sha256"],
        "window_contract_sha256": EXPECTED["window_contract_sha256"],
        "teacher_done_sha256": EXPECTED["teacher_done_sha256"],
        "teacher_manifest_sha256": EXPECTED["teacher_manifest_sha256"],
        "teacher_selected_sha256_set": EXPECTED["teacher_selected_sha256_set"],
        "instrument_id_sha256": EXPECTED["instrument_id_sha256"],
        "canonical_builder_sha256": EXPECTED["canonical_builder_sha256"],
        "canonical_reader_sha256": EXPECTED["canonical_reader_sha256"],
        "canonical_delta_source_sha256": EXPECTED["canonical_delta_source_sha256"],
        "scorer_path": str(Path(__file__)),
        "scorer_sha256": scorer_sha,
        "source_score_receipt_sha256": EXPECTED["source_score_sha256"],
        "canary_receipt": str(mission / "receipts/CANARY.json"),
        "canary_receipt_sha256": sha256(mission / "receipts/CANARY.json"),
        "elapsed_seconds": elapsed,
        "windows_per_second": count / elapsed if elapsed else None,
        "windows_per_minute": count * 60 / elapsed if elapsed else None,
        "created_unix": time.time(),
    }
    name = {8: "EARLY_8.json", 64: "INTERIM_64.json", 512: "FINAL_512.json"}[count]
    out = mission / "out" / name
    atomic_json(out, payload)
    payload["artifact_path"] = str(out)
    payload["artifact_sha256"] = sha256(out)
    return payload


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} MISSION")
    mission = Path(sys.argv[1]).resolve()
    started = time.monotonic()
    claim_sha = validate_claim(mission)
    write_progress(mission, state="STAGING_CANARY_INPUTS", completed=0, started=started, detail={"claim_sha256": claim_sha})

    source_score_path = mission / "inputs/SOURCE_FULL512_RECEIPT.json"
    checkpoint_path = mission / "inputs/UPDATE_030.pt"
    class_map_path = mission / "inputs/BQ3_STEP0_PER_CLASS.json"
    window_contract_path = mission / "inputs/WINDOW_CONTRACT.json"
    stage_one(SOURCE_SCORE_REL, source_score_path)
    stage_one(CHECKPOINT_REL, checkpoint_path)
    stage_one(CLASS_MAP_REL, class_map_path)
    stage_one(WINDOW_CONTRACT_REL, window_contract_path)
    source = json.loads(source_score_path.read_text())
    canary = identity_canary(
        mission, source, source_score_path, checkpoint_path,
        class_map_path, window_contract_path, started,
    )
    canary_sha = sha256(mission / "receipts/CANARY.json")
    write_progress(mission, state="CANARY_PASS_WAITING_SECOND_SSH", completed=0, started=started, detail={"canary_sha256": canary_sha})
    # Acceptance requires an independent SSH observer to prove the exact scorer
    # PID is live and to witness atomic PROGRESS motion.  The observer creates
    # this task-local handshake only after checking /proc/SCORER_PID.
    observer = mission / "run/SECOND_SSH_SEEN.json"
    deadline = time.monotonic() + 30.0
    while not observer.is_file() and time.monotonic() < deadline:
        write_progress(mission, state="CANARY_PASS_WAITING_SECOND_SSH", completed=0, started=started, detail={"canary_sha256": canary_sha})
        time.sleep(0.5)
    if not observer.is_file():
        raise RuntimeError("second-SSH scorer liveness handshake not observed within 30 seconds")
    observer_payload = json.loads(observer.read_text())
    if observer_payload.get("scorer_pid") != os.getpid() or observer_payload.get("host") != "compute-node-6":
        raise RuntimeError(f"second-SSH handshake identity drift: {observer_payload}")
    write_progress(mission, state="SECOND_SSH_LIVENESS_VERIFIED", completed=0, started=started, detail={"observer": observer_payload})
    time.sleep(2.0)

    expected_rows = source["outputs"]["per_window"]
    if len(expected_rows) != 512 or [int(row["win"]) for row in expected_rows] != list(range(512)):
        raise RuntimeError("source per-window manifest order/coverage drift")
    raw_dir = mission / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    tensors: list[torch.Tensor] = []
    for start, stop in ((0, 8), (8, 64), (64, 512)):
        include = mission / f"run/rsync_{start:03d}_{stop-1:03d}.files"
        include.write_text("".join(f"kld_win{win}.pt\n" for win in range(start, stop)))
        run([
            "rsync", "-a", "--partial", "--timeout=180", "--files-from", str(include),
            "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3",
            f"{SOURCE_HOST}:{SOURCE_ROOT / SOURCE_OUT_REL}/", str(raw_dir) + "/",
        ])
        for win in range(start, stop):
            tensor, mean = load_and_verify(raw_dir / f"kld_win{win}.pt", expected_rows[win])
            tensors.append(tensor)
            rows.append({
                "win": win,
                "source_class": str(expected_rows[win]["source_class"]),
                "mean_kld": mean,
                "bytes": int(expected_rows[win]["bytes"]),
                "sha256": str(expected_rows[win]["sha256"]),
            })
            if win == 0 or (win + 1) % 8 == 0:
                write_progress(mission, state="SCORING", completed=win + 1, started=started)
        if stop in (8, 64, 512):
            result = make_table(mission, source, rows, tensors, stop, started)
            write_progress(mission, state={8: "EARLY_8_READY", 64: "INTERIM_64_READY", 512: "FINAL_512_READY"}[stop], completed=stop, started=started, detail={"artifact_path": result["artifact_path"], "artifact_sha256": result["artifact_sha256"]})

    final_path = mission / "out/FINAL_512.json"
    final = json.loads(final_path.read_text())
    exact_float(float(final["global"]["mean_kld"]), SEALED_GLOBAL, "reduced final global")
    for cls in CLASSES:
        exact_float(float(final["by_class"][cls]["mean_kld"]), SEALED_BY_CLASS[cls], f"reduced final class {cls}")
    receipt = {
        "schema": "p623-current-genesis-baseline-seal-v1",
        "status": "PASS_FINAL_512_MATCHED_BASELINE",
        "task_id": TASK,
        "label": LABEL,
        "host": "compute-node-6",
        "claim_sha256": claim_sha,
        "artifacts": {
            name: {"path": str(mission / "out" / name), "sha256": sha256(mission / "out" / name)}
            for name in ("EARLY_8.json", "INTERIM_64.json", "FINAL_512.json")
        },
        "canary": {"path": str(mission / "receipts/CANARY.json"), "sha256": sha256(mission / "receipts/CANARY.json")},
        "source_score_receipt_sha256": EXPECTED["source_score_sha256"],
        "scorer_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.monotonic() - started,
        "created_unix": time.time(),
    }
    atomic_json(mission / "receipts/SEAL.json", receipt)
    write_progress(mission, state="PASS_SEALED", completed=512, started=started, detail={"seal_sha256": sha256(mission / "receipts/SEAL.json")})
    print(json.dumps({"status": receipt["status"], "seal": str(mission / "receipts/SEAL.json"), "seal_sha256": sha256(mission / "receipts/SEAL.json")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            mission = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
            atomic_json(mission / "receipts/FAILED.json", {
                "schema": "p623-current-genesis-baseline-failure-v1",
                "status": "FAIL_CLOSED",
                "task_id": TASK,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_unix": time.time(),
            })
            atomic_json(mission / "run/PROGRESS.json", {
                "schema": "p623-current-genesis-baseline-progress-v1",
                "task_id": TASK,
                "label": LABEL,
                "host": "compute-node-6",
                "state": "FAIL_CLOSED",
                "scorer_pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_unix": time.time(),
            })
        finally:
            raise
