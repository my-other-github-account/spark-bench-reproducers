#!/usr/bin/env python3
"""P651 same-host physical baseline parity gate using the exact pinned P632 stack.

This controller imports (rather than copies or edits) the exact P632 scorer module,
canonical reader, builder, loader, inputs, and objective.  It intentionally runs the
unchanged P602 physical package with no checkpoint/candidate overlay.  Candidate
scoring is forbidden unless the resulting ordered CODE76 window means reproduce the
sealed PRE_REPAIR_FULL512 rows to 1e-12 on compute-node-7.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

TASK = "PUBLIC_TASK"
MISSION = Path("$HOME/run-bundles/P651_STREAM_CONSUMER_PUBLIC_TASK_s7")
P632_ROOT = Path("$HOME/run-bundles/P632_DIRECTIONAL_PUBLIC_TASK_s7")
P632_SCORER = P632_ROOT / "code/p632_score.py"
EXPECTED_P632_SHA = "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f"
RECEIPT = MISSION / "receipts/BASELINE_PARITY.json"
FAILURE = MISSION / "receipts/BASELINE_PARITY_MISMATCH.json"
OUT = MISSION / "out/BASELINE_PHYSICAL_FAST"
CACHE = MISSION / "scratch/BASELINE_PHYSICAL_FAST"
PROGRESS = MISSION / "run/PROGRESS_BASELINE_PHYSICAL_FAST.json"
SENTINEL = MISSION / "run/LOADER_SENTINEL_BASELINE_PHYSICAL_FAST.json"
DISK_FLOOR = 6 * (1 << 30)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def exact_claim() -> tuple[bytes, dict]:
    path = Path("$HOME/HOST_CLAIM.json")
    raw = path.read_bytes()
    claim = json.loads(raw)
    expected = {
        "host": "compute-node-7",
        "owner": TASK,
        "task_id": TASK,
        "mission": str(MISSION),
    }
    drift = {k: (claim.get(k), v) for k, v in expected.items() if claim.get(k) != v}
    if drift:
        raise RuntimeError(f"P651 host claim drift: {drift}")
    return raw, claim


def load_exact_p632():
    if sha256(P632_SCORER) != EXPECTED_P632_SHA:
        raise RuntimeError("exact P632 scorer SHA drift")
    sys.path.insert(0, str(P632_SCORER.parent))
    spec = importlib.util.spec_from_file_location("p651_exact_p632", P632_SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import exact P632 scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execution ownership/output paths are task-local. All frozen scorer inputs,
    # objective code, canonical sources, and P602 physical package remain P632's.
    module.TASK = TASK
    module.ROOT = MISSION
    module.current_claim = exact_claim
    return module


def run() -> dict:
    started = time.time()
    if any(path.exists() for path in (RECEIPT, FAILURE, OUT, CACHE, PROGRESS, SENTINEL)):
        raise RuntimeError("once-only P651 baseline parity target already exists")
    if shutil.disk_usage(MISSION).free < DISK_FLOOR:
        raise RuntimeError("disk free below 6 GiB baseline gate floor")

    p = load_exact_p632()
    sealed = p.preflight_contract("fast")
    claim_before, _ = exact_claim()
    claim_sha = hashlib.sha256(claim_before).hexdigest()
    gpu_before = p.gpu_snapshot(require_zero_util=True)
    CACHE.mkdir(parents=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)

    run_id = "P651_BASELINE_PHYSICAL_FAST"
    base = p.load_module(f"canonical_parent_{run_id}", p.CANONICAL_READER)
    p.configure_parent_module(base, cache=CACHE, progress=PROGRESS, sentinel=SENTINEL)
    # Exact P632's CheckpointTierSource applies the scorer's 6 GiB post-stage
    # floor instead of the canonical parent's historical 20 GiB remote-stage
    # floor. The unchanged physical path must inherit that same frozen contract.
    base.DISK_FLOOR = p.DISK_FLOOR
    env_contract = p.install_environment()
    sys.path.insert(0, str(p.PACKAGE))
    import t8192_ds4_build_v3 as builder

    class LocalPhysicalTierSource(base.BananaSmasherTierSource):
        """Exact P632 source-local receipt-first stage, without any overlay."""
        def _stage_remote(self, layer: int, row: dict) -> Path:
            self._cleanup_stage()
            stage = self.CACHE / f"layer_{layer:03d}" if hasattr(self, "CACHE") else base.CACHE / f"layer_{layer:03d}"
            cache_root = stage.parent
            temp = cache_root / f".layer_{layer:03d}.partial"
            if stage.exists():
                shutil.rmtree(stage)
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir(parents=True)
            wire_row = self.wire_rows[layer]
            required_bytes = int(wire_row["physical_wire_bytes"])
            free = shutil.disk_usage(cache_root).free
            if free - required_bytes < DISK_FLOOR:
                raise RuntimeError(f"disk floor before source-local stage L{layer}: free={free} required={required_bytes}")
            source_layer = Path(p.REMOTE_PACKAGE) / f"layer_{layer:03d}"
            receipt_path = temp / "LAYER_RECEIPT.json"
            shutil.copy2(source_layer / "LAYER_RECEIPT.json", receipt_path)
            if p.sha256_file(receipt_path) != wire_row["receipt_sha256"]:
                raise RuntimeError(f"L{layer} receipt SHA drift after source-local stage")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("schema") != "banana_smasher-materialized-layer-v1"
                or receipt.get("status") != "PASS"
                or int(receipt.get("layer", -1)) != layer
                or receipt.get("assignment_sha256") != p.ASSIGNMENT_SHA
                or receipt.get("builder_sha256") != base.BUILD_BUILDER_SHA
                or int(receipt.get("physical_wire_bytes", -1)) != required_bytes
            ):
                raise RuntimeError(f"L{layer} receipt identity drift")
            receipt_files = receipt.get("files")
            if not isinstance(receipt_files, list) or not receipt_files:
                raise RuntimeError(f"L{layer} receipt file list missing")
            allowed_paths: list[str] = []
            for item in receipt_files:
                rel = item.get("path")
                rel_path = Path(rel) if isinstance(rel, str) else Path("/")
                if not isinstance(rel, str) or rel_path.is_absolute() or ".." in rel_path.parts or rel in allowed_paths:
                    raise RuntimeError(f"L{layer} receipt path unsafe or duplicate: {rel}")
                allowed_paths.append(rel)
                target = temp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_layer / rel, target)
            stage_accounting = base.validate_staged_layer(
                temp, receipt, required_bytes=required_bytes,
                free_bytes_after=shutil.disk_usage(cache_root).free, floor=DISK_FLOOR,
            )
            os.replace(temp, stage)
            self.active_stage = stage
            p.atomic_json(MISSION / f"run/BASELINE_LAYER_{layer:03d}_STAGE.json", {
                "schema": "p651-physical-baseline-layer-stage-v1", "task_id": TASK, "layer": layer,
                "receipt_sha256": wire_row["receipt_sha256"], **stage_accounting,
                "allowed_files": allowed_paths, "stage": str(stage),
                "transport": "source-local receipt-first allowlist copy; exact P632 canonical torch-mmap path unchanged",
                "created_unix": time.time(),
            })
            return stage

    # Exact unchanged-input path: canonical physical values and P632's exact
    # source-local staging semantics, with no candidate/checkpoint overlay.
    builder.PlaneSource = LocalPhysicalTierSource

    labels_payload = json.loads(p.LABELS.read_text())
    classes = {int(row["win"]): str(row["source_class"]) for row in labels_payload["per_window"]}
    counts = Counter(classes.values())
    if set(classes) != set(range(512)) or counts != Counter(sealed["window_contract"]["full512_class_counts"]):
        raise RuntimeError("class label surface drift")
    wins = list(p.EXPECTED_CODE76_IDS)
    OUT.mkdir(parents=True)
    original_argv = sys.argv
    original_cwd = Path.cwd()
    rc = -1
    try:
        sys.argv = [
            "t8192_ds4_build_v3.py", "--mode", "planes",
            "--planes-dir", str(p.COMPACT_MANIFEST),
            "--ref-dir", str(p.TEACHER / "t8192_eval"),
            "--corpus", str(p.CORPUS),
            "--meta-dir", str(p.MODEL), "--local-dir", str(p.MODEL),
            "--out", str(OUT), "--cand-pos-limit", "1024",
            "--count", str(len(wins)), "--chunk", str(len(wins)), "--mb", "2",
            "--windows", ",".join(map(str, wins)),
            "--tag", run_id,
        ]
        os.chdir(p.TEACHER)
        with p.force_weights_only_torch_loads() as weights_only_stats:
            rc = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        base.retire_scratch(CACHE)
    if rc:
        raise RuntimeError(f"canonical builder rc={rc}")
    if any(CACHE.iterdir()):
        raise RuntimeError("layer scratch not retired")

    progress = json.loads(PROGRESS.read_text())
    expected_visits = list(range(43))
    if (
        progress.get("completed_layers") != expected_visits
        or progress.get("mmap_completed_layers") != expected_visits
        or progress.get("completed_chunks") != 1
        or progress.get("local_stage_retired") is not True
        or progress.get("mmap_loader_mode") != "torch-mmap"
        or progress.get("mmap_loader_sha256") != p.CANONICAL_SHA256[p.LOADER_SOURCE]
        or progress.get("mmap_input_identity_sha256") != p.WIRE_SHA
    ):
        raise RuntimeError("fresh exact P632 loader coverage/proof drift")
    if not SENTINEL.is_file():
        raise RuntimeError("loader sentinel missing")

    reduced = p.reduce_outputs(OUT, wins, classes)
    observed = {int(row["win"]): float(row["mean"]) for row in reduced["per_window"]}
    baseline = json.loads(p.BASELINE_FULL512.read_text())
    expected = {int(row["win"]): float(row["mean"]) for row in baseline["per_window"]}
    if sorted(observed) != sorted(wins) or not set(wins).issubset(expected):
        raise RuntimeError("baseline parity window surface drift")
    rows = [{
        "win": win,
        "source_class": classes[win],
        "observed": observed[win],
        "expected": expected[win],
        "delta": observed[win] - expected[win],
    } for win in wins]
    deltas = [row["delta"] for row in rows]
    if not all(math.isfinite(value) for value in deltas):
        raise RuntimeError("non-finite baseline parity delta")
    maximum = max(abs(value) for value in deltas)
    claim_after, _ = exact_claim()
    if claim_after != claim_before:
        raise RuntimeError("P651 claim changed during baseline parity")

    completed = time.time()
    instrument = {
        "exact_p632_scorer_path": str(P632_SCORER),
        "exact_p632_scorer_sha256": EXPECTED_P632_SHA,
        "controller_path": str(Path(__file__)),
        "controller_sha256": sha256(Path(__file__)),
        "canonical_reader_sha256": p.CANONICAL_SHA256[p.CANONICAL_READER],
        "canonical_builder_sha256": p.CANONICAL_SHA256[p.CANONICAL_BUILDER],
        "canonical_delta_source_sha256": p.CANONICAL_SHA256[p.CANONICAL_DELTA],
        "canonical_safety_sha256": p.CANONICAL_SHA256[p.CANONICAL_SAFETY],
        "loader_sha256": p.CANONICAL_SHA256[p.LOADER_SOURCE],
        "window_contract_sha256": p.EXPECTED_INPUT_SHA256[p.WINDOW_CONTRACT],
        "compact_manifest_sha256": p.COMPACT_SHA,
        "assignment_sha256": p.ASSIGNMENT_SHA,
        "wire_manifest_sha256": p.WIRE_SHA,
        "model_index_sha256": p.MODEL_INDEX_SHA,
        "teacher_done_sha256": p.TEACHER_DONE_SHA,
        "environment_contract": env_contract,
        "torch_load_safety": weights_only_stats,
        "attention": "eager",
        "microbatch": 2,
        "candidate_adapter": None,
        "physical_inputs_unchanged": True,
    }
    payload = {
        "schema": "p651-p632-substituted-artifact-baseline-gate-v1",
        "status": "PASS_EXACT_SAME_HOST_INSTRUMENT" if maximum <= 1e-12 else "FAIL_BASELINE_PARITY_MISMATCH",
        "task_id": TASK,
        "host": "compute-node-7",
        "measurement_label": "MEASURED_HELDOUT_BASELINE_GATE",
        "exactness_label": "EXACT_PINNED_P632_PHYSICAL_WIRE_NO_OVERLAY",
        "direction": "KL(teacher||candidate)",
        "support": 8192,
        "cutoff": 1024,
        "windows": len(wins),
        "window_ids": wins,
        "maximum_absolute_window_mean_delta": maximum,
        "mean_window_delta": sum(deltas) / len(deltas),
        "tolerance": 1e-12,
        "per_window_comparison": rows,
        "global": reduced["global"],
        "by_class": reduced["by_class"],
        "sealed_baseline_receipt": str(p.BASELINE_FULL512),
        "sealed_baseline_receipt_sha256": p.EXPECTED_INPUT_SHA256[p.BASELINE_FULL512],
        "instrument": instrument,
        "instrument_id_sha256": p.canonical_json_sha256(instrument),
        "local_claim_sha256": claim_sha,
        "coverage": {
            "expected_chunks": 1,
            "expected_layer_visits": 43,
            "completed_layers": progress["completed_layers"],
            "mmap_completed_layers": progress["mmap_completed_layers"],
            "local_stage_retired": True,
            "remote_physical_package_mutated": False,
            "persistent_model_mutated": False,
        },
        "outputs": {
            "directory": str(OUT),
            "window_output_set_sha256": reduced["window_output_set_sha256"],
            "per_window": reduced["per_window"],
        },
        "loader_proof": {
            "progress": str(PROGRESS),
            "progress_sha256": sha256(PROGRESS),
            "sentinel": str(SENTINEL),
            "sentinel_sha256": sha256(SENTINEL),
        },
        "gpu_snapshot_before": gpu_before,
        "gpu_snapshot_before_exit": p.gpu_snapshot(own_pid=os.getpid(), require_zero_util=False),
        "started_unix": started,
        "completed_unix": completed,
        "elapsed_seconds": completed - started,
        "windows_per_minute": len(wins) * 60 / (completed - started),
    }
    target = RECEIPT if payload["status"].startswith("PASS") else FAILURE
    atomic_json(target, payload)
    print(json.dumps({
        "status": payload["status"],
        "maximum_absolute_window_mean_delta": maximum,
        "global": reduced["global"]["mean"],
        "receipt": str(target),
        "receipt_sha256": sha256(target),
        "elapsed_seconds": payload["elapsed_seconds"],
    }, sort_keys=True), flush=True)
    if maximum > 1e-12:
        raise RuntimeError(f"baseline parity mismatch max_abs={maximum}")
    return payload


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        if not FAILURE.exists():
            atomic_json(FAILURE, {
                "schema": "p651-p632-substituted-artifact-baseline-gate-failure-v1",
                "status": "FAIL_CLOSED",
                "task_id": TASK,
                "host": "compute-node-7",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc().splitlines()[-40:],
                "exact_p632_scorer_sha256": sha256(P632_SCORER) if P632_SCORER.is_file() else None,
                "controller_sha256": sha256(Path(__file__)),
                "created_unix": time.time(),
            })
        raise
