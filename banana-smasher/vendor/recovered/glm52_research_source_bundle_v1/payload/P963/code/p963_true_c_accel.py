#!/usr/bin/env python3
"""Run exact terminal TRUE-C f521-T on the proven P921 BALANCED64 instrument."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

TASK = "task-redacted"
CANONICAL_TASK = "task-redacted"
CANONICAL_MISSION = Path("${SPARK_HOME}/missions/P929_TRUE_C_REFIT_t_a2b3a979_s6")
ROOT = Path(os.environ.get("P770_ROOT", "${SPARK_HOME}/missions/P963_ACCEL_TRUE_C_t_d78a699d_s1"))
HOST = os.environ.get("P770_HOST", "spark-1")
CLAIM = Path("${SPARK_HOME}/HOST_CLAIM.json")
BASE_SOURCE_HOST = "${QSFP_HOST}"
BASE_REMOTE_PACKAGE = "${SPARK_HOME}/missions/GENESIS_FANIN_t_81c3a62d_s8/package/wire43"
MODEL = Path("${SPARK_HOME}/models/hf/DeepSeek-V4-Flash")
TEACHER = Path("${SPARK_HOME}/missions/DS4_TEACHER")
P632_SCORE = ROOT / "code/p632_score.py"
MECHANICS = ROOT / "code/p760_slice_mechanics.py"
OVERLAY_ADAPTER = Path(os.environ.get("P885_OVERLAY_ADAPTER", str(ROOT / "code/p963_true_c_overlay_adapter.py")))
QTIP_SOURCE = ROOT / "code/p605r_run_qtip_anchor.py"
QTIP_KERNEL = ROOT / "code/qtip_kernel_decompress.py"
QTIP_TLUT = ROOT / "inputs/P641_QTIP_TLUT_SOURCE.pt"
BASE_ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
PARITY = ROOT / "receipts/INSTANCE_PARITY_GATE.json"
BASELINE = ROOT / "inputs/TRUE_PRE_REPAIR_FULL512.json"
FAST_CORRECTION = ROOT / "inputs/FAST64_CORRECTION.json"
BALANCED64 = ROOT / "inputs/BALANCED64_V1.json"
OLD_CORRUPT_ANCHOR = ROOT / "inputs/P783_OLD_CORRUPT_BALANCED64.json"
FROZEN_CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
import sys as _p874_sys
_p874_sys.path.insert(0, str(Path(__file__).resolve().parent)); _p874_sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "code"))
_p874_sys.path.insert(0, str(ROOT / "code"))
PINS = {
    "base_assignment": "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d",
    "base_wire": "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755",
    "compact": "d9421f1f6d0e696608bb0ce9b09131e63790c18e9cd536e440b1884b727db00d",
    "labels": "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88",
    "window_contract": "91a33069d7d2f5648d63ef10b4a11eb122dbce740eec2ac9acd0bc202325fbad",
    "reader": "f5b17ab3a12a3b4d042d9fc712032d9007c6d99d76ad77f256830314900ee51e",
    "builder": "873a98a37a6cf854572983ebfdffc15e2292f6599a7fb3206c14cb866f2f8784",
    "delta": "2aeed7527631050ad440a52fe796502ff01dcd98096f86dd20e8ca9e9187625f",
    "lp4": "7a8e48547824a87a48db4c7142ec53f73303a91ce6a0c95cf1a88b1b87d22350",
    "planes": "aeb3e473a00b48426f56b9f80aefc6bc086b7791ec2372606c724e90db126334",
    "contracts": "0842784bfba78032f122c8e859f2a1df1d67885823e1aa323cc020d3ae6fccbf",
    "safety": "b45f6eef933ac51d2c5f1693f21f1859945a9ad9d18741dd0732fa3956275e0c",
    "loader": "155310d1e6701d6cb2d1c04558514366a2304cb2a8d6d26402ed7c800b8b6c89",
    "model_index": "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
    "teacher_done": "6338af84f907a26dfdf0f784edc322aa672738542ed884b70e4d9b6e96aa33b0",
    "corpus": "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def claim_snapshot() -> tuple[bytes, dict[str, Any]]:
    raw = CLAIM.read_bytes()
    claim = json.loads(raw)
    expected = {"host": HOST, "owner": TASK, "task": TASK, "task_id": TASK, "mission": str(ROOT)}
    drift = {k: (claim.get(k), v) for k, v in expected.items() if claim.get(k) != v}
    if drift or claim.get("status") not in ("CLAIMED", "ACTIVE") or "NO build/fitting/repair/capture" not in claim.get("launch_policy", ""):
        raise RuntimeError(f"P770 claim drift: drift={drift} status={claim.get('status')}")
    return raw, claim


def gpu_snapshot(*, own_pid: int | None = None, require_zero_util: bool = False) -> dict[str, Any]:
    raw = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    apps = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if own_pid is not None and line.split(",", 1)[0].strip() == str(own_pid):
            continue
        apps.append(line.strip())
    if apps:
        raise RuntimeError(f"foreign GPU applications present: {apps}")
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    return {"foreign_compute_apps": apps, "nvidia_reported_utilization": util, "zero_util_required": require_zero_util}


def configure_p632(p: Any, *, measurement: str, wins: list[int]) -> None:
    package, inputs = ROOT / "code/eval_package", ROOT / "inputs"
    p.TASK, p.ROOT, p.CLAIM = TASK, ROOT, CLAIM
    p.SOURCE_HOST, p.REMOTE_PACKAGE, p.PACKAGE = BASE_SOURCE_HOST, BASE_REMOTE_PACKAGE, package
    p.MODEL, p.TEACHER, p.CORPUS = MODEL, TEACHER, TEACHER / "static/windows_ds4_eval.json"
    p.INPUTS = inputs
    p.COMPACT_MANIFEST = inputs / "GENESIS_COMPACT_FANIN.json"
    p.ASSIGNMENT = inputs / "NOMINATED_ASSIGNMENT.json"
    p.LABELS = inputs / "BQ3_STEP0_PER_CLASS.json"
    p.WINDOW_CONTRACT = inputs / "WINDOW_CONTRACT.json"
    p.WIRE_MANIFEST = inputs / "WIRE_43_MANIFEST.json"
    p.BASELINE_FULL512 = BASELINE
    p.CANONICAL_READER = ROOT / "code/genesis_remote_full512.py"
    p.CANONICAL_BUILDER = ROOT / "code_ckpt/t8192_ds4_build_v3.py"  # P874 checkpointed builder
    p.CANONICAL_DELTA = package / "delta_pack_sources.py"
    p.CANONICAL_LP4_PACK = package / "lp4_pack.py"
    p.CANONICAL_PLANES_UNPACK = package / "planes_unpack.py"
    p.CANONICAL_EVAL_CONTRACTS = package / "readapt_eval_contracts.py"
    p.CANONICAL_SAFETY = ROOT / "code/full512_safety.py"
    p.LOADER_SOURCE = ROOT / "code/rail_loading.py"
    p.PHYSICAL_MARKER = inputs / "PHYSICAL_CODE76.json"
    p.PHYSICAL_PASS_MARKER = inputs / "PHYSICAL_CODE76.PASS.json"
    p.ASSIGNMENT_SHA, p.COMPACT_SHA, p.WIRE_SHA = PINS["base_assignment"], PINS["compact"], PINS["base_wire"]
    p.MODEL_INDEX_SHA, p.TEACHER_DONE_SHA, p.CORPUS_SHA = PINS["model_index"], PINS["teacher_done"], PINS["corpus"]
    p.CANONICAL_SHA256 = {
        p.CANONICAL_READER: PINS["reader"], p.CANONICAL_BUILDER: PINS["builder"],
        p.CANONICAL_DELTA: PINS["delta"], p.CANONICAL_LP4_PACK: PINS["lp4"],
        p.CANONICAL_PLANES_UNPACK: PINS["planes"], p.CANONICAL_EVAL_CONTRACTS: PINS["contracts"],
        p.CANONICAL_SAFETY: PINS["safety"], p.LOADER_SOURCE: PINS["loader"],
    }
    p.EXPECTED_INPUT_SHA256 = {p.BASELINE_FULL512: sha256(BASELINE)}
    p.current_claim, p.gpu_snapshot = claim_snapshot, gpu_snapshot
    original = p.configure_parent_module
    def configure_parent_module(base: Any, *, cache: Path, progress: Path, sentinel: Path) -> None:
        original(base, cache=cache, progress=progress, sentinel=sentinel)
        base.DISK_FLOOR = 8 * (1 << 30)
        base.REMOTE_PACKAGE = BASE_REMOTE_PACKAGE
        base.PHYSICAL_PACKAGE = cache
        p.REMOTE_PACKAGE = str(cache)
    p.configure_parent_module = configure_parent_module
    def preflight_contract(mode: str) -> dict[str, Any]:
        expected_mode = "full512" if measurement == "full512" else "interim64"
        if mode != expected_mode:
            raise RuntimeError(f"P770 scoring-mode drift: {mode} != {expected_mode}")
        required = {
            p.CANONICAL_READER: PINS["reader"], p.CANONICAL_BUILDER: PINS["builder"],
            p.CANONICAL_DELTA: PINS["delta"], p.CANONICAL_LP4_PACK: PINS["lp4"],
            p.CANONICAL_PLANES_UNPACK: PINS["planes"], p.CANONICAL_EVAL_CONTRACTS: PINS["contracts"],
            p.CANONICAL_SAFETY: PINS["safety"], p.LOADER_SOURCE: PINS["loader"],
            p.COMPACT_MANIFEST: PINS["compact"], p.ASSIGNMENT: PINS["base_assignment"],
            p.LABELS: PINS["labels"], p.WINDOW_CONTRACT: PINS["window_contract"], p.WIRE_MANIFEST: PINS["base_wire"],
            MODEL / "model.safetensors.index.json": PINS["model_index"], TEACHER / "t8192_eval/DONE.jsonl": PINS["teacher_done"], p.CORPUS: PINS["corpus"],
        }
        drift = {str(path): {"expected": expected, "observed": sha256(path) if path.is_file() else None} for path, expected in required.items() if not path.is_file() or sha256(path) != expected}
        if drift:
            raise RuntimeError(f"canonical physical input drift: {drift}")
        parity = json.loads(PARITY.read_text())
        if parity.get("status") != "PASS_EXACT_SAME_HOST_INSTRUMENT" or float(parity.get("maximum_absolute_window_mean_delta", math.inf)) != 0.0:
            raise RuntimeError("P770 baseline parity prerequisite drift")
        labels = json.loads(p.LABELS.read_text())["per_window"]
        classes = {int(row["win"]): str(row["source_class"]) for row in labels}
        counts = Counter(classes[win] for win in wins)
        if [int(row["win"]) for row in labels] != list(range(512)):
            raise RuntimeError("label order drift")
        return {"window_contract": {"selected_windows": wins, "selected_class_counts": dict(counts)}, "artifacts": {}, "p770_p760_authority": True}
    p.preflight_contract = preflight_contract


def run(anchor: str, measurement: str) -> int:
    if measurement != "balanced64":
        raise RuntimeError("P819 is BALANCED64-only; full512 is a later ship gate")
    window_spec = json.loads(BALANCED64.read_text())
    wins = [int(x) for x in window_spec["windows"]]
    if window_spec.get("name") != "BALANCED64_V1" or len(wins) != 64 or len(set(wins)) != 64:
        raise RuntimeError("BALANCED64_V1 contract drift")
    scoring_mode = "full512" if measurement == "full512" else "interim64"
    os.environ["P770_ROOT"], os.environ["P770_ANCHOR"] = str(ROOT), anchor
    os.environ["P770_MEASUREMENT"] = measurement
    # Exact P841 acceleration patch, applied without the three-layer canary-only
    # activation/logit retention gates (which would retain 43 x 4 GiB).
    os.environ["P832_MATERIALIZE_WORKERS"] = "8"
    os.environ["P832_QTIP_BATCH"] = "1"
    os.environ["P832_QTIP_STREAMS"] = "8"
    os.environ["P832_PIPELINE_MATERIALIZE"] = "1"
    os.environ["P835_VQ_BATCH"] = "8"
    os.environ["TWOBIN_STREAMS"] = "2"
    os.environ["TWOBIN_PREFIX_LOGITS"] = "1"
    mechanics = load_module("p770_mechanics", MECHANICS)
    adapter = load_module("p770_overlay", OVERLAY_ADAPTER)
    mechanics.TASK, mechanics.ROOT, mechanics.CLAIM = TASK, ROOT, CLAIM
    mechanics.P632_ROOT, mechanics.P632_SCORE = ROOT, P632_SCORE
    mechanics.ASSIGNMENT = Path(os.environ["P885_ACTIVE_ASSIGNMENT"])
    mechanics.BASE_ASSIGNMENT, mechanics.BASELINE_PARITY = BASE_ASSIGNMENT, PARITY
    mechanics.QTIP_SOURCE, mechanics.QTIP_KERNEL, mechanics.QTIP_TLUT = QTIP_SOURCE, QTIP_KERNEL, QTIP_TLUT
    mechanics.current_claim = claim_snapshot
    mechanics.preflight_manifests = adapter.preflight_manifest
    mechanics.install_stream_source = lambda base, manifest, cache, mode: adapter.install_stream_source(mechanics, base, manifest, cache, mode, QTIP_SOURCE, QTIP_KERNEL, QTIP_TLUT)
    manifest = adapter.preflight_manifest()
    parity = json.loads(PARITY.read_text())
    if parity.get("status") != "PASS_EXACT_SAME_HOST_INSTRUMENT" or float(parity.get("maximum_absolute_window_mean_delta", math.inf)) > 1e-12:
        raise RuntimeError("baseline parity gate failed")
    run_id = os.environ["P885_RUN_ID"]
    out, receipt_path = ROOT / f"out/{run_id}", ROOT / f"receipts/{run_id}.json"
    cache = Path("${SCRATCH_ROOT}/P963_ACCEL_TRUE_C_t_d78a699d_s1") / run_id
    progress = ROOT / f"run/{run_id}/CANONICAL_PROGRESS.json"
    sentinel = ROOT / f"run/{run_id}/LOADER_SENTINEL.json"
    if any(p.exists() for p in (out, receipt_path, cache, progress, sentinel)):
        raise RuntimeError(f"once-only rail target already exists: {run_id}")
    out.mkdir(parents=True); cache.mkdir(parents=True); progress.parent.mkdir(parents=True, exist_ok=True)
    p632 = mechanics.load_module(f"p770_p632_{anchor}", P632_SCORE)
    configure_p632(p632, measurement=measurement, wins=wins)
    sealed = p632.preflight_contract(scoring_mode)
    claim_raw_before, claim = claim_snapshot()
    claim_sha = hashlib.sha256(claim_raw_before).hexdigest()
    gpu_before = gpu_snapshot(require_zero_util=True)
    base = p632.load_module(f"p770_reader_{anchor}", p632.CANONICAL_READER)
    p632.configure_parent_module(base, cache=cache, progress=progress, sentinel=sentinel)
    base.PHYSICAL_PACKAGE = Path(p632.REMOTE_PACKAGE)
    env_contract = p632.install_environment()
    env_contract["TWOBIN_LAYER_OVERLAP"] = "1"
    os.environ["TWOBIN_LAYER_OVERLAP"] = "1"
    sys.path.insert(0, str(p632.PACKAGE))
    import t8192_ds4_build_v3 as builder
    StreamSource, overlay = adapter.install_stream_source(mechanics, base, manifest, cache, measurement, QTIP_SOURCE, QTIP_KERNEL, QTIP_TLUT)
    builder.PlaneSource = StreamSource
    labels_payload = json.loads(p632.LABELS.read_text())
    classes = {int(row["win"]): str(row["source_class"]) for row in labels_payload["per_window"]}
    if set(classes) != set(range(512)) or Counter(classes[win] for win in wins) != Counter(sealed["window_contract"]["selected_class_counts"]):
        raise RuntimeError("class label surface drift")
    original_argv, original_cwd = sys.argv, Path.cwd()
    started = time.time(); rc = -1
    try:
        microbatch = int(os.environ.get("P963_MB", "2"))
        if microbatch != 2:
            raise RuntimeError("P963 exact accelerated rail is bound to highest numerically-safe microbatch=2; mb4 failed closed")
        sys.argv = ["t8192_ds4_build_v3.py", "--mode", "planes", "--planes-dir", str(p632.COMPACT_MANIFEST), "--ref-dir", str(p632.TEACHER/"t8192_eval"), "--corpus", str(p632.CORPUS), "--meta-dir", str(p632.MODEL), "--local-dir", str(p632.MODEL), "--out", str(out), "--cand-pos-limit", "1024", "--count", str(len(wins)), "--chunk", "64", "--mb", str(microbatch), "--windows", ",".join(map(str,wins)), "--tag", run_id]
        os.chdir(p632.TEACHER)
        with p632.force_weights_only_torch_loads() as weights_only_stats:
            rc = int(builder.main() or 0)
    finally:
        sys.argv = original_argv; os.chdir(original_cwd); base.retire_scratch(cache)
    if rc:
        raise RuntimeError(f"canonical builder rc={rc}")
    if cache.exists() and any(cache.iterdir()):
        raise RuntimeError("rail scratch not retired")
    expected_keys = {(r["layer"],r["expert"],r["projection"]) for r in manifest["rows"]}
    if set(overlay["applied"]) != expected_keys:
        raise RuntimeError("uniform application coverage drift")
    pp = json.loads(progress.read_text())
    if (pp.get("completed_layers") != list(range(43))
            or pp.get("mmap_completed_layers") != [0, 1, 2]
            or pp.get("overlay_full_completed_layers") != list(range(3, 43))
            or pp.get("local_stage_retired") is not True
            or pp.get("mmap_loader_mode") != "torch-mmap"):
        raise RuntimeError("canonical layer/mmap coverage drift")
    reduced = p632.reduce_outputs(out, wins, classes)
    # Fail closed against the sealed P951 candidate at the tensor level.  Output
    # files carry task-local progress hashes, so raw file hashes may differ even
    # when every decision-bearing float is bit-identical.
    p951_out = Path("${SPARK_HOME}/missions/P951_TRUE_C_BALANCED64_t_463a3c8e_s1/out/P951_INDEPENDENT_TERMINAL_TRUE_C_80_OF_80")
    numerical_rows = []
    maximum_absolute_kld_delta = 0.0
    for win in wins:
        current_payload = p632.torch.load(out / f"kld_win{win}.pt", map_location="cpu", weights_only=False)
        sealed_payload = p632.torch.load(p951_out / f"kld_win{win}.pt", map_location="cpu", weights_only=False)
        delta = float((current_payload["kld"].double() - sealed_payload["kld"].double()).abs().max())
        maximum_absolute_kld_delta = max(maximum_absolute_kld_delta, delta)
        numerical_rows.append({"win": win, "maximum_absolute_kld_delta": delta})
    if maximum_absolute_kld_delta > 1e-12:
        raise RuntimeError(f"P963 decision-scale numerical drift {maximum_absolute_kld_delta}")
    baseline = json.loads(BASELINE.read_text())
    summaries, deltas = mechanics.six_class_summary(reduced, wins, classes, baseline, p632)
    global_delta = p632.paired_delta(reduced["per_window"], baseline["per_window"], wins, "global")
    old = json.loads(OLD_CORRUPT_ANCHOR.read_text())
    if old.get("window_ids") != wins or old.get("measurement_label") != "balanced64_v1":
        raise RuntimeError("P783 old-anchor BALANCED64 basis drift")
    old_paired = {
        "global": p632.paired_delta(reduced["per_window"], old["outputs"]["per_window"], wins, "global"),
        "six_classes": {
            label: p632.paired_delta(reduced["per_window"], old["outputs"]["per_window"], wins, label)
            for label in FROZEN_CLASSES
        },
    }
    first = reduced["per_window"][0]
    first_window_sanity = {
        "status": "PASS" if int(first.get("win", -1)) == wins[0] and math.isfinite(float(first.get("mean", math.nan))) and 0.0 <= float(first["mean"]) <= 2.0 else "FAIL",
        "expected_window": wins[0], "observed_window": first.get("win"),
        "source_class": first.get("source_class"), "mean": first.get("mean"),
        "rule": "exact first BALANCED64 id; finite nonnegative KL <=2.0",
    }
    if first_window_sanity["status"] != "PASS":
        raise RuntimeError(f"first-window sanity failed: {first_window_sanity}")
    corrected = None
    correction_ref = None
    if measurement == "fast64":
        correction = json.loads(FAST_CORRECTION.read_text())
        if correction.get("status") != "PASS" or correction.get("window_ids") != wins:
            raise RuntimeError("fast64 correction contract drift")
        def apply_correction(summary: dict[str, Any], label: str) -> dict[str, Any]:
            row = correction["factors"][label]
            measured = float(summary["mean"])
            factor = float(row["factor"])
            factor_error = float(row["factor_absolute_error"])
            return {
                "source_class": label,
                "fast64_mean": measured,
                "correction_factor": factor,
                "correction_factor_absolute_error": factor_error,
                "full512_equivalent_estimate": measured * factor,
                "calibration_absolute_error": abs(measured) * factor_error,
                "fast64_window_mean_se": summary["window_mean_se"],
                "n_windows": summary["n_windows"],
            }
        corrected = {
            "global": apply_correction(reduced["global"], "global"),
            "six_classes": {label: apply_correction(summaries[label], label) for label in FROZEN_CLASSES},
        }
        correction_ref = {"path": str(FAST_CORRECTION), "sha256": sha256(FAST_CORRECTION)}
    completed = time.time()
    if claim_snapshot()[0] != claim_raw_before:
        raise RuntimeError("claim changed during P770 rail")
    stage_rows = overlay["stage_rows"]
    instrument = {
        "lineage": "P963 pipeline-only acceleration of the sealed P951 exact TRUE-C candidate on the parity-proven BALANCED64 rail", "p632_exact_scorer_sha256": sha256(P632_SCORE),
        "canonical_reader_sha256": p632.CANONICAL_SHA256[p632.CANONICAL_READER], "canonical_builder_sha256": p632.CANONICAL_SHA256[p632.CANONICAL_BUILDER],
        "canonical_loader_sha256": p632.CANONICAL_SHA256[p632.LOADER_SOURCE], "runner_sha256": sha256(Path(__file__)), "adapter_sha256": sha256(OVERLAY_ADAPTER),
        "inventory_sha256": manifest["inventory_sha256"], "assignment_sha256": manifest["assignment_sha256"], "identity_set_sha256": manifest["identity_set_sha256"],
        "base_assignment_sha256": PINS["base_assignment"], "base_wire_manifest_sha256": PINS["base_wire"], "baseline_parity_receipt_sha256": sha256(PARITY),
        "environment_contract": env_contract, "attention": "eager", "microbatch": microbatch, "chunk_size": 64, "layer_overlap": True, "torch_load_safety": weights_only_stats,
    }
    full_wire = manifest["coverage_layers"] == list(range(43))
    receipt = {
        "schema": "p963-accelerated-terminal-true-c-balanced64-v1", "result_label": "P963_ACCELERATED_EXACT_P951_TRUE_C_80_OF_80", "wire_semantic_label": "WIRE_C_TRUE_C_f521_T_PRE_REPAIR_UNDOSED", "status": ("PASS_P963_ACCELERATED_EXACT_P951" if full_wire else "FAIL_INCOMPLETE_TRUE_C_F521_T_BALANCED64_V1"), "task_id": TASK, "canonical_task_id": CANONICAL_TASK, "host": HOST, "anchor": anchor,
        "measurement": measurement, "measurement_label": "balanced64_v1", "coverage_status": "WIRE_C_TRUE_C_F521_T_EXACT_21472_ON_IMMUTABLE_BASE", "direction": "KL(teacher||candidate)", "support": 8192, "cutoff": 1024,
        "windows": len(wins), "window_ids": wins, "global": reduced["global"], "six_classes": summaries,
        "metric_basis": {"primary": "mean KL(teacher||candidate) over exact BALANCED64_V1 windows, support=8192, cutoff=1024", "top1": "not emitted by the canonical BALANCED64 scorer; no top1 value is inferred or fabricated"},
        "direct_comparison": {"predicted_first_feasible": 0.069132, "delta_vs_predicted_first_feasible": float(reduced["global"]["mean"])-0.069132, "prior_c": 0.118138, "delta_vs_prior_c": float(reduced["global"]["mean"])-0.118138},
        "corrected_full512_equivalent": corrected, "fast64_correction": correction_ref,
        "matched_delta_vs_measured_pre_repair": {"global": global_delta, "six_classes": deltas},
        "matched_delta_vs_prior_wire_a_projection_anchor": old_paired,
        "prior_wire_a_projection_anchor": {"basis":"paired W0-127 projection; not a prior BALANCED64 physical read", "path": str(OLD_CORRUPT_ANCHOR), "sha256": sha256(OLD_CORRUPT_ANCHOR), "global_mean": old["global"]["mean"]},
        "window_manifest": {"path": str(BALANCED64), "sha256": sha256(BALANCED64), "name": "BALANCED64_V1"},
        "first_window_sanity": first_window_sanity,
        "instrument": instrument, "instrument_id_sha256": canonical(instrument),
        "active_overlay_manifest": {"path": os.environ["P885_ACTIVE_MANIFEST"], "sha256": sha256(Path(os.environ["P885_ACTIVE_MANIFEST"]))},
        "active_assignment": {"path": os.environ["P885_ACTIVE_ASSIGNMENT"], "sha256": sha256(Path(os.environ["P885_ACTIVE_ASSIGNMENT"]))},
        "source_physical_manifest": manifest.get("source_physical_manifest"), "compatibility_binding": manifest.get("compatibility_binding"),
        "exactness": {"pack_fraction": manifest.get("pack_fraction"), "changed_cells": manifest["changed_cells"], "identity_set_sha256": manifest["identity_set_sha256"], "active_rows_sha256": manifest["active_rows_sha256"], "zero_substitution": True, "zero_quarantine": True},
        "full_wire": full_wire, "repair_and_dosing_blocked": True, "direct_true_c_measurement": True, "codebook_deviation_disclosure": manifest.get("codebook_deviation_disclosure"),
        "coverage": {"changed_cells_expected": manifest["changed_cells"], "changed_cells_applied": len(overlay["applied"]), "unchanged_cells_bound_to_immutable_base": manifest["unchanged_cells"], "qtip2_cells": manifest["qtip2_cells"], "qtip3_cells": manifest["qtip3_cells"], "coverage_layers": manifest["coverage_layers"], "completed_layers": pp["completed_layers"], "mmap_completed_layers": pp["mmap_completed_layers"], "overlay_full_completed_layers": pp["overlay_full_completed_layers"], "overlay_stage_rows": stage_rows, "overlay_stage_retired": all(bool(x.get("stage_retired")) for x in stage_rows), "immutable_base_mutated": False},
        "throughput": {"elapsed_seconds": completed-started, "windows_per_minute": len(wins)*60/(completed-started), "overlay_transferred_bytes": sum(int(x.get("bytes",0)) for x in stage_rows)},
        "outputs": {"directory": str(out), "window_output_set_sha256": reduced["window_output_set_sha256"], "per_window": reduced["per_window"]},
        "loader_proof": {"progress": str(progress), "progress_sha256": sha256(progress), "sentinel": str(sentinel), "sentinel_sha256": sha256(sentinel), "mode": "mixed exact full-overlay plus torch-mmap immutable base", "double_buffer": True},
        "numerical_validation": {"authority": "sealed P951 output tensors", "authority_output_set_sha256": "3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451", "maximum_absolute_kld_delta": maximum_absolute_kld_delta, "tolerance": 1e-12, "per_window": numerical_rows},
        "claim_sha256": claim_sha, "gpu_snapshot_before": gpu_before, "gpu_snapshot_before_child_exit": gpu_snapshot(own_pid=os.getpid()), "started_unix": started, "completed_unix": completed,
    }
    mechanics.atomic_json(receipt_path, receipt, exclusive=True)
    print(json.dumps({"status": receipt["status"], "anchor": anchor, "global": receipt["global"]["mean"], "six_classes": {k:v["mean"] for k,v in receipt["six_classes"].items()}, "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path), "elapsed_seconds": completed-started}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", choices=("qtip3",), required=True)
    parser.add_argument("--measurement", choices=("balanced64",), default="balanced64")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.environ["P770_ROOT"], os.environ["P770_ANCHOR"] = str(ROOT), args.anchor
    adapter = load_module(f"p770_preflight_{args.anchor}", OVERLAY_ADAPTER)
    if args.preflight_only:
        doc = adapter.preflight_manifest()
        public = {k:v for k,v in doc.items() if k not in ("rows","by_layer")}
        print(json.dumps({"status":"PASS_P770_PREFLIGHT","anchor":args.anchor,**public},sort_keys=True))
        return 0
    return run(args.anchor, args.measurement)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        ROOT.joinpath("receipts").mkdir(parents=True, exist_ok=True)
        failure = ROOT / f"receipts/P937_FAILURE_{int(time.time())}.json"
        failure.write_text(json.dumps({"schema":"p937-failure-v1","status":"FAIL_CLOSED","task_id":TASK,"error_type":type(exc).__name__,"error":str(exc),"created_unix":time.time()},indent=2,sort_keys=True)+"\n")
        raise
