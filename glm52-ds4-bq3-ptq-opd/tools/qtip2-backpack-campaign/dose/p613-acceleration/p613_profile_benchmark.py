#!/usr/bin/env python3
"""P613 real-model profile + cold activation-cache acceleration benchmark."""
from __future__ import annotations

import gc
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

import torch

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P613_ACTCACHE_ACCEL_PUBLIC_TASK_s5w")
CODE = ROOT / "code"
BENCH = ROOT / "bench"
RESULTS = ROOT / "results"
SEALED_L0 = Path(
    "$HOME/run-bundles/GENESIS_SEAMS_REPAIR_PUBLIC_TASK_s8/"
    "run/basic_harness/BR_ACTCACHE_L000"
)
CORPUS = Path(
    "$HOME/run-bundles/GENESIS_SEAMS_REPAIR_PUBLIC_TASK_s8/"
    "inputs/BASIC_COMBINED_768.json"
)
PACKAGE = ROOT / "inputs/compute-node-wire.example.invalid"
ASSIGNMENT = Path(
    "$HOME/run-bundles/GENESIS_FANIN_PUBLIC_TASK_s8/inputs/NOMINATED_ASSIGNMENT.json"
)
TEACH = Path("$HOME/run-bundles/DS4_TEACHER/t8192_train")
TRAIN_WINS = [20, 21, 22, 23, 24, 25, 26, 27]
PROFILE_WINS = TRAIN_WINS[:4]
EVAL_WINS = list(range(20, 28))
SOURCE_P600_BASELINE_SECONDS_40 = 1995.9726622104645


def atomic_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_sha(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.view(torch.int16).numpy().tobytes()).hexdigest()


def ensure_inputs():
    required = [
        CODE / "base_binrepair_e2e.py",
        CODE / "base_binrepair_e2e_accel.py",
        CODE / "genesis_physical_surface.py",
        PACKAGE,
        ASSIGNMENT,
        CORPUS,
        TEACH,
        SEALED_L0,
    ]
    missing = [str(path) for path in required if not path.exists()]
    for win in sorted(set(EVAL_WINS + TRAIN_WINS)):
        for path in (SEALED_L0 / f"win{win}.pt", TEACH / f"t8192_win{win}.pt"):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing P613 inputs: " + repr(missing))


def set_env():
    os.environ.update({
        "PYTHONHASHSEED": "0",
        "CUDA_MODULE_LOADING": "EAGER",
        "GENESIS_REPAIR_DEVICE": "cuda",
        "GENESIS_REPAIR_ROOT": str(ROOT),
        "GENESIS_REPAIR_MEM_FLOOR_BYTES": str(8 * 1024**3),
        "GENESIS_REPAIR_EVICT": "1",
        "GENESIS_REPAIR_DEQ_CHUNK": "1",
        "GENESIS_REPAIR_NATIVE_CHUNK": "1",
        "GENESIS_REPAIR_EXPERT_RESIDENT_SCOPE": "4",
        "GENESIS_PHYSICAL_PACKAGE": str(PACKAGE),
        "GENESIS_ASSIGNMENT": str(ASSIGNMENT),
        "BR_MANIFEST": str(ASSIGNMENT),
        "BR_DELTA_DIR": "$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/delta",
        "BR_VQ3B_DIR": "$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/planes",
        "BR_TRAINABLE": "4,9,10,11,12,13,14,15",
        "BR_TRAIN": ",".join(map(str, TRAIN_WINS)),
        "BR_PROBE": "278,279",
        "BR_STEPS": "16",
        "BR_LR": "0.03",
        "BR_BATCH": "4",
        "BR_PROBE_EVERY": "4",
        "BR_CACHE_ONLY": "0",
        "BR_CACHE_BATCH": "8",
        "BR_CACHE_BUILD_BATCH": "8",
        "BR_CACHE_IO_WORKERS": "4",
        "BR_MAX_HOURS": "8",
        "BR_EARLY_STOP": "999",
        "BR_TEACH": str(TEACH),
        "BR_CORPUS": str(CORPUS),
        "BR_TAG": "p613_profile",
    })


def set_cache_identity(module, *, first_train: int, directory: Path):
    module.FIRST_TRAIN = int(first_train)
    module.CACHE_ID = f"binrepair|{module.AMD5[:12]}|L{first_train}"
    module.ACTCACHE_DIR = directory
    directory.mkdir(parents=True, exist_ok=True)


def profile_production_batch(legacy, student, corpus):
    set_cache_identity(legacy, first_train=0, directory=SEALED_L0)
    acache = legacy.ActCache(student)
    metrics = {
        "activation_cache_load": 0.0,
        "full_model_forward": 0.0,
        "teacher_ref_load_normalize": 0.0,
        "loss_total": 0.0,
    }

    original_load = acache._load
    original_forward = legacy.fast_forward
    original_teacher = legacy.T.teacher_rows
    original_loss = legacy.loss_window

    def timed_load(win):
        started = time.perf_counter()
        try:
            return original_load(win)
        finally:
            torch.cuda.synchronize()
            metrics["activation_cache_load"] += time.perf_counter() - started

    def timed_forward(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_forward(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            metrics["full_model_forward"] += time.perf_counter() - started

    def timed_teacher(win):
        started = time.perf_counter()
        try:
            return original_teacher(win)
        finally:
            torch.cuda.synchronize()
            metrics["teacher_ref_load_normalize"] += time.perf_counter() - started

    def timed_loss(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_loss(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            metrics["loss_total"] += time.perf_counter() - started

    acache._load = timed_load
    legacy.fast_forward = timed_forward
    legacy.T.teacher_rows = timed_teacher
    legacy.loss_window = timed_loss
    started = time.perf_counter()
    try:
        values = legacy.batch_kld_values(student, corpus, acache, PROFILE_WINS)
        torch.cuda.synchronize()
    finally:
        total = time.perf_counter() - started
        legacy.fast_forward = original_forward
        legacy.T.teacher_rows = original_teacher
        legacy.loss_window = original_loss
    metrics["scorer_without_teacher_load"] = max(
        0.0, metrics["loss_total"] - metrics["teacher_ref_load_normalize"]
    )
    metrics["python_and_other"] = max(
        0.0,
        total
        - metrics["activation_cache_load"]
        - metrics["full_model_forward"]
        - metrics["loss_total"],
    )
    metrics["total"] = total
    metrics["seconds_per_window"] = total / len(PROFILE_WINS)
    return metrics, values


def cold_build_benchmark(legacy, accel, student, corpus):
    legacy_dir = BENCH / "legacy_cold" / "BR_ACTCACHE_L004"
    accel_dir = BENCH / "accel_cold" / "BR_ACTCACHE_L004"
    shutil.rmtree(legacy_dir.parent, ignore_errors=True)
    shutil.rmtree(accel_dir.parent, ignore_errors=True)
    set_cache_identity(legacy, first_train=4, directory=legacy_dir)
    set_cache_identity(accel, first_train=4, directory=accel_dir)

    legacy_cache = legacy.ActCache(student)
    started = time.perf_counter()
    # This is the actual old production call shape: list-comprehension get(),
    # so every cold miss invokes build_many([one_window]) separately.
    legacy_hidden = torch.cat([legacy_cache.get(corpus, win) for win in TRAIN_WINS], 0)
    torch.cuda.synchronize()
    legacy_seconds = time.perf_counter() - started
    legacy_hashes = {str(win): tensor_sha(legacy_cache.mem[win]) for win in TRAIN_WINS}
    del legacy_hidden

    torch.cuda.empty_cache()
    accel_cache = accel.ActCache(student)
    started = time.perf_counter()
    accel_hidden = accel_cache.get_many(corpus, TRAIN_WINS)
    torch.cuda.synchronize()
    accel_seconds = time.perf_counter() - started
    accel_hashes = {str(win): tensor_sha(accel_cache.mem[win]) for win in TRAIN_WINS}
    del accel_hidden

    comparisons = []
    for win in TRAIN_WINS:
        before = torch.load(
            legacy_dir / f"win{win}.pt", map_location="cpu", mmap=True,
            weights_only=False,
        )["h"]
        after = torch.load(
            accel_dir / f"win{win}.pt", map_location="cpu", mmap=True,
            weights_only=False,
        )["h"]
        delta = (before.float() - after.float()).abs()
        comparisons.append({
            "win": win,
            "shape": list(before.shape),
            "dtype": str(before.dtype),
            "exact_equal": bool(torch.equal(before, after)),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "legacy_sha256": legacy_hashes[str(win)],
            "accel_sha256": accel_hashes[str(win)],
        })
        del before, after, delta

    # Load the exact same eight sealed files through each loader.
    set_cache_identity(legacy, first_train=4, directory=legacy_dir)
    legacy_load = legacy.ActCache(student)
    started = time.perf_counter()
    loaded = torch.cat([legacy_load.get(corpus, win) for win in TRAIN_WINS], 0)
    torch.cuda.synchronize()
    legacy_load_seconds = time.perf_counter() - started
    del loaded

    set_cache_identity(accel, first_train=4, directory=legacy_dir)
    accel_load = accel.ActCache(student)
    started = time.perf_counter()
    loaded = accel_load.get_many(corpus, TRAIN_WINS)
    torch.cuda.synchronize()
    accel_load_seconds = time.perf_counter() - started
    del loaded

    return {
        "windows": TRAIN_WINS,
        "legacy": {
            "seconds": legacy_seconds,
            "seconds_per_window": legacy_seconds / len(TRAIN_WINS),
            "call_shape": "serial get(win) cold misses",
        },
        "accelerated": {
            "seconds": accel_seconds,
            "seconds_per_window": accel_seconds / len(TRAIN_WINS),
            "call_shape": "single get_many(wins), batch=8",
        },
        "speedup": legacy_seconds / accel_seconds,
        "saved_seconds_per_window": (
            legacy_seconds / len(TRAIN_WINS) - accel_seconds / len(TRAIN_WINS)
        ),
        "load_only": {
            "legacy_seconds_8": legacy_load_seconds,
            "accelerated_seconds_8": accel_load_seconds,
            "speedup": legacy_load_seconds / accel_load_seconds,
        },
        "comparisons": comparisons,
        "all_exact_equal": all(row["exact_equal"] for row in comparisons),
        "worst_max_abs": max(row["max_abs"] for row in comparisons),
    }


def eval_batch_benchmark(accel, student, corpus, baseline_values, baseline_metrics):
    set_cache_identity(accel, first_train=0, directory=SEALED_L0)
    attempts = []
    for batch_size in (len(EVAL_WINS), 4):
        wins = EVAL_WINS[:batch_size]
        cache = accel.ActCache(student)
        torch.cuda.empty_cache()
        gc.collect()
        started = time.perf_counter()
        try:
            values = accel.batch_kld_values(student, corpus, cache, wins)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            attempts.append({"batch_size": batch_size, "status": "PASS", "seconds": seconds})
            row_delta = [
                float(values[index]) - float(baseline_values[index])
                for index in range(len(PROFILE_WINS))
            ]
            return {
                "selected_batch": batch_size,
                "seconds": seconds,
                "seconds_per_window": seconds / batch_size,
                "throughput_speedup_vs_batch4": (
                    baseline_metrics["seconds_per_window"] / (seconds / batch_size)
                ),
                "first4_kld": values[:4],
                "baseline_first4_kld": baseline_values,
                "first4_delta": row_delta,
                "first4_worst_abs_delta": max(map(abs, row_delta)),
                "attempts": attempts,
            }
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            text = f"{type(exc).__name__}: {exc}"
            if "MEMORY_FLOOR_STOP" not in text and "out of memory" not in text.lower():
                raise
            attempts.append({"batch_size": batch_size, "status": "OOM_FALLBACK", "error": text})
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError("no directional evaluation batch fits")


def main():
    started_unix = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "legacy").mkdir(parents=True, exist_ok=True)
    (BENCH / "accel").mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ensure_inputs()
    set_env()
    sys.path.insert(0, str(CODE))

    os.environ["BR_OUTDIR"] = str(BENCH / "legacy")
    legacy = load_module("p613_legacy_base", CODE / "base_binrepair_e2e.py")
    os.environ["BR_OUTDIR"] = str(BENCH / "accel")
    accel = load_module("p613_accel_base", CODE / "base_binrepair_e2e_accel.py")

    from genesis_physical_surface import GenesisPhysicalExperts
    legacy.T.TrainableExperts = GenesisPhysicalExperts
    legacy.T.PILOT = tuple(range(43))

    model_started = time.perf_counter()
    student = legacy.T.Student()
    torch.cuda.synchronize()
    model_seconds = time.perf_counter() - model_started
    corpus = legacy.T.load_corpus()

    profile_metrics, profile_values = profile_production_batch(legacy, student, corpus)
    cold = cold_build_benchmark(legacy, accel, student, corpus)
    eval_batch = eval_batch_benchmark(
        accel, student, corpus, profile_values, profile_metrics
    )

    sinks = {
        "full_model_forward": profile_metrics["full_model_forward"],
        "teacher_ref_load_normalize": profile_metrics["teacher_ref_load_normalize"],
        "scorer_without_teacher_load": profile_metrics["scorer_without_teacher_load"],
        "activation_cache_load": profile_metrics["activation_cache_load"],
        "python_and_other": profile_metrics["python_and_other"],
    }
    top3 = [
        {"rank": rank + 1, "name": name, "seconds": seconds,
         "fraction": seconds / profile_metrics["total"]}
        for rank, (name, seconds) in enumerate(
            sorted(sinks.items(), key=lambda item: item[1], reverse=True)[:3]
        )
    ]

    projected_cold_windows = 40
    cold_saved_minutes = (
        cold["saved_seconds_per_window"] * projected_cold_windows / 60.0
    )
    old_directional_minutes_per_measure = SOURCE_P600_BASELINE_SECONDS_40 / 60.0
    new_directional_minutes_per_measure = (
        eval_batch["seconds_per_window"] * 40 / 60.0
    )
    directional_measure_count = 3
    directional_saved_minutes = (
        old_directional_minutes_per_measure - new_directional_minutes_per_measure
    ) * directional_measure_count
    projected = {
        "cold_unique_windows_per_dose": projected_cold_windows,
        "cold_build_saved_minutes": cold_saved_minutes,
        "directional_measurements_per_24_update_dose": directional_measure_count,
        "old_directional_minutes_per_measure_source_p600": old_directional_minutes_per_measure,
        "new_directional_minutes_per_measure_projected_from_measured_batch": new_directional_minutes_per_measure,
        "directional_saved_minutes": directional_saved_minutes,
        "total_projected_minutes_saved_per_24_update_dose": (
            cold_saved_minutes + directional_saved_minutes
        ),
        "arithmetic": (
            f"40*({cold['legacy']['seconds_per_window']:.6f}-"
            f"{cold['accelerated']['seconds_per_window']:.6f})/60 + 3*("
            f"{SOURCE_P600_BASELINE_SECONDS_40:.6f}/60 - 40*"
            f"{eval_batch['seconds_per_window']:.6f}/60)"
        ),
    }

    profile = {
        "schema": "p613-actcache-profile-v1",
        "task_id": TASK,
        "host": os.uname().nodename,
        "started_unix": started_unix,
        "completed_unix": time.time(),
        "base_model_resident_once": True,
        "base_model_assembly_seconds": model_seconds,
        "production_batch": {
            "wins": PROFILE_WINS,
            "kld": profile_values,
            **profile_metrics,
        },
        "top3_sinks": top3,
        "source_p600_train40_seconds": SOURCE_P600_BASELINE_SECONDS_40,
        "diagnosis": (
            "actcache_load markers are boundaries, not the 3.4-minute sink; "
            "the resident model walk between markers dominates. Cold misses were "
            "nevertheless serialized by get() despite build_many supporting batches."
        ),
    }
    validation = {
        "schema": "p613-actcache-validation-v1",
        "task_id": TASK,
        "host": os.uname().nodename,
        "cold_build": cold,
        "directional_eval": eval_batch,
        "row_match_gate": {
            "cache_exact_required": True,
            "cache_exact_pass": cold["all_exact_equal"],
            "decision_scale_kld_tolerance": 1e-5,
            "decision_scale_kld_pass": eval_batch["first4_worst_abs_delta"] <= 1e-5,
        },
    }
    benchmark = {
        "schema": "p613-actcache-before-after-v1",
        "task_id": TASK,
        "host": os.uname().nodename,
        "cold_build": {
            "before_seconds_per_window": cold["legacy"]["seconds_per_window"],
            "after_seconds_per_window": cold["accelerated"]["seconds_per_window"],
            "speedup": cold["speedup"],
            "gate": ">=3x",
            "gate_pass": cold["speedup"] >= 3.0,
        },
        "mmap_load": cold["load_only"],
        "directional_eval": eval_batch,
        "projected_24_update_dose": projected,
    }

    profile_sha = atomic_json(ROOT / "PROFILE.json", profile)
    validation_sha = atomic_json(ROOT / "VALIDATION.json", validation)
    benchmark_sha = atomic_json(ROOT / "BENCHMARK.json", benchmark)
    progress = json.loads((ROOT / "PROGRESS.json").read_text())
    progress.update({
        "phase": "BENCHMARK_COMPLETE",
        "updated_unix": time.time(),
        "deliverables": {
            "profile_json": {"path": str(ROOT / "PROFILE.json"), "sha256": profile_sha},
            "accelerated_patch": {
                "base": str(CODE / "base_binrepair_e2e_accel.py"),
                "driver": str(CODE / "genesis_basic_repair_accel.py"),
            },
            "validation_8_windows": {
                "path": str(ROOT / "VALIDATION.json"), "sha256": validation_sha,
                "pass": (
                    validation["row_match_gate"]["cache_exact_pass"]
                    and validation["row_match_gate"]["decision_scale_kld_pass"]
                ),
            },
            "before_after": {"path": str(ROOT / "BENCHMARK.json"), "sha256": benchmark_sha},
        },
    })
    progress_sha = atomic_json(ROOT / "PROGRESS.json", progress)
    final = {
        "profile_sha256": profile_sha,
        "validation_sha256": validation_sha,
        "benchmark_sha256": benchmark_sha,
        "progress_sha256": progress_sha,
        "cold_speedup": cold["speedup"],
        "cache_exact_pass": cold["all_exact_equal"],
        "directional_batch": eval_batch["selected_batch"],
        "directional_throughput_speedup": eval_batch["throughput_speedup_vs_batch4"],
        "directional_kld_worst_abs_delta": eval_batch["first4_worst_abs_delta"],
        "projected_minutes_saved_per_24_update_dose": projected[
            "total_projected_minutes_saved_per_24_update_dose"
        ],
    }
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    if cold["speedup"] < 3.0:
        raise RuntimeError(f"cold-build speedup gate failed: {cold['speedup']:.3f}x")
    if not cold["all_exact_equal"]:
        raise RuntimeError("cache tensor exact-match gate failed")
    if eval_batch["first4_worst_abs_delta"] > 1e-5:
        raise RuntimeError(
            "decision-scale KLD delta; scorer/batching path must be fixed: "
            f"{eval_batch['first4_worst_abs_delta']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            atomic_json(ROOT / "results" / "BENCHMARK_FAILURE.json", {
                "schema": "p613-benchmark-failure-v1",
                "task_id": TASK,
                "host": os.uname().nodename,
                "error": traceback.format_exc(),
                "failed_unix": time.time(),
            })
        except Exception:
            pass
        raise
