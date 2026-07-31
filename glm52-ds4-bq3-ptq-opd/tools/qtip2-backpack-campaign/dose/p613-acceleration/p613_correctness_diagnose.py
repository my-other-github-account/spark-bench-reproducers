#!/usr/bin/env python3
"""P613 phase-1 correctness isolation using one resident real model on s5w."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path("$HOME/run-bundles/P613_ACTCACHE_ACCEL_PUBLIC_TASK_s5w")
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
import p613_profile_benchmark as P


def timed_eval(module, student, corpus, directory: Path, first_train: int, wins: list[int]):
    P.set_cache_identity(module, first_train=first_train, directory=directory)
    cache = module.ActCache(student)
    started = time.perf_counter()
    values = module.batch_kld_values(student, corpus, cache, wins)
    torch.cuda.synchronize()
    return values, time.perf_counter() - started


def deltas(a, b):
    rows = [float(y) - float(x) for x, y in zip(a, b)]
    return rows, max(map(abs, rows))


def main():
    P.ensure_inputs()
    P.set_env()
    profile = json.loads((ROOT / "PROFILE.json").read_text())
    legacy = P.load_module("p613_diag_legacy", CODE / "base_binrepair_e2e.py")
    accel = P.load_module("p613_diag_accel", CODE / "base_binrepair_e2e_accel.py")
    from banana_smasher_physical_surface import BananaSmasherPhysicalExperts
    legacy.T.TrainableExperts = BananaSmasherPhysicalExperts
    legacy.T.PILOT = tuple(range(43))
    student = legacy.T.Student()
    corpus = legacy.T.load_corpus()
    wins = P.PROFILE_WINS

    baseline_l0 = profile["production_batch"]["kld"]
    accel_l0_b4, accel_l0_seconds = timed_eval(
        accel, student, corpus, P.SEALED_L0, 0, wins
    )
    l0_delta, l0_worst = deltas(baseline_l0, accel_l0_b4)

    legacy_l4_dir = P.BENCH / "legacy_cold" / "BR_ACTCACHE_L004"
    accel_l4_dir = P.BENCH / "accel_cold" / "BR_ACTCACHE_L004"
    legacy_l4_kld, legacy_l4_seconds = timed_eval(
        accel, student, corpus, legacy_l4_dir, 4, wins
    )
    accel_l4_kld, accel_l4_seconds = timed_eval(
        accel, student, corpus, accel_l4_dir, 4, wins
    )
    l4_delta, l4_worst = deltas(legacy_l4_kld, accel_l4_kld)

    tensor_rows = []
    for win in P.TRAIN_WINS:
        before = torch.load(
            legacy_l4_dir / f"win{win}.pt", map_location="cpu", mmap=True,
            weights_only=False,
        )["h"]
        after = torch.load(
            accel_l4_dir / f"win{win}.pt", map_location="cpu", mmap=True,
            weights_only=False,
        )["h"]
        diff = (after.float() - before.float()).abs()
        tensor_rows.append({
            "win": win,
            "exact_fraction": float((before == after).float().mean()),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "allclose_atol_0p125_rtol0": bool(torch.allclose(before, after, atol=0.125, rtol=0)),
        })

    previous_b8 = json.loads((ROOT / "VALIDATION.json").read_text())["directional_eval"]
    result = {
        "schema": "p613-correctness-isolation-v1",
        "task_id": P.TASK,
        "host": __import__("os").uname().nodename,
        "completed_unix": time.time(),
        "same_batch_shape_loader_scorer_isolation": {
            "wins": wins,
            "legacy_l0_batch4_kld": baseline_l0,
            "accelerated_mmap_l0_batch4_kld": accel_l0_b4,
            "delta": l0_delta,
            "worst_abs_delta": l0_worst,
            "seconds": accel_l0_seconds,
            "pass_1e_5": l0_worst <= 1e-5,
        },
        "batch_shape_change_existing_evidence": {
            "baseline_batch": 4,
            "accelerated_batch": previous_b8["selected_batch"],
            "worst_abs_delta": previous_b8["first4_worst_abs_delta"],
        },
        "cold_cache_decision_impact_same_suffix_batch4": {
            "wins": wins,
            "serial_l4_kld": legacy_l4_kld,
            "batched_l4_kld": accel_l4_kld,
            "delta": l4_delta,
            "worst_abs_delta": l4_worst,
            "serial_seconds": legacy_l4_seconds,
            "batched_seconds": accel_l4_seconds,
            "pass_1e_5": l4_worst <= 1e-5,
        },
        "cold_cache_tensor_drift": tensor_rows,
        "hypothesis": (
            "If same-batch mmap/scorer delta passes while batch4-vs-batch8 fails, "
            "the directional defect is forward batch-shape numerical drift, not mmap. "
            "The L4 comparison separately measures whether batched cold-cache drift "
            "survives into decision rows under a matched suffix batch shape."
        ),
    }
    sha = P.atomic_json(ROOT / "CORRECTNESS_DIAGNOSIS.json", result)
    print(json.dumps({"path": str(ROOT / "CORRECTNESS_DIAGNOSIS.json"), "sha256": sha, **result}, indent=2))


if __name__ == "__main__":
    main()
