#!/usr/bin/env python3
"""Reproduce the sealed P623 EARLY_8 reduction without changing its inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P656_EARLY8_PUBLIC_TASK_s6")
P623 = Path("$HOME/run-bundles/P623_BANANA_SMASHER_BASELINE_PUBLIC_TASK_s6")
P623_SCORER = ROOT / "code/score_p623.py"
P651_READER = ROOT / "code/p651_baseline_parity.py"
OUT = ROOT / "receipts/P623_BASELINE_PARITY_EARLY8.json"
VIEW = ROOT / "inputs/P623_BASELINE_FULL512_VIEW.json"
EXPECTED = {
    "p623_scorer": "844d7e06c5c221e4138a4be931f61e9616ec6ac46123aba1bdd02902b58dffb9",
    "p651_reader": "c6f13c3e82a9d1a21ac7f50a3ab0f59239158b7736cd077f28c751087fca725c",
    "p623_early8": "cb7baf7f681528dc2c69e39a313eee2da84789b9f883313274ced18e30090510",
    "p623_canary": "d246c4ad39cad0c797eb9f41ba0cf5025c9a7da58c552b5026384bf2093cb307",
    "p623_seal": "e508ef87cb95fcbbb2962416a54110fd1877612385bf84607d9e0fc4b58a0faa",
    "p623_source": "c3ba83fddf8f39d4b300c2baf8ad242bfdef21d3a90ac758b005fd01b078d3d5",
    "labels": "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88",
    "window_contract": "91a33069d7d2f5648d63ef10b4a11eb122dbce740eec2ac9acd0bc202325fbad",
}
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("p656_exact_p623_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_equal(actual: float | None, expected: float | None, name: str) -> float:
    if actual is None or expected is None:
        if actual is not expected:
            raise RuntimeError(f"{name}: null mismatch observed={actual!r} expected={expected!r}")
        return 0.0
    delta = float(actual) - float(expected)
    if delta != 0.0:
        raise RuntimeError(f"{name}: exact mismatch observed={actual!r} expected={expected!r} delta={delta!r}")
    return delta


def main() -> int:
    started = time.time()
    paths = {
        "p623_scorer": P623_SCORER,
        "p651_reader": P651_READER,
        "p623_early8": P623 / "out/EARLY_8.json",
        "p623_canary": P623 / "receipts/CANARY.json",
        "p623_seal": P623 / "receipts/SEAL.json",
        "p623_source": P623 / "inputs/SOURCE_FULL512_RECEIPT.json",
        "labels": P623 / "inputs/BQ3_STEP0_PER_CLASS.json",
        "window_contract": P623 / "inputs/WINDOW_CONTRACT.json",
    }
    observed_hashes = {name: sha256(path) for name, path in paths.items()}
    drift = {name: {"expected": EXPECTED[name], "observed": got} for name, got in observed_hashes.items() if got != EXPECTED[name]}
    if drift:
        raise RuntimeError(f"canonical input hash drift: {drift}")
    p623 = load_module(P623_SCORER)
    source = json.loads(paths["p623_source"].read_text())
    sealed = json.loads(paths["p623_early8"].read_text())
    rows = source["outputs"]["per_window"]
    if len(rows) != 512 or [int(row["win"]) for row in rows] != list(range(512)):
        raise RuntimeError("P623 source window surface drift")
    tensors: list[torch.Tensor] = []
    verified_rows = []
    for expected in rows[:8]:
        win = int(expected["win"])
        tensor, mean = p623.load_and_verify(P623 / "raw" / f"kld_win{win}.pt", expected)
        tensors.append(tensor)
        if mean != float(expected["mean"]):
            raise RuntimeError(f"P623 per-window mean mismatch win={win}")
        verified_rows.append({
            "win": win,
            "source_class": str(expected["source_class"]),
            "mean": mean,
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
        })
    joined = torch.cat(tensors)
    global_mean = float(joined.mean())
    by_class: dict[str, dict] = {}
    deltas = {"global": exact_equal(global_mean, float(sealed["global"]["mean_kld"]), "global")}
    for cls in CLASSES:
        selected = [tensors[i] for i, row in enumerate(verified_rows) if row["source_class"] == cls]
        mean = float(torch.cat(selected).mean()) if selected else None
        expected = sealed["by_class"][cls]["mean_kld"]
        deltas[cls] = exact_equal(mean, expected, f"by_class.{cls}")
        by_class[cls] = {
            "mean_kld": mean,
            "window_count": len(selected),
            "position_count": len(selected) * 1024,
        }
    view = {
        "schema": "p656-p623-baseline-view-v1",
        "status": "PASS_EXACT_VIEW_OF_SEALED_P623_SOURCE",
        "source_receipt": str(paths["p623_source"]),
        "source_receipt_sha256": EXPECTED["p623_source"],
        "direction": "KL(teacher||candidate)",
        "support": 8192,
        "cutoff": 1024,
        "per_window": rows,
        "global": source["global"],
        "by_class": source["by_class"],
    }
    atomic_json(VIEW, view)
    receipt = {
        "schema": "p656-p623-unchanged-input-early8-parity-v1",
        "status": "PASS_EXACT_SAME_HOST_INSTRUMENT",
        "task_id": TASK,
        "host": "compute-node-6",
        "window_count": 8,
        "window_ids": list(range(8)),
        "direction": "KL(teacher||candidate)",
        "support": 8192,
        "cutoff": 1024,
        "global_mean_kld": global_mean,
        "by_class": by_class,
        "per_window": verified_rows,
        "exact_deltas": deltas,
        "maximum_absolute_window_mean_delta": max(abs(x) for x in deltas.values()),
        "pinned_p623_scorer": str(P623_SCORER),
        "pinned_p623_scorer_sha256": EXPECTED["p623_scorer"],
        "pinned_p651_reader": str(P651_READER),
        "pinned_p651_reader_sha256": EXPECTED["p651_reader"],
        "sealed_p623_early8": str(paths["p623_early8"]),
        "sealed_p623_early8_sha256": EXPECTED["p623_early8"],
        "p623_source_receipt": str(paths["p623_source"]),
        "p623_source_receipt_sha256": EXPECTED["p623_source"],
        "p623_canary_sha256": EXPECTED["p623_canary"],
        "p623_seal_sha256": EXPECTED["p623_seal"],
        "class_map_sha256": EXPECTED["labels"],
        "window_contract_sha256": EXPECTED["window_contract"],
        "baseline_view": str(VIEW),
        "baseline_view_sha256": sha256(VIEW),
        "elapsed_seconds": time.time() - started,
        "created_unix": time.time(),
    }
    if OUT.exists():
        raise RuntimeError(f"once-only parity receipt exists: {OUT}")
    atomic_json(OUT, receipt)
    print(json.dumps({"status": receipt["status"], "global": global_mean, "max_abs_delta": receipt["maximum_absolute_window_mean_delta"], "receipt": str(OUT), "receipt_sha256": sha256(OUT)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = ROOT / "receipts/P623_BASELINE_PARITY_MISMATCH.json"
        if not failure.exists():
            atomic_json(failure, {
                "schema": "p656-p623-unchanged-input-early8-parity-failure-v1",
                "status": "FAIL_CLOSED_BASELINE_MISMATCH",
                "task_id": TASK,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_unix": time.time(),
            })
        raise
