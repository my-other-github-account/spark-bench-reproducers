#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Tuple

CELLS = {
    "canonical_4_4": (4.0, 4.0),
    "relax_min_10_4": (10.0, 4.0),
    "relax_factor_4_10": (4.0, 10.0),
    "relax_both_10_10": (10.0, 10.0),
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: pathlib.Path) -> Dict[str, List[Dict[str, Any]]]:
    doc = json.loads(path.read_text())
    return doc["eval"]


def passed(row: Dict[str, Any], scope: str) -> bool:
    if scope == "base":
        return row.get("base_status") == "pass"
    return row.get("base_status") == row.get("plus_status") == "pass"


def classify(paths: Dict[str, pathlib.Path]) -> Dict[str, Any]:
    docs = {name: load(path) for name, path in paths.items()}
    task_sets = {name: set(doc) for name, doc in docs.items()}
    if len(set(map(frozenset, task_sets.values()))) != 1:
        raise ValueError("timing cells have different task sets")
    task_ids = sorted(next(iter(task_sets.values())))
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {}
    for scope in ["base", "plus"]:
        counts = {"stable_pass": 0, "stable_semantic_fail": 0, "timing_only": 0, "marginal_or_nondeterministic": 0}
        for task_id in task_ids:
            lengths = {name: len(doc[task_id]) for name, doc in docs.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError("sample count mismatch for %s: %r" % (task_id, lengths))
            for sample_index in range(next(iter(lengths.values()))):
                states = {name: passed(doc[task_id][sample_index], scope) for name, doc in docs.items()}
                canonical = states["canonical_4_4"]
                relaxed = [states[x] for x in ["relax_min_10_4", "relax_factor_4_10", "relax_both_10_10"]]
                if canonical and all(relaxed):
                    label = "stable_pass"
                elif not canonical and all(relaxed):
                    label = "timing_only"
                elif not canonical and not any(relaxed):
                    label = "stable_semantic_fail"
                else:
                    label = "marginal_or_nondeterministic"
                counts[label] += 1
                if label not in ["stable_pass", "stable_semantic_fail"]:
                    rows.append({
                        "task_id": task_id,
                        "sample_index": sample_index,
                        "scope": scope,
                        "classification": label,
                        "pass_states": states,
                        "canonical_fail_tests": {
                            "base": docs["canonical_4_4"][task_id][sample_index].get("base_fail_tests", []),
                            "plus": docs["canonical_4_4"][task_id][sample_index].get("plus_fail_tests", []),
                        },
                    })
        summary[scope] = counts
    return {
        "schema": "p968-timing-adjudication-v1",
        "method": "Canonical EvalPlus min_time=4, gt_factor=4 is compared against independent full reruns at (8,4), (4,8), and (8,8). A canonical failure that passes all three relaxed cells is timing-only; mixed cells are marginal/nondeterministic; failure in all cells is semantic/stable under this instrument.",
        "cells": {name: {"min_time_limit": values[0], "gt_time_limit_factor": values[1], "path": str(paths[name]), "sha256": sha256(paths[name])} for name, values in CELLS.items()},
        "summary": summary,
        "nonstable_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in CELLS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in CELLS}
    report = classify(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "out": str(args.out), "summary": report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
