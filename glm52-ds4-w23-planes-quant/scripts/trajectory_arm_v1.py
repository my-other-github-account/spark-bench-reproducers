#!/usr/bin/env python3
"""Prepare clean BIN T dose plans and score the public trajectory objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" not in row:
                raise ValueError(f"{path}:{line_no}: missing task_id")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.train)
    heldout_payload = json.loads(args.heldout_ids.read_text(encoding="utf-8"))
    heldout_values = heldout_payload.get("heldout_ids", heldout_payload)
    if not isinstance(heldout_values, list):
        raise ValueError("heldout IDs must be a JSON list or {'heldout_ids': [...]} object")
    heldout = {str(value) for value in heldout_values}
    train_ids = {str(row["task_id"]) for row in rows}
    overlap = sorted(heldout & train_ids)
    if overlap:
        raise ValueError(f"contamination: held-out task IDs appear in train rows: {overlap}")

    classes: dict[str, int] = {}
    for row in rows:
        label = str(row.get("class", "unlabeled"))
        classes[label] = classes.get(label, 0) + 1

    plan = {
        "schema": "bin-t-clean-trajectory-plan-v1",
        "seed": args.seed,
        "steps": args.steps,
        "trajectory_weight": args.trajectory_weight,
        "class_schedule": "code/reasoning 50/50 unless an explicitly sealed plan says otherwise",
        "source_rule": "every arm starts independently from the same immutable checkpoint",
        "train_rows": len(rows),
        "train_task_ids": len(train_ids),
        "train_class_counts": classes,
        "heldout_ids": sorted(heldout),
        "zero_overlap": True,
        "inputs": {
            "train_sha256": sha256(args.train),
            "heldout_ids_sha256": sha256(args.heldout_ids),
        },
        "gates": [
            "candidate-own rollouts at frozen budgets",
            "prompt-macro teacher NLL and approval",
            "reasoning-length ratio and null/cap counts",
            "spot32 all-class static mean-KLD safety",
            "fixed allocation and exact package bytes",
        ],
    }
    write_json(args.output, plan)
    print(json.dumps(plan, sort_keys=True))
    return 0


def score(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.rows)
    by_task: dict[str, dict[str, float]] = {}
    for line_no, row in enumerate(rows, 1):
        task_id = str(row["task_id"])
        if task_id in by_task:
            raise ValueError(f"duplicate task_id in loss ledger: {task_id}")
        try:
            mean_kld = float(row["mean_kld"])
            hard_nll = float(row["trajectory_hard_nll"])
        except KeyError as exc:
            raise ValueError(f"row {line_no}: missing {exc.args[0]}") from exc
        if not math.isfinite(mean_kld) or not math.isfinite(hard_nll):
            raise ValueError(f"row {line_no}: non-finite loss")
        by_task[task_id] = {
            "mean_kld": mean_kld,
            "trajectory_hard_nll": hard_nll,
            "objective": mean_kld + args.trajectory_weight * hard_nll,
        }

    output = {
        "schema": "bin-t-trajectory-objective-v1",
        "trajectory_weight": args.trajectory_weight,
        "tasks": len(by_task),
        "macro": {
            "mean_kld": statistics.fmean(row["mean_kld"] for row in by_task.values()),
            "trajectory_hard_nll": statistics.fmean(
                row["trajectory_hard_nll"] for row in by_task.values()
            ),
            "objective": statistics.fmean(row["objective"] for row in by_task.values()),
        },
        "input_sha256": sha256(args.rows),
    }
    write_json(args.output, output)
    print(json.dumps(output, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="seal and validate a contamination-free dose plan")
    prep.add_argument("--train", type=Path, required=True)
    prep.add_argument("--heldout-ids", type=Path, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--steps", type=int, nargs="+", required=True)
    prep.add_argument("--trajectory-weight", type=float, default=0.25)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=prepare)

    scoring = sub.add_parser("score", help="compute mean_KLD + weight * trajectory_hard_NLL")
    scoring.add_argument("--rows", type=Path, required=True)
    scoring.add_argument("--trajectory-weight", type=float, default=0.25)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.set_defaults(func=score)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.trajectory_weight < 0:
        parser.error("trajectory weight must be non-negative")
    if args.command == "prepare" and any(step <= 0 for step in args.steps):
        parser.error("steps must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
