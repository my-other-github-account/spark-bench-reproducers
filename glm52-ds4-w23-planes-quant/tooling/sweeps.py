#!/usr/bin/env python3
"""Print or serialize held-out probe trajectories from BINREPAIR JSONL ledgers."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="JSONL paths or shell-independent globs")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per ledger")
    return parser.parse_args()


def expand(patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        if matches:
            found.update(Path(item) for item in matches)
        else:
            found.add(Path(pattern).expanduser())
    return sorted(found)


def summarize(path: Path) -> dict[str, Any]:
    probes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    max_step = 0
    events = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        events += 1
        step = int(row.get("step", 0) or 0)
        max_step = max(max_step, step)
        if row.get("event") == "probe" and step > 0:
            probes[step].append(row)

    trajectory = []
    for step in sorted(probes):
        rows = probes[step]
        deltas = [float(row["delta_pct"]) for row in rows]
        baselines = [float(row["baseline"]) for row in rows]
        repaired = [float(row["kld"]) for row in rows]
        baseline_sum = sum(baselines)
        trajectory.append(
            {
                "step": step,
                "n": len(rows),
                "wins": sum(delta > 0 for delta in deltas),
                "mean_window_delta_pct": statistics.mean(deltas),
                "pooled_kld_delta_pct": (
                    100.0 * (baseline_sum - sum(repaired)) / baseline_sum
                    if baseline_sum
                    else None
                ),
                "baseline_mean": statistics.mean(baselines),
                "repaired_mean": statistics.mean(repaired),
                "windows": [
                    {
                        "win": int(row["win"]),
                        "delta_pct": float(row["delta_pct"]),
                    }
                    for row in rows
                ],
            }
        )
    return {
        "ledger": path.name,
        "events": events,
        "max_step": max_step,
        "trajectory": trajectory,
    }


def main() -> int:
    args = parse_args()
    for path in expand(args.paths):
        result = summarize(path)
        if args.json:
            print(json.dumps(result, sort_keys=True))
            continue
        print(f"== {result['ledger']} max_step={result['max_step']}")
        if not result["trajectory"]:
            print("  no completed post-step probe panel")
        for panel in result["trajectory"]:
            windows = " ".join(
                f"w{row['win']}{row['delta_pct']:+.2f}"
                for row in panel["windows"]
            )
            print(
                f"  step{panel['step']}: {windows} "
                f"MEAN_WINDOW={panel['mean_window_delta_pct']:+.4f}% "
                f"POOLED_KLD={panel['pooled_kld_delta_pct']:+.4f}% "
                f"wins={panel['wins']}/{panel['n']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
