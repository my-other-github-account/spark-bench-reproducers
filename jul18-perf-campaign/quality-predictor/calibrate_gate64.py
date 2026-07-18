#!/usr/bin/env python3
"""Fit and apply a paired gate64 -> full512 linear quality model."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two paired rows")
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        raise ValueError("correlation undefined for a constant column")
    return sum(a * b for a, b in zip(dx, dy)) / denom


def fit_rows(rows: list[tuple[float, float]]) -> dict:
    if len(rows) < 2:
        raise ValueError("need at least two paired rows")
    gate = [r[0] for r in rows]
    full = [r[1] for r in rows]
    mx, my = statistics.fmean(gate), statistics.fmean(full)
    var = sum((x - mx) ** 2 for x in gate)
    if var == 0:
        raise ValueError("gate64 column is constant")
    slope = sum((x - mx) * (y - my) for x, y in rows) / var
    intercept = my - slope * mx
    ratios = [y / x for x, y in rows if x != 0]
    return {
        "schema": "gate64-full512-linear-fit-v1",
        "n": len(rows),
        "intercept": intercept,
        "slope": slope,
        "pearson_r": pearson(gate, full),
        "spearman_rho": pearson(average_ranks(gate), average_ranks(full)),
        "median_full512_over_gate64": statistics.median(ratios) if ratios else None,
        "gate64_domain": [min(gate), max(gate)],
        "paired_rows_required": True,
    }


def read_rows(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"gate64", "full512"} <= set(reader.fieldnames):
            raise ValueError("CSV must contain gate64 and full512 columns")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                rows.append((float(row["gate64"]), float(row["full512"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric row at CSV line {line}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pfit = sub.add_parser("fit")
    pfit.add_argument("csv", type=Path)
    pfit.add_argument("--output", type=Path, required=True)
    ppred = sub.add_parser("predict")
    ppred.add_argument("fit", type=Path)
    ppred.add_argument("gate64", type=float)
    ppred.add_argument("--allow-extrapolation", action="store_true")
    args = parser.parse_args()

    if args.command == "fit":
        result = fit_rows(read_rows(args.csv))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    model = json.loads(args.fit.read_text())
    lo, hi = model["gate64_domain"]
    extrapolated = not (lo <= args.gate64 <= hi)
    if extrapolated and not args.allow_extrapolation:
        raise SystemExit(
            f"refusing extrapolation: {args.gate64} outside fitted domain [{lo}, {hi}]"
        )
    result = {
        "gate64": args.gate64,
        "predicted_full512": model["intercept"] + model["slope"] * args.gate64,
        "extrapolated": extrapolated,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
