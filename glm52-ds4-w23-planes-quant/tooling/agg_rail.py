#!/usr/bin/env python3
"""Merge fleet-sharded eval ledgers and fail closed on gaps or conflicts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="shard JSONL paths or globs")
    parser.add_argument("--expected-windows", type=int, default=512)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--include-partial-metrics",
        action="store_true",
        help="print preview metrics for incomplete coverage (never a seal)",
    )
    parser.add_argument("--output", type=Path)
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


def equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("win", "kld", "ledger_ref", "manifest_md5")
    return all(left.get(key) == right.get(key) for key in keys)


def main() -> int:
    args = parse_args()
    files = expand(args.paths)
    if not files:
        raise ValueError("no shard ledgers matched")

    rows: dict[int, dict[str, Any]] = {}
    source_of: dict[int, str] = {}
    identities: dict[str, set[str]] = {
        "manifest_md5": set(),
        "corpus_md5": set(),
    }
    for path in files:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            for key in identities:
                if row.get(key):
                    identities[key].add(str(row[key]))
            if row.get("event") != "baseline":
                continue
            win = int(row["win"])
            if win in rows and not equivalent(rows[win], row):
                raise ValueError(
                    f"conflicting duplicate window {win}: "
                    f"{source_of[win]} vs {path.name}"
                )
            rows[win] = row
            source_of[win] = path.name

    for key, values in identities.items():
        if len(values) > 1:
            raise ValueError(f"mixed {key} identities: {sorted(values)}")

    expected = set(range(args.expected_windows))
    observed = set(rows)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(f"unexpected windows: {unexpected}")

    measured = [float(rows[win]["kld"]) for win in sorted(rows)]
    references = [
        float(rows[win]["ledger_ref"])
        for win in sorted(rows)
        if rows[win].get("ledger_ref") is not None
    ]
    finite = all(math.isfinite(value) for value in measured + references)
    if not finite:
        raise ValueError("non-finite KLD value")

    result: dict[str, Any] = {
        "format": "binrepair-rail-aggregate-v1",
        "expected_windows": args.expected_windows,
        "observed_windows": len(rows),
        "complete": not missing,
        "missing_windows": missing,
        "sources": [
            {"file": path.name, "sha256": digest(path)} for path in files
        ],
    }
    for key, values in identities.items():
        if values:
            result[key] = next(iter(values))
    include_metrics = not missing or args.include_partial_metrics
    if include_metrics:
        result["measured_kld_mean"] = (
            sum(measured) / len(measured) if measured else None
        )
    if len(references) == len(measured) and measured and include_metrics:
        per_window = [
            100.0 * (float(rows[win]["ledger_ref"]) - float(rows[win]["kld"]))
            / float(rows[win]["ledger_ref"])
            for win in sorted(rows)
        ]
        result.update(
            {
                "reference_kld_mean": sum(references) / len(references),
                "mean_window_delta_pct": sum(per_window) / len(per_window),
                "pooled_kld_delta_pct": (
                    100.0 * (sum(references) - sum(measured)) / sum(references)
                ),
                "windows_improved": sum(delta > 0 for delta in per_window),
            }
        )

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, args.output)
    print(text, end="")
    if missing and not args.allow_partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
