#!/usr/bin/env python3
"""Extract DFlash verification/perf markers and forbidden code-path markers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_MARKERS = [
    "DFLASH K=gamma verify",
    "DFlash K=gamma verifier hidden carry active",
    "DFlash fast K=16 FFN/GEMM path active",
]

FORBIDDEN_MARKERS = [
    "force_accept",
    "candidate_posterior",
    "skip_verify",
]

ACCEPT_RE = re.compile(r"accepted=(\d+)/(\d+)")


def scan(paths: list[str]) -> dict[str, object]:
    marker_counts = {marker: 0 for marker in REQUIRED_MARKERS}
    forbidden_counts = {marker: 0 for marker in FORBIDDEN_MARKERS}
    accepted: list[dict[str, object]] = []
    files_scanned = 0

    for raw in paths:
        path = Path(raw)
        candidates = [path]
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            files_scanned += 1
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for marker in marker_counts:
                marker_counts[marker] += text.count(marker)
            for marker in forbidden_counts:
                forbidden_counts[marker] += text.count(marker)
            for match in ACCEPT_RE.finditer(text):
                accepted.append(
                    {
                        "file": str(candidate),
                        "accepted": int(match.group(1)),
                        "total": int(match.group(2)),
                    }
                )

    totals = sorted({item["total"] for item in accepted})
    accepted_values = sorted({item["accepted"] for item in accepted})
    return {
        "files_scanned": files_scanned,
        "required_marker_counts": marker_counts,
        "forbidden_marker_counts": forbidden_counts,
        "forbidden_clean": all(v == 0 for v in forbidden_counts.values()),
        "accepted_count": len(accepted),
        "accepted_totals": totals,
        "accepted_values": accepted_values,
        "has_zero_accept": 0 in accepted_values,
        "has_partial_accept": any(
            0 < int(item["accepted"]) < int(item["total"]) for item in accepted
        ),
        "has_full_accept": any(
            int(item["accepted"]) == int(item["total"]) for item in accepted
        ),
        "accepted_samples": accepted[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = scan(args.paths)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["forbidden_clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
