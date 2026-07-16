#!/usr/bin/env python3
"""Small JSONL projection/filter helper for campaign ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_FIELDS = (
    "row",
    "variant",
    "kl_vs_fp8",
    "kl_mean",
    "top1_agree",
    "js",
    "total_GB",
    "gguf_gb",
    "size_gb",
    "bpw",
    "expert_gb",
    "n_windows",
    "manifest_md5",
    "corpus_md5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", nargs="+", type=Path)
    parser.add_argument("--contains", action="append", default=[])
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fields = [field for field in args.fields.split(",") if field]
    needles = [item.lower() for item in args.contains]
    for path in args.ledgers:
        for line_number, line in enumerate(path.expanduser().read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            identity = " ".join(str(row.get(key, "")) for key in ("row", "variant", "tag"))
            if needles and not all(needle in identity.lower() for needle in needles):
                continue
            print(json.dumps({key: row[key] for key in fields if key in row}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
