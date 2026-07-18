#!/usr/bin/env python3
"""Regenerate per-topic and campaign SHA-256/size manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [p for p in ROOT.iterdir() if p.is_dir() and p.name != "tools"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(base: Path) -> list[dict]:
    rows = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "MANIFEST.sha256"}:
            continue
        rows.append(
            {
                "path": str(path.relative_to(base)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    return rows


def write_manifest(base: Path) -> None:
    rows = build_manifest(base)
    (base / "MANIFEST.json").write_text(json.dumps(rows, indent=2) + "\n")
    (base / "MANIFEST.sha256").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    )


def main() -> None:
    for topic in TOPICS:
        write_manifest(topic)
    write_manifest(ROOT)
    print(f"updated {len(TOPICS)} topic manifests plus campaign manifest")


if __name__ == "__main__":
    main()
