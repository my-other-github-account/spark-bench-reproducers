#!/usr/bin/env python3
"""Hash-seal AOT cubins plus Triton/FlashInfer warm caches."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path


def row(root: Path, path: Path, label: str) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return {"context": label, "path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    work = []
    roots = {}
    for label, raw in args.context:
        root = Path(raw).resolve()
        if not root.is_dir():
            raise SystemExit(f"missing cache context {label}: {root}")
        if label in roots:
            raise SystemExit(f"duplicate cache context label: {label}")
        roots[label] = root
        before = len(work)
        for path in root.rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"cache symlink is forbidden: {label}:{path.relative_to(root)} -> {os.readlink(path)}")
            if not path.is_file():
                continue
            work.append((root, path, label))
        if len(work) == before:
            raise SystemExit(f"empty cache context is forbidden: {label}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda args: row(*args), work))
    rows.sort(key=lambda item: (item["context"], item["path"]))
    payload = {
        "schema": "genesis-golden-runtime-cache-manifest-v1",
        "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
        "contexts": sorted(roots),
        "file_count": len(rows),
        "total_bytes": sum(item["bytes"] for item in rows),
        "files": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "file_count": len(rows), "total_bytes": payload["total_bytes"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
