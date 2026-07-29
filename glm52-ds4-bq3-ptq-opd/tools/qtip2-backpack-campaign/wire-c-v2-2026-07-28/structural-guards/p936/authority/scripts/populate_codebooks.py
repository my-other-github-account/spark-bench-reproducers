#!/usr/bin/env python3
"""Discover codebook files under explicit roots and ingest them by SHA."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import AuthorityStore, GuardViolation, _atomic_json  # noqa: E402

_SUFFIXES = (".bin", ".npy", ".pt")


def discover(roots):
    seen = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise GuardViolation("scan root must be a real directory: %s" % root)
        for base, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name not in {
                "cache", "stage_cache", "runtime", "__pycache__", "node_modules"}]
            for filename in filenames:
                path = Path(base) / filename
                lower = str(path).lower()
                if not lower.endswith(_SUFFIXES) or "codebook" not in lower:
                    continue
                if path.is_symlink() or not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="authority store root")
    parser.add_argument("--scan-root", action="append", required=True)
    parser.add_argument("--receipt", required=True)
    arguments = parser.parse_args(argv)
    store = AuthorityStore(arguments.root)
    rows = []
    for path in discover(arguments.scan_root):
        digest = store.ingest(path, metadata={"role": "codebook", "discovered_by": "populate_codebooks.py"})
        rows.append({"source_path": str(path), "sha256": digest, "bytes": path.stat().st_size})
    unique = sorted({row["sha256"] for row in rows})
    receipt = {
        "schema": "p936-authority-population-v1",
        "status": "PASS",
        "store_root": str(store.root),
        "scan_roots": [str(Path(root).expanduser().resolve()) for root in arguments.scan_root],
        "source_files": len(rows),
        "unique_sha256": len(unique),
        "unique_sha256_set": unique,
        "rows": rows,
        "completed_unix": time.time(),
    }
    _atomic_json(Path(arguments.receipt).expanduser().resolve(), receipt)
    print(json.dumps({
        "status": "PASS", "source_files": len(rows), "unique_sha256": len(unique),
        "receipt": str(Path(arguments.receipt).expanduser().resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
