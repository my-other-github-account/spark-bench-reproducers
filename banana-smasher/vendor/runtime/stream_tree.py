#!/usr/bin/env python3
"""Stream a verified deterministic prefix of a remote package tree to stdout."""
from __future__ import annotations

import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
count = int(sys.argv[2])
expected_files = int(sys.argv[3])
expected_total = int(sys.argv[4])
files = sorted(path for path in root.rglob("*") if path.is_file())
total = sum(path.stat().st_size for path in files)
if len(files) != expected_files or total != expected_total:
    raise SystemExit(f"inventory drift files={len(files)}/{expected_files} bytes={total}/{expected_total}")
if not 0 <= count <= total:
    raise SystemExit(f"invalid count {count}/{total}")
remaining = count
written = 0
for path in files:
    if remaining == 0:
        break
    with path.open("rb", buffering=0) as handle:
        while remaining:
            chunk = handle.read(min(8 << 20, remaining))
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            written += len(chunk)
            remaining -= len(chunk)
if remaining:
    raise SystemExit(f"short tree stream {written}/{count}")
sys.stdout.buffer.flush()
print(json.dumps({"root": str(root), "files": len(files), "tree_bytes": total,
                  "requested_bytes": count, "written_bytes": written}, sort_keys=True),
      file=sys.stderr, flush=True)
