#!/usr/bin/env python3
"""Offline Docker build-context closure check."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    dockerfile = (ROOT / "Dockerfile").read_text()
    failures = []
    copies = []
    for line_number, line in enumerate(dockerfile.splitlines(), 1):
        match = re.match(r"\s*COPY\s+([^\s]+)\s+", line)
        if not match:
            continue
        source = match.group(1)
        copies.append(source)
        if source.startswith(("/", "~")) or ".." + "/" in source:
            failures.append({"line": line_number, "source": source, "reason": "context escape"})
        elif not (ROOT / source).exists():
            failures.append({"line": line_number, "source": source, "reason": "missing"})
    required = {"vendor/runtime", "vendor/kernel", "configs", "locks"}
    missing = sorted(required - set(copies))
    if missing:
        failures.append({"missing_copy_roots": missing})
    result = {
        "schema": "banana-smasher-docker-context-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "copies": copies,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
