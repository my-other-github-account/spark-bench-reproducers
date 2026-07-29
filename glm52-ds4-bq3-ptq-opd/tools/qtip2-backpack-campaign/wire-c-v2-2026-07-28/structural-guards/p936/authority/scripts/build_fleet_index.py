#!/usr/bin/env python3
"""Discover sealed fleet manifests and build the protected SHA reverse index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import (  # noqa: E402
    GuardViolation,
    _is_sealed_document,
    _sha_references,
    build_protected_index,
)

_SKIP_DIRS = {"cache", "stage_cache", "runtime", "__pycache__", "node_modules"}


def documents(path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def discover_sealed_manifests(mission_roots, max_bytes=0):
    """Return every valid sealed JSON document carrying at least one SHA.

    Discovery deliberately has no filename allowlist: K1/layer archives and new
    manifest classes are protected even when their names predate this guard.
    Invalid JSON is reported as a parse skip and cannot qualify as a seal.
    """
    selected = []
    parse_skips = []
    scanned = 0
    for raw_root in mission_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir() or root.is_symlink():
            raise GuardViolation("mission root must be a real directory: %s" % root)
        for base, directories, filenames in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if name not in _SKIP_DIRS]
            for filename in filenames:
                if not filename.endswith((".json", ".jsonl")):
                    continue
                path = Path(base) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                if max_bytes and path.stat().st_size > max_bytes:
                    raise GuardViolation(
                        "manifest candidate exceeds configured byte ceiling: %s" % path
                    )
                scanned += 1
                try:
                    docs = documents(path)
                except (OSError, UnicodeError, ValueError):
                    parse_skips.append(str(path.resolve()))
                    continue
                if any(
                    _is_sealed_document(document)
                    and any(True for _ in _sha_references(document))
                    for document in docs
                ):
                    selected.append(path.resolve())
    selected = sorted(set(selected))
    return selected, {
        "scanned_candidates": scanned,
        "selected_manifests": len(selected),
        "parse_skips": sorted(set(parse_skips)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="optional per-document ceiling; zero (default) scans without a size limit",
    )
    arguments = parser.parse_args(argv)
    selected, stats = discover_sealed_manifests(
        arguments.mission_root, max_bytes=arguments.max_bytes)
    if not selected:
        raise GuardViolation("no sealed manifests with SHA references discovered")
    index = build_protected_index(selected, arguments.output)
    path_set_payload = json.dumps([str(path) for path in selected], separators=(",", ":")).encode("utf-8")
    summary = {
        "status": "PASS",
        "scanned_candidates": stats["scanned_candidates"],
        "selected_manifests": stats["selected_manifests"],
        "parse_skips": len(stats["parse_skips"]),
        "selected_path_set_sha256": hashlib.sha256(path_set_payload).hexdigest(),
        "protected_sha_count": index["protected_sha_count"],
        "output": str(Path(arguments.output).expanduser().resolve()),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
