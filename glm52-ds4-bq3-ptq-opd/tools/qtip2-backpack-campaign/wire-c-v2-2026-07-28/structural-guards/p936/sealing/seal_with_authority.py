#!/usr/bin/env python3
"""Seal a manifest only after a two-host dependency copy census passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import (  # noqa: E402
    GuardViolation,
    _atomic_json,
    assert_seal_dependencies,
    sha256_file,
)
from authority.cli import _remote_probe_factory  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dependencies", required=True)
    parser.add_argument("--locations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-copies", type=int, default=2)
    arguments = parser.parse_args(argv)
    manifest_path = Path(arguments.manifest).expanduser().resolve()
    dependency_path = Path(arguments.dependencies).expanduser().resolve()
    location_path = Path(arguments.locations).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependency_document = json.loads(dependency_path.read_text(encoding="utf-8"))
    locations = json.loads(location_path.read_text(encoding="utf-8"))
    roots = {host: specification["root"] for host, specification in locations.items()}
    census = assert_seal_dependencies(
        dependency_document.get("dependencies", dependency_document),
        roots,
        min_copies=arguments.min_copies,
        probe=_remote_probe_factory(locations),
    )
    seal = {
        "schema": "p936-authority-gated-seal-v1",
        "status": "PASS",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_status": manifest.get("status") if isinstance(manifest, dict) else None,
        "dependencies_path": str(dependency_path),
        "dependencies_sha256": sha256_file(dependency_path),
        "locations_path": str(location_path),
        "locations_sha256": sha256_file(location_path),
        "dependency_census": census,
        "completed_unix": time.time(),
    }
    _atomic_json(Path(arguments.output).expanduser().resolve(), seal)
    print(json.dumps({
        "status": "PASS", "output": str(Path(arguments.output).expanduser().resolve()),
        "dependency_count": census["dependency_count"],
        "manifest_sha256": seal["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GuardViolation, OSError, ValueError) as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
