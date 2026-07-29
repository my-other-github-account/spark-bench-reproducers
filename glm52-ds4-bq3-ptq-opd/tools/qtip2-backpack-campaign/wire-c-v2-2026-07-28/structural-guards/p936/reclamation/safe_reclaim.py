#!/usr/bin/env python3
"""Fail-closed reclaimer with protected-SHA and archive-first enforcement."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import (  # noqa: E402
    GuardViolation,
    _atomic_json,
    assert_reclaim_allowed,
)

DEFAULT_PROTECTED_INDEX = "~/authority_store/protected_sha_index.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-index", default=os.environ.get(
        "QTIP2_PROTECTED_SHA_INDEX", os.path.expanduser(DEFAULT_PROTECTED_INDEX)))
    parser.add_argument("--archive-first", default=os.environ.get("QTIP2_ARCHIVE_FIRST_RECEIPT"))
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--execute", action="store_true", help="delete after all guards pass")
    parser.add_argument("paths", nargs="+")
    arguments = parser.parse_args(argv)
    targets = [Path(path).expanduser().resolve() for path in arguments.paths]
    forbidden = {Path("/"), Path.home().resolve()}
    for target in targets:
        if target in forbidden or target.is_symlink() or os.path.ismount(target):
            raise GuardViolation("forbidden reclaim target: %s" % target)
        if not target.exists():
            raise GuardViolation("reclaim target is absent: %s" % target)
    assert_reclaim_allowed(
        [str(target) for target in targets],
        arguments.protected_index,
        arguments.archive_first,
    )
    receipt = {
        "schema": "p936-safe-reclaim-v1",
        "status": "PREDELETE_PASS" if arguments.execute else "PASS_DRY_RUN",
        "protected_index": str(Path(arguments.protected_index).expanduser().resolve()),
        "archive_first_receipt": (
            str(Path(arguments.archive_first).expanduser().resolve())
            if arguments.archive_first else None),
        "targets": [str(target) for target in targets],
        "checked_unix": time.time(),
    }
    _atomic_json(Path(arguments.receipt).expanduser().resolve(), receipt)
    if not arguments.execute:
        print(json.dumps(receipt, sort_keys=True))
        return 0
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        if os.path.lexists(target):
            raise GuardViolation("reclaim deletion incomplete: %s" % target)
    receipt["status"] = "PASS"
    receipt["completed_unix"] = time.time()
    _atomic_json(Path(arguments.receipt).expanduser().resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
