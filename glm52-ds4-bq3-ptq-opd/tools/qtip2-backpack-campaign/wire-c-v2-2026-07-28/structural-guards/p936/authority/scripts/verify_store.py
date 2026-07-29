#!/usr/bin/env python3
"""Full-read and register every payload in an authority-store union."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import (  # noqa: E402
    AuthorityStore,
    GuardViolation,
    _atomic_json,
    sha256_file,
)


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--host", required=True)
    arguments = parser.parse_args(argv)
    store = AuthorityStore(arguments.root)
    rows = []
    for path in sorted(store.payload_root.glob("*.bin")):
        digest = path.stem
        if len(digest) != 64 or sha256_file(path) != digest:
            raise GuardViolation("authority payload filename/hash drift: %s" % path)
        registered = store.ingest(path, metadata={"role": "peer_union_readback", "host": arguments.host})
        if registered != digest:
            raise GuardViolation("authority registration drift: %s" % path)
        rows.append({"sha256": digest, "bytes": path.stat().st_size})
    digests = [row["sha256"] for row in rows]
    receipt = {
        "schema": "p936-authority-store-readback-v1",
        "status": "PASS",
        "host": arguments.host,
        "store_root": str(store.root),
        "payloads": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "sha256_set": digests,
        "sha256_set_sha256": canonical_sha(digests),
        "rows": rows,
        "completed_unix": time.time(),
    }
    _atomic_json(Path(arguments.receipt).expanduser().resolve(), receipt)
    print(json.dumps({key: receipt[key] for key in (
        "status", "host", "payloads", "bytes", "sha256_set_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
