#!/usr/bin/env python3
"""Run the four P936 fail-closed acceptance probes without deleting payloads."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2]
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))

from authority.authority_guard import (  # noqa: E402
    AuthorityStore,
    GuardViolation,
    _atomic_json,
    _load_protected_index,
    assert_reclaim_allowed,
    assert_seal_dependencies,
    resolve_plan_codebook,
    sha256_file,
)


def expect_refusal(name, operation, contains):
    try:
        operation()
    except GuardViolation as exc:
        message = str(exc)
        if contains not in message:
            raise GuardViolation(f"{name} refused for the wrong reason: {message}") from exc
        return {"name": name, "status": "PASS_EXPECTED_REFUSAL", "message": message}
    raise GuardViolation(f"{name} unexpectedly succeeded")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--protected-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True)
    arguments = parser.parse_args(argv)
    store = AuthorityStore(arguments.store)
    protected = _load_protected_index(Path(arguments.protected_index).resolve())
    payloads = sorted(store.payload_root.glob("*.bin"))
    protected_payloads = [path for path in payloads if path.stem in protected]
    if not protected_payloads:
        raise GuardViolation("no authority payload intersects the protected SHA index")
    protected_path = protected_payloads[0]
    expected = protected_path.stem
    alternate_path = next((path for path in payloads if path.stem != expected), None)
    if alternate_path is None:
        raise GuardViolation("negative substitution probe requires two payloads")
    alternate = alternate_path.stem
    before_sha = sha256_file(protected_path)
    tests = []
    tests.append(expect_refusal(
        "delete_referenced_codebook",
        lambda: assert_reclaim_allowed([protected_path], arguments.protected_index, None),
        "protected SHA",
    ))
    with tempfile.TemporaryDirectory(prefix="p936-negative-") as td:
        root = Path(td)
        plan = root / "random-missing-mission" / "PLAN.json"
        plan.parent.mkdir(parents=True)
        plan.write_text("{}\n")
        row = {
            "path": str(root / "random-missing-mission" / "deleted-codebook.bin"),
            "codebook": {"sha256": expected, "bytes": protected_path.stat().st_size},
        }
        resolved = resolve_plan_codebook(store, plan, row)
        if resolved != protected_path:
            raise GuardViolation("SHA build resolved the wrong authority object")
        tests.append({
            "name": "missing_random_mission_path_sha_build",
            "status": "PASS",
            "missing_path": row["path"],
            "resolved_path": str(resolved),
            "resolved_sha256": sha256_file(resolved),
        })
        tests.append(expect_refusal(
            "substitute_without_measured_waiver",
            lambda: resolve_plan_codebook(store, plan, row, requested_sha256=alternate),
            "requires SUBSTITUTION_WAIVER.json",
        ))
    dependency = {
        "sha256": expected,
        "bytes": protected_path.stat().st_size,
        "role": "negative-one-copy-codebook",
    }
    tests.append(expect_refusal(
        "seal_with_one_copy",
        lambda: assert_seal_dependencies(
            [dependency], {arguments.host: store.root}, min_copies=2),
        "requires 2 independent copies",
    ))
    if sha256_file(protected_path) != before_sha:
        raise GuardViolation("negative probes mutated the protected payload")
    receipt = {
        "schema": "p936-host-grounded-negative-tests-v1",
        "status": "PASS",
        "host": arguments.host,
        "store_root": str(store.root),
        "protected_index": str(Path(arguments.protected_index).resolve()),
        "protected_test_sha256": expected,
        "alternate_test_sha256": alternate,
        "tests": tests,
        "test_count": len(tests),
        "payload_readback_sha256": sha256_file(protected_path),
        "completed_unix": time.time(),
    }
    _atomic_json(Path(arguments.output).resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "host": receipt["host"],
        "test_count": receipt["test_count"],
        "protected_test_sha256": receipt["protected_test_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GuardViolation as exc:
        print("FAIL_CLOSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
