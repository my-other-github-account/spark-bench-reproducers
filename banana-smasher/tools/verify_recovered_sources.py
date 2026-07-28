#!/usr/bin/env python3
"""Verify recovered research source closure and adoption boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "vendor" / "recovered" / "glm52_research_source_bundle_v1"
PROMOTABLE = {"P234", "P486", "P530", "P951", "P959", "P963", "P968"}
HOLD_ONLY = {"P526", "P948", "P950"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    manifest_path = BUNDLE / "RECOVERY_MANIFEST.json"
    if not manifest_path.is_file():
        return {"schema": "banana-smasher-recovered-verification-v1", "status": "FAIL", "failures": [{"reason": "missing recovery manifest"}]}
    manifest = json.loads(manifest_path.read_text())
    adoption = json.loads((BUNDLE / "ADOPTION_MAP.json").read_text())
    source_manifest = json.loads((BUNDLE / "SOURCE_MANIFEST.json").read_text())
    rows = manifest["files"]
    for row in rows:
        path = BUNDLE / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        if path.stat().st_size != row["bytes"]:
            failures.append({"path": row["path"], "reason": "size mismatch"})
        actual = sha256(path)
        if actual != row["sha256"]:
            failures.append({"path": row["path"], "reason": "sha256 mismatch", "actual": actual})
    if manifest["source_archive_sha256"] != "572b4ec1d04d512b4cab30ba47fa033cca63b2ab6ff8bcc2b66642fff3e882c4":
        failures.append({"reason": "source archive pin mismatch"})
    if manifest["recovered_source_entries"] != 89 or len(source_manifest["entries"]) != 89:
        failures.append({"reason": "recovered source entry count mismatch"})

    families = adoption["families"]
    if set(families) != PROMOTABLE | HOLD_ONLY:
        failures.append({"reason": "family set mismatch", "families": sorted(families)})
    for family in sorted(PROMOTABLE):
        record = families[family]
        if not record["disposition"].startswith(("PROMOTE", "REFERENCE")):
            failures.append({"family": family, "reason": "promotable family disposition drift"})
        primary = record.get("primary_files", [])
        if not any("/code/" in path for path in primary):
            failures.append({"family": family, "reason": "no working code"})
        if not any("/receipts/" in path for path in primary):
            failures.append({"family": family, "reason": "no receipt gate"})
        for relative in primary:
            if not (BUNDLE / relative).is_file():
                failures.append({"family": family, "path": relative, "reason": "primary file missing"})
    for family in sorted(HOLD_ONLY):
        disposition = families[family]["disposition"]
        if not (disposition.startswith("HOLD") or disposition.startswith("NEGATIVE")):
            failures.append({"family": family, "reason": "hold boundary drift"})

    p963 = families["P963"]["evidence"]
    if p963["status"] != "PASS_EXACT_EQUAL_ACCELERATION_GE_2X" or p963["speedup"] < 2.0:
        failures.append({"family": "P963", "reason": "exact acceleration gate drift"})
    if p963["maximum_absolute_per_position_delta"] != 0.0:
        failures.append({"family": "P963", "reason": "exactness gate drift"})

    return {
        "schema": "banana-smasher-recovered-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "source_archive_sha256": manifest["source_archive_sha256"],
        "verified_files": len(rows),
        "recovered_source_entries": manifest["recovered_source_entries"],
        "promotable_families": sorted(PROMOTABLE),
        "hold_only_families": sorted(HOLD_ONLY),
        "failures": failures,
    }


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
