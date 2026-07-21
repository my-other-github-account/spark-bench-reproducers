#!/usr/bin/env python3
"""Verify the normalized public receipt bundle and claim-source coverage."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
CLAIM_FILES = (
    ROOT / "RESULTS.md",
    ROOT / "CENSORING.md",
    ROOT / "TRANSFER.md",
    RECEIPTS / "SEALED_RESULTS.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((RECEIPTS / "RECEIPTS_MANIFEST.json").read_text())
    if manifest.get("schema") != "ptq-opd-public-receipt-manifest-v1":
        raise SystemExit("RECEIPT_VERIFY_FAIL bad manifest schema")

    failures: list[str] = []
    resolved: set[str] = set()
    expected_sha_lines: list[str] = []
    for entry in manifest.get("receipts", []):
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe receipt path: {relative}")
            continue
        path = RECEIPTS / relative
        if not path.is_file():
            failures.append(f"missing receipt: {relative}")
            continue
        observed_public = sha256(path)
        if observed_public != entry["public_sha256"]:
            failures.append(f"public hash mismatch: {relative}")
        wrapper = json.loads(path.read_text())
        if wrapper.get("schema") != "ptq-opd-public-receipt-v1":
            failures.append(f"bad wrapper schema: {relative}")
        if wrapper.get("source_receipt_sha256") != entry["source_sha256"]:
            failures.append(f"source hash mismatch: {relative}")
        transform = wrapper.get("privacy_transform", {})
        if transform.get("numeric_values_preserved") is not True:
            failures.append(f"numeric preservation not asserted: {relative}")
        resolved.update((entry["source_sha256"], entry["public_sha256"]))
        expected_sha_lines.append(f"{entry['public_sha256']}  {entry['file']}")

    observed_sha_lines = [
        line for line in (RECEIPTS / "RECEIPTS_MANIFEST.sha256").read_text().splitlines()
        if line.strip()
    ]
    if observed_sha_lines != expected_sha_lines:
        failures.append("RECEIPTS_MANIFEST.sha256 does not match JSON manifest order/content")

    referenced: set[str] = set()
    for path in CLAIM_FILES:
        referenced.update(HEX64.findall(path.read_text()))
    unresolved = sorted(referenced - resolved)
    if unresolved:
        failures.append("unresolved claim source/public hashes: " + ", ".join(unresolved))

    if failures:
        print("RECEIPT_VERIFY_FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print(
        "RECEIPT_VERIFY_PASS "
        f"receipts={len(manifest['receipts'])} referenced={len(referenced)} resolved={len(referenced)}"
    )


if __name__ == "__main__":
    main()
