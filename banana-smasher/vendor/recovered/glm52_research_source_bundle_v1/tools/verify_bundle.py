#!/usr/bin/env python3
"""Verify a recovered bundle and its optional privacy-redaction overlay."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat

FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".npy", ".npz",
    ".ckpt", ".engine", ".cubin", ".fatbin", ".so", ".dylib", ".dll",
}
BASE_META_FILES = {"BUNDLE_MANIFEST.json", "SHA256SUMS.txt"}
RECOVERY_META_FILE = "RECOVERY_MANIFEST.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(errors: list[str], rule: str, detail: str) -> None:
    errors.append(f"{rule}: {detail}")


def safe_relative(value: object) -> bool:
    return (
        isinstance(value, str)
        and not value.startswith("/")
        and ".." not in pathlib.PurePosixPath(value).parts
    )


def load_json(path: pathlib.Path, errors: list[str], label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(errors, f"{label}_JSON_PARSE", type(exc).__name__)
        return {}
    if not isinstance(value, dict):
        fail(errors, f"{label}_NOT_OBJECT", type(value).__name__)
        return {}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(json.dumps({"status": "FAIL", "errors": ["ROOT_NOT_DIRECTORY"]}, sort_keys=True))
        return 1

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            fail(errors, "SYMLINK_FORBIDDEN", rel)
        elif not (path.is_file() or path.is_dir()):
            fail(errors, "NON_REGULAR_ENTRY", rel)
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(errors, "FORBIDDEN_ARTIFACT_EXTENSION", rel)

    manifest_path = root / "BUNDLE_MANIFEST.json"
    sums_path = root / "SHA256SUMS.txt"
    source_manifest_path = root / "SOURCE_MANIFEST.json"
    recovery_path = root / RECOVERY_META_FILE
    for required in (manifest_path, sums_path, source_manifest_path):
        if not required.is_file():
            fail(errors, "REQUIRED_FILE_MISSING", required.name)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2, sort_keys=True))
        return 1

    manifest = load_json(manifest_path, errors, "BUNDLE_MANIFEST")
    source_manifest = load_json(source_manifest_path, errors, "SOURCE_MANIFEST")
    recovery = load_json(recovery_path, errors, "RECOVERY_MANIFEST") if recovery_path.is_file() else None

    if manifest.get("schema") != "glm52-bundle-manifest-v1":
        fail(errors, "BAD_BUNDLE_SCHEMA", str(manifest.get("schema")))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        fail(errors, "BAD_MANIFEST_ENTRIES", "entries must be a list")
        entries = []

    original_rows: dict[str, dict] = {}
    original_total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            fail(errors, "BAD_MANIFEST_ENTRY", repr(entry))
            continue
        rel = entry.get("path")
        if not safe_relative(rel):
            fail(errors, "UNSAFE_MANIFEST_PATH", repr(rel))
            continue
        assert isinstance(rel, str)
        if rel in original_rows:
            fail(errors, "DUPLICATE_MANIFEST_PATH", rel)
            continue
        original_rows[rel] = entry
        try:
            original_total_bytes += int(entry["bytes"])
        except Exception:
            fail(errors, "BAD_MANIFEST_SIZE", rel)

    recovery_rows: dict[str, dict] = {}
    if recovery is not None:
        if recovery.get("schema") != "banana-smasher-recovered-source-manifest-v1":
            fail(errors, "BAD_RECOVERY_SCHEMA", str(recovery.get("schema")))
        source_archive_sha = recovery.get("source_archive_sha256")
        if not isinstance(source_archive_sha, str) or len(source_archive_sha) != 64:
            fail(errors, "BAD_SOURCE_ARCHIVE_SHA256", repr(source_archive_sha))
        rows = recovery.get("files")
        if not isinstance(rows, list):
            fail(errors, "BAD_RECOVERY_FILES", "files must be a list")
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                fail(errors, "BAD_RECOVERY_ROW", repr(row))
                continue
            rel = row.get("path")
            if not safe_relative(rel):
                fail(errors, "UNSAFE_RECOVERY_PATH", repr(rel))
                continue
            assert isinstance(rel, str)
            if rel in recovery_rows:
                fail(errors, "DUPLICATE_RECOVERY_PATH", rel)
                continue
            recovery_rows[rel] = row

        required_recovery_rows = set(original_rows) | BASE_META_FILES
        if set(recovery_rows) != required_recovery_rows:
            fail(
                errors,
                "RECOVERY_FILE_SET_MISMATCH",
                f"expected={len(required_recovery_rows)} actual={len(recovery_rows)}",
            )

        for rel, row in sorted(recovery_rows.items()):
            path = root / rel
            if not path.is_file() or path.is_symlink():
                fail(errors, "RECOVERY_FILE_MISSING", rel)
                continue
            actual_size = path.stat().st_size
            actual_sha = sha256(path)
            if actual_size != row.get("bytes"):
                fail(errors, "RECOVERY_SIZE_MISMATCH", rel)
            if actual_sha != row.get("sha256"):
                fail(errors, "RECOVERY_SHA256_MISMATCH", rel)
            if rel in original_rows:
                original_sha = original_rows[rel].get("sha256")
            else:
                original_sha = actual_sha
            if row.get("recovered_bundle_sha256") != original_sha:
                fail(errors, "RECOVERED_BUNDLE_SHA_MISMATCH", rel)
            transformed = actual_sha != original_sha
            if row.get("transformed") is not transformed:
                fail(errors, "RECOVERY_TRANSFORMED_FLAG_MISMATCH", rel)
            if transformed and not isinstance(row.get("transformations"), dict):
                fail(errors, "RECOVERY_TRANSFORMATION_LEDGER_MISSING", rel)

    total_current_payload_bytes = 0
    for rel, entry in sorted(original_rows.items()):
        path = root / rel
        if not path.is_file() or path.is_symlink():
            fail(errors, "DECLARED_FILE_MISSING", rel)
            continue
        expected = recovery_rows.get(rel, entry)
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        total_current_payload_bytes += actual_size
        if actual_size != expected.get("bytes"):
            fail(errors, "SIZE_MISMATCH", rel)
        if actual_sha != expected.get("sha256"):
            fail(errors, "SHA256_MISMATCH", rel)
        expected_mode = entry.get("mode")
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        if expected_mode is not None and actual_mode != int(expected_mode, 8):
            fail(errors, "MODE_MISMATCH", f"{rel}: {actual_mode:o}")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = set(original_rows) | BASE_META_FILES
    if recovery is not None:
        expected_files.add(RECOVERY_META_FILE)
    for rel in sorted(actual_files - expected_files):
        fail(errors, "UNDECLARED_FILE", rel)
    for rel in sorted(expected_files - actual_files):
        fail(errors, "EXPECTED_FILE_MISSING", rel)

    if manifest.get("file_count") != len(entries):
        fail(errors, "FILE_COUNT_MISMATCH", f"manifest={manifest.get('file_count')} actual={len(entries)}")
    if manifest.get("total_bytes") != original_total_bytes:
        fail(errors, "TOTAL_BYTES_MISMATCH", f"manifest={manifest.get('total_bytes')} actual={original_total_bytes}")

    sum_rows: dict[str, str] = {}
    try:
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, rel = line.split("  ", 1)
            if rel in sum_rows:
                fail(errors, "DUPLICATE_SHA256SUM_PATH", rel)
            sum_rows[rel] = digest
    except Exception as exc:
        fail(errors, "BAD_SHA256SUMS", type(exc).__name__)
    sums_expected = set(original_rows) | {"BUNDLE_MANIFEST.json"}
    if set(sum_rows) != sums_expected:
        fail(errors, "SHA256SUM_SET_MISMATCH", f"expected={len(sums_expected)} actual={len(sum_rows)}")
    for rel, declared_sha in sorted(sum_rows.items()):
        expected_sha = sha256(manifest_path) if rel == "BUNDLE_MANIFEST.json" else original_rows.get(rel, {}).get("sha256")
        if declared_sha != expected_sha:
            fail(errors, "SHA256SUM_ORIGINAL_CLOSURE_MISMATCH", rel)
        if recovery is None:
            path = root / rel
            if path.is_file() and sha256(path) != declared_sha:
                fail(errors, "SHA256SUM_MISMATCH", rel)

    source_entries = source_manifest.get("entries", [])
    if not isinstance(source_entries, list):
        fail(errors, "BAD_SOURCE_ENTRIES", "entries must be a list")
        source_entries = []
    if source_manifest.get("source_entry_count") != len(source_entries):
        fail(errors, "SOURCE_ENTRY_COUNT_MISMATCH", str(len(source_entries)))
    source_destinations: set[str] = set()
    for entry in source_entries:
        if not isinstance(entry, dict):
            fail(errors, "BAD_SOURCE_ENTRY", repr(entry))
            continue
        rel = entry.get("destination")
        if not safe_relative(rel):
            fail(errors, "BAD_SOURCE_DESTINATION", repr(rel))
            continue
        assert isinstance(rel, str)
        if rel in source_destinations:
            fail(errors, "DUPLICATE_SOURCE_DESTINATION", rel)
        source_destinations.add(rel)
        path = root / rel
        if not path.is_file():
            fail(errors, "RECOVERED_SOURCE_MISSING", rel)
            continue
        current = recovery_rows.get(rel)
        expected_sha = current.get("sha256") if current else entry.get("shipped_sha256")
        expected_size = current.get("bytes") if current else entry.get("shipped_bytes")
        if sha256(path) != expected_sha:
            fail(errors, "RECOVERED_SOURCE_SHA_MISMATCH", rel)
        if path.stat().st_size != expected_size:
            fail(errors, "RECOVERED_SOURCE_SIZE_MISMATCH", rel)
        if not isinstance(entry.get("source_sha256"), str) or len(entry["source_sha256"]) != 64:
            fail(errors, "BAD_SOURCE_SHA_PIN", rel)

    result = {
        "schema": "glm52-bundle-verification-v2",
        "status": "PASS" if not errors else "FAIL",
        "bundle_id": manifest.get("bundle_id"),
        "recovery_overlay_verified": recovery is not None,
        "verified_original_payload_files": len(entries),
        "verified_current_files": len(recovery_rows) if recovery is not None else len(entries),
        "verified_original_payload_bytes": original_total_bytes,
        "verified_current_payload_bytes": total_current_payload_bytes,
        "recovered_source_entries": len(source_entries),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
