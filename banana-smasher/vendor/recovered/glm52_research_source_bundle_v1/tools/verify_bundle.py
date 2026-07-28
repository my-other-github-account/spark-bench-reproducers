#!/usr/bin/env python3
"""Verify an extracted GLM-5.2 research source bundle using only stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat
import sys

FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".gguf", ".onnx", ".npy", ".npz",
    ".ckpt", ".engine", ".cubin", ".fatbin", ".so", ".dylib", ".dll",
}
META_FILES = {"BUNDLE_MANIFEST.json", "SHA256SUMS.txt"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors: list[str], rule: str, detail: str) -> None:
    errors.append(f"{rule}: {detail}")


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
    for required in (manifest_path, sums_path, source_manifest_path):
        if not required.is_file():
            fail(errors, "REQUIRED_FILE_MISSING", required.name)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2, sort_keys=True))
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "errors": [f"JSON_PARSE: {type(exc).__name__}"]}, sort_keys=True))
        return 1

    if manifest.get("schema") != "glm52-bundle-manifest-v1":
        fail(errors, "BAD_BUNDLE_SCHEMA", str(manifest.get("schema")))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        fail(errors, "BAD_MANIFEST_ENTRIES", "entries must be a list")
        entries = []

    expected_payload: set[str] = set()
    total_bytes = 0
    for entry in entries:
        rel = entry.get("path")
        if not isinstance(rel, str) or rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts:
            fail(errors, "UNSAFE_MANIFEST_PATH", repr(rel))
            continue
        if rel in expected_payload:
            fail(errors, "DUPLICATE_MANIFEST_PATH", rel)
            continue
        expected_payload.add(rel)
        path = root / rel
        if not path.is_file() or path.is_symlink():
            fail(errors, "DECLARED_FILE_MISSING", rel)
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        total_bytes += actual_size
        if actual_size != entry.get("bytes"):
            fail(errors, "SIZE_MISMATCH", rel)
        if actual_sha != entry.get("sha256"):
            fail(errors, "SHA256_MISMATCH", rel)
        expected_mode = entry.get("mode")
        actual_mode = stat.S_IMODE(path.stat().st_mode)
        if expected_mode is not None and actual_mode != int(expected_mode, 8):
            fail(errors, "MODE_MISMATCH", f"{rel}: {actual_mode:o}")

    actual_files = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    expected_files = expected_payload | META_FILES
    for rel in sorted(actual_files - expected_files):
        fail(errors, "UNDECLARED_FILE", rel)
    for rel in sorted(expected_files - actual_files):
        fail(errors, "EXPECTED_FILE_MISSING", rel)

    if manifest.get("file_count") != len(entries):
        fail(errors, "FILE_COUNT_MISMATCH", f"manifest={manifest.get('file_count')} actual={len(entries)}")
    if manifest.get("total_bytes") != total_bytes:
        fail(errors, "TOTAL_BYTES_MISMATCH", f"manifest={manifest.get('total_bytes')} actual={total_bytes}")

    sum_rows: dict[str, str] = {}
    try:
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            digest, rel = line.split("  ", 1)
            if rel in sum_rows:
                fail(errors, "DUPLICATE_SHA256SUM_PATH", rel)
            sum_rows[rel] = digest
    except Exception as exc:
        fail(errors, "BAD_SHA256SUMS", type(exc).__name__)
    sums_expected = expected_payload | {"BUNDLE_MANIFEST.json"}
    if set(sum_rows) != sums_expected:
        fail(errors, "SHA256SUM_SET_MISMATCH", f"expected={len(sums_expected)} actual={len(sum_rows)}")
    for rel, expected_sha in sorted(sum_rows.items()):
        path = root / rel
        if path.is_file() and sha256(path) != expected_sha:
            fail(errors, "SHA256SUM_MISMATCH", rel)

    source_entries = source_manifest.get("entries", [])
    if source_manifest.get("source_entry_count") != len(source_entries):
        fail(errors, "SOURCE_ENTRY_COUNT_MISMATCH", str(len(source_entries)))
    source_destinations: set[str] = set()
    for entry in source_entries:
        rel = entry.get("destination")
        if not isinstance(rel, str):
            fail(errors, "BAD_SOURCE_DESTINATION", repr(rel))
            continue
        if rel in source_destinations:
            fail(errors, "DUPLICATE_SOURCE_DESTINATION", rel)
        source_destinations.add(rel)
        path = root / rel
        if not path.is_file():
            fail(errors, "RECOVERED_SOURCE_MISSING", rel)
            continue
        if sha256(path) != entry.get("shipped_sha256"):
            fail(errors, "RECOVERED_SOURCE_SHA_MISMATCH", rel)
        if path.stat().st_size != entry.get("shipped_bytes"):
            fail(errors, "RECOVERED_SOURCE_SIZE_MISMATCH", rel)
        if not isinstance(entry.get("source_sha256"), str) or len(entry["source_sha256"]) != 64:
            fail(errors, "BAD_SOURCE_SHA_PIN", rel)

    result = {
        "schema": "glm52-bundle-verification-v1",
        "status": "PASS" if not errors else "FAIL",
        "bundle_id": manifest.get("bundle_id"),
        "verified_files": len(entries),
        "verified_bytes": total_bytes,
        "recovered_source_entries": len(source_entries),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
