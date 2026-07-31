#!/usr/bin/env python3
"""Fail-closed validator for BANANA_SMASHER export packs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_NAME = "banana_smasher-pack"
SCHEMA_VERSION = 1
EXPECTED_SERVING = {
    "layers": 43,
    "experts": 256,
    "topk": 6,
    "tier_names": ["qtip", "truevq_d4", "truevq_d8", "native_mxfp4"],
    "prefill_mode": "dense_all",
    "dense_threshold": 64,
    "layer_stride": 43,
}


class PackValidationError(ValueError):
    """The mounted export pack does not satisfy the container contract."""


def canonical_inventory_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["path"]):
        digest.update(
            f"{row['path']}\0{int(row['bytes'])}\0{row['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pack(
    pack_root: str | Path,
    *,
    expected_schema_version: int = SCHEMA_VERSION,
    workers: int = 4,
) -> dict[str, Any]:
    root = Path(pack_root).resolve()
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PackValidationError(f"pack MANIFEST.json is missing or unsafe under {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError(f"pack MANIFEST.json is unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackValidationError("pack MANIFEST.json must contain a JSON object")
    if manifest.get("schema") != SCHEMA_NAME:
        raise PackValidationError(
            f"pack schema name mismatch: container={SCHEMA_NAME} pack={manifest.get('schema')!r}"
        )
    actual_version = manifest.get("schema_version")
    if actual_version != expected_schema_version:
        raise PackValidationError(
            "pack schema version mismatch: "
            f"container={expected_schema_version} pack={actual_version!r}"
        )

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackValidationError("manifest files must be a non-empty list")

    resolved: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PackValidationError("each manifest file row must be an object")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise PackValidationError(f"invalid or duplicate manifest path: {relative!r}")
        if not isinstance(row.get("bytes"), int) or int(row["bytes"]) < 0:
            raise PackValidationError(f"invalid payload byte count: {relative!r}")
        digest = row.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise PackValidationError(f"invalid payload sha256: {relative!r}")
        seen.add(relative)
        target = (root / relative).resolve()
        if root not in target.parents:
            raise PackValidationError(f"manifest path escapes pack root: {relative!r}")
        unresolved = root / relative
        if unresolved.is_symlink() or not target.is_file():
            raise PackValidationError(f"manifest payload missing: {relative}")
        if target.stat().st_size != row.get("bytes"):
            raise PackValidationError(f"payload byte count mismatch: {relative}")
        resolved.append((row, target))

    declared = set(seen)
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    extras = sorted(present - declared)
    if extras:
        raise PackValidationError(f"unlisted payload files: {extras}")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        actual_hashes = list(pool.map(lambda item: _file_sha256(item[1]), resolved))
    for (row, _), actual_hash in zip(resolved, actual_hashes):
        if actual_hash != row.get("sha256"):
            raise PackValidationError(
                f"payload sha256 mismatch: {row['path']} expected={row.get('sha256')} actual={actual_hash}"
            )

    envelope = manifest.get("resident_envelope", {})
    plane_rows = [row for row in rows if row.get("role") == "plane"]
    if envelope.get("root") != "planes":
        raise PackValidationError("resident_envelope.root must be 'planes'")
    if envelope.get("files") != len(plane_rows):
        raise PackValidationError("resident envelope file count mismatch")
    if envelope.get("bytes") != sum(int(row["bytes"]) for row in plane_rows):
        raise PackValidationError("resident envelope byte count mismatch")
    inventory_sha = canonical_inventory_sha256(plane_rows)
    if envelope.get("inventory_sha256") != inventory_sha:
        raise PackValidationError("resident envelope inventory sha256 mismatch")
    source_inventory_sha = envelope.get("sealed_source_inventory_sha256")
    if (
        not isinstance(source_inventory_sha, str)
        or len(source_inventory_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_inventory_sha)
    ):
        raise PackValidationError(
            "resident envelope sealed source inventory sha256 must be lowercase hex"
        )

    serving = manifest.get("serving", {})
    if manifest.get("validation_scope") != "systems-serving-only":
        raise PackValidationError("validation_scope must be 'systems-serving-only'")
    if manifest.get("quality_validated") is not False:
        raise PackValidationError("quality_validated must be false for this systems-only pack")
    if not isinstance(manifest.get("model_id"), str) or not manifest["model_id"].strip():
        raise PackValidationError("model_id must be a non-empty string")
    for field, expected in EXPECTED_SERVING.items():
        if serving.get(field) != expected:
            raise PackValidationError(
                f"serving.{field} mismatch: expected={expected!r} actual={serving.get(field)!r}"
            )
    roles = {row.get("path"): row.get("role") for row in rows}
    if roles.get(serving.get("artifact")) != "mixed_tier_overlay":
        raise PackValidationError("serving.artifact must name a mixed_tier_overlay payload")
    if roles.get(serving.get("tokenizer")) != "tokenizer":
        raise PackValidationError("serving.tokenizer must name a tokenizer payload")
    try:
        tokenizer = json.loads((root / str(serving["tokenizer"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError(f"tokenizer payload is not valid JSON: {exc}") from exc
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("version"), str):
        raise PackValidationError("tokenizer payload must contain a string version")

    return {
        "status": "PASS",
        "schema": "banana_smasher-pack-validation-v1",
        "schema_version": actual_version,
        "model_id": manifest.get("model_id"),
        "manifest_sha256": _file_sha256(manifest_path),
        "payload_files": len(rows),
        "payload_bytes": sum(int(row["bytes"]) for row in rows),
        "resident_envelope": envelope,
        "serving": serving,
    }
