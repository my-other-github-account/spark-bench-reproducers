#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed verifier for an iq3-vq-wire-pack-v1 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def require_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing file: {relative}")
    return path


def verify(root: Path, quick: bool) -> dict[str, Any]:
    manifest_path = require_file(root, "PACK_MANIFEST.json")
    require_file(root, "PACK_COMPLETE")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "iq3-vq-wire-pack-v1":
        raise ValueError(f"unsupported format: {manifest.get('format')!r}")

    digest_sidecar = root / "PACK_MANIFEST.json.sha256"
    manifest_sha256 = sha256_file(manifest_path)
    if digest_sidecar.is_file():
        expected = digest_sidecar.read_text().split()[0]
        if expected != manifest_sha256:
            raise ValueError("PACK_MANIFEST.json sidecar SHA256 mismatch")

    source_copy = require_file(root, str(manifest["source_manifest_copy"]))
    if sha256_file(source_copy) != manifest["source_manifest_sha256"]:
        raise ValueError("source manifest SHA256 mismatch")

    meta_count = 0
    payload_count = 0
    payload_bytes = 0
    for layer in manifest["layers"]:
        layer_id = int(layer["layer"])
        meta_name = f"layer_{layer_id:03d}.meta.json"
        meta_path = require_file(root, meta_name)
        if sha256_file(meta_path) != layer["meta_sha256"]:
            raise ValueError(f"{meta_name}: SHA256 mismatch")
        meta = json.loads(meta_path.read_text())
        if int(meta["layer"]) != layer_id:
            raise ValueError(f"{meta_name}: layer id mismatch")
        if len(meta["files"]) != int(layer["files"]):
            raise ValueError(f"{meta_name}: payload count mismatch")
        meta_count += 1
        for name, receipt in meta["files"].items():
            path = require_file(root, name)
            size = path.stat().st_size
            if size != int(receipt["bytes"]):
                raise ValueError(f"{name}: byte-size mismatch")
            if not quick and sha256_file(path) != receipt["sha256"]:
                raise ValueError(f"{name}: SHA256 mismatch")
            payload_count += 1
            payload_bytes += size

    expected_payload_count = manifest.get("payload_file_count", manifest.get("payload_files"))
    expected_payload_bytes = manifest.get("payload_total_bytes", manifest.get("payload_bytes"))
    if expected_payload_count is None:
        raise ValueError("manifest missing payload count (payload_file_count or payload_files)")
    if expected_payload_bytes is None:
        raise ValueError("manifest missing payload bytes (payload_total_bytes or payload_bytes)")
    if payload_count != int(expected_payload_count):
        raise ValueError("pack payload count mismatch")
    if payload_bytes != int(expected_payload_bytes):
        raise ValueError("pack payload bytes mismatch")
    return {
        "format": manifest["format"],
        "mode": "quick" if quick else "full",
        "manifest_sha256": manifest_sha256,
        "layers": meta_count,
        "payload_files": payload_count,
        "payload_bytes": payload_bytes,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wire-root", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Verify manifests, metadata hashes, names, and sizes but skip payload hashes.",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.wire_root, args.quick), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
