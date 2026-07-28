#!/usr/bin/env python3
"""Verify that the architecture-specific Triton cache is complete and immutable."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(os.environ.get("TRITON_CACHE_DIR", "/opt/genesis/triton-cache"))
    manifest_path = root / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "genesis-triton-cache-v1":
        raise RuntimeError(f"wrong Triton cache schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "PASS" or manifest.get("architecture") != "sm_121":
        raise RuntimeError("Triton cache was not baked and verified for sm_121")
    required = {
        "_qtip_gemv",
        "_truevq_d4_gemv",
        "_truevq_d8_gemv",
        "_native_mxfp4_gemv",
        "streaming_dequant_bf16_dense_gemm",
    }
    compiled = {row.get("kernel") for row in manifest.get("compiled", [])}
    missing = sorted(required - compiled)
    if missing:
        raise RuntimeError(f"Triton cache manifest is missing kernels: {missing}")
    if manifest.get("marlin_api") not in {
        "vllm._custom_ops.marlin_gemm",
        "vllm._custom_ops.gptq_marlin_gemm",
    }:
        raise RuntimeError("MARLIN operator proof is missing")
    for row in manifest.get("cache_files", []):
        path = root / row["path"]
        if not path.is_file():
            raise RuntimeError(f"Triton cache file missing: {row['path']}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Triton cache file drift: {row['path']}")
    print(json.dumps({
        "status": "PASS",
        "schema": manifest["schema"],
        "architecture": manifest["architecture"],
        "cache_files": len(manifest["cache_files"]),
        "kernels": sorted(compiled),
        "additional_architectures": manifest.get("additional_architectures", []),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
