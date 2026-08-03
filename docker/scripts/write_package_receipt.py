#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

from verify_public_image import (
    EXPECTED_PACKAGES,
    verify_asset_set,
    verify_provenance_manifests,
)

PACKAGES = (
    "banana-smasher",
    "banana-smasher-plugin",
    "deep-gemm",
    "flashinfer-python",
    "numpy",
    "quack-kernels",
    "safetensors",
    "tilelang",
    "torch",
    "triton",
    "vllm",
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_package_receipt.py OUTPUT")
    output = Path(sys.argv[1])
    plugin = importlib.util.find_spec("banana_smasher_plugin")
    if plugin is None or plugin.origin is None:
        raise RuntimeError("banana-smasher plugin package is missing")
    provenance_root = Path("/opt/banana-smasher/provenance")
    provenance = verify_provenance_manifests(provenance_root)
    assets = verify_asset_set(
        provenance_root / "ASSET_MANIFEST.json",
        Path("/opt/banana-smasher/aot"),
        Path(plugin.origin).parent / "qtip_tlut.npy",
    )
    packages = [
        {"name": name, "version": importlib.metadata.version(name)}
        for name in PACKAGES
    ]
    core = {item["name"]: item["version"] for item in packages if item["name"] in EXPECTED_PACKAGES}
    if core != EXPECTED_PACKAGES:
        raise RuntimeError(
            f"package receipt core mismatch: actual={core} expected={EXPECTED_PACKAGES}"
        )
    payload = {
        "active_assets": assets,
        "packages": packages,
        "provenance_manifests": provenance["manifest_sha256"],
        "schema": "banana-smasher-package-inventory-v2",
        "status": "PASS",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
