#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

PACKAGES = (
    "banana-smasher",
    "banana-smasher-plugin",
    "flashinfer-python",
    "numpy",
    "quack-kernels",
    "safetensors",
    "tilelang",
    "torch",
    "triton",
    "vllm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_package_receipt.py OUTPUT")
    output = Path(sys.argv[1])
    aot_root = Path("/opt/banana-smasher/aot")
    aot = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(aot_root).as_posix(),
            "sha256": sha256(path),
        }
        for path in sorted(aot_root.rglob("*.cubin"))
    ]
    payload = {
        "aot_assets": aot,
        "packages": [
            {"name": name, "version": importlib.metadata.version(name)}
            for name in PACKAGES
        ],
        "schema": "banana-smasher-package-inventory-v1",
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
