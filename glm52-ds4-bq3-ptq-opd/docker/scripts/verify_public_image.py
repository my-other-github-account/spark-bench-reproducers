#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
from pathlib import Path


def one(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {matches}")
    return matches[0]


def main() -> None:
    expected = {
        "banana-smasher": "1.0.0",
        "banana-smasher-plugin": "0.2.0",
        "flashinfer-python": "0.6.12",
        "tilelang": "0.1.9",
        "vllm": "0.24.0",
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"package identity mismatch: actual={actual} expected={expected}")

    entries = importlib.metadata.entry_points(group="vllm.general_plugins")
    if not any(
        entry.name == "banana_smasher_plugin"
        and entry.value == "banana_smasher_plugin:register"
        for entry in entries
    ):
        raise RuntimeError("banana-smasher vLLM general plugin entry point is missing")

    spec = importlib.util.find_spec("tilelang")
    if spec is None or spec.origin is None:
        raise RuntimeError("tilelang package is missing")
    stub = Path(spec.origin).parent / "lib/libcudart_stub.so"
    real = one(
        "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13"
    )
    if not stub.is_symlink() or stub.resolve() != real.resolve():
        raise RuntimeError(f"TileLang libcudart link mismatch: {stub} -> {stub.resolve()}")
    getattr(ctypes.CDLL(real, mode=ctypes.RTLD_GLOBAL), "cudaDeviceReset")
    import flashinfer.comm  # noqa: F401

    aot = Path("/opt/banana-smasher/aot")
    if len(list((aot / "cubins-sm120").glob("*.cubin"))) < 20:
        raise RuntimeError("SM12x AOT cubin set is incomplete")
    if len(list((aot / "cubins-e43").glob("*.cubin"))) < 4:
        raise RuntimeError("expert AOT cubin set is incomplete")
    print({"status": "PASS", "packages": actual, "real_libcudart": str(real)})


if __name__ == "__main__":
    main()
