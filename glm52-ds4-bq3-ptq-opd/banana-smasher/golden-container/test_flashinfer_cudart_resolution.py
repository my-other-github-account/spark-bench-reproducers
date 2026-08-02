from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
from pathlib import Path


def _mapped_cudart_paths() -> list[str]:
    paths: list[str] = []
    with Path("/proc/self/maps").open(encoding="utf-8") as maps:
        for line in maps:
            if "libcudart" not in line or "/" not in line:
                continue
            path = line[line.index("/") :].strip()
            if path not in paths:
                paths.append(path)
    return paths


def _single_path(pattern: str) -> str:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {pattern}, found {matches}")
    return str(matches[0])


def _run(case: str) -> dict[str, object]:
    stub = _single_path(
        "/usr/local/lib/python3.12/dist-packages/tilelang/lib/libcudart_stub.so"
    )
    real = _single_path(
        "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13"
    )

    if case == "stub-first":
        import torch  # noqa: F401
        import tilelang  # noqa: F401
    elif case == "real-first":
        ctypes.CDLL(real, mode=ctypes.RTLD_GLOBAL)
    else:
        raise ValueError(f"unknown case: {case}")

    before_import = _mapped_cudart_paths()
    if case == "stub-first":
        if not before_import or before_import[0] != stub:
            raise RuntimeError(
                "stub-first precondition not reproduced: "
                f"first={before_import[:1]} expected={stub} all={before_import}"
            )
    elif not before_import or before_import[0] != real:
        raise RuntimeError(
            "real-first precondition not reproduced: "
            f"first={before_import[:1]} expected={real} all={before_import}"
        )

    # A30 failed at this exact production import seam, reached from vLLM's
    # kernel_warmup -> MiniMax sparse warmup -> allreduce RMS fusion chain.
    import flashinfer.comm  # noqa: F401
    from flashinfer.comm import cuda_ipc

    selected = str(cuda_ipc.cudart.lib._name)
    selected_name = Path(selected).name
    if selected == stub or selected_name == "libcudart_stub.so":
        raise RuntimeError(f"selected TileLang stub: {selected}")
    getattr(cuda_ipc.cudart.lib, "cudaDeviceReset")

    return {
        "case": case,
        "flashinfer_version": importlib.metadata.version("flashinfer-python"),
        "mapped_before_import": before_import,
        "real_libcudart": real,
        "selected_libcudart": selected,
        "selected_exports_cudaDeviceReset": True,
        "status": "PASS",
        "stub_libcudart": stub,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("stub-first", "real-first"), required=True)
    args = parser.parse_args()
    print(json.dumps(_run(args.case), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
