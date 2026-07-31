#!/usr/bin/env python3
"""Compile every shipped mixed-tier Triton kernel and seal the cache."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backend(path: Path):
    spec = importlib.util.spec_from_file_location("mixed_tier_backend_warmup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import torch
    import triton

    runtime = Path(os.environ.get("BANANA_SMASHER_RUNTIME", "/opt/banana_smasher/runtime"))
    artifact = Path(os.environ.get(
        "BANANA_SMASHER_WARMUP_ARTIFACT", "/opt/banana_smasher/artifacts/mixed_tier_compact.pt"
    ))
    cache = Path(os.environ.get("TRITON_CACHE_DIR", "/cache"))
    cache.mkdir(parents=True, exist_ok=True)
    capability = torch.cuda.get_device_capability()
    sm = capability[0] * 10 + capability[1]
    if sm != 121:
        raise RuntimeError(f"kernel bake requires GB10 sm_121; got sm_{sm}")

    backend = load_backend(runtime / "mixed_tier_backend.py")
    layer = backend.MixedTierLayer.from_file(artifact, layer_index=0, device="cuda")
    started = time.monotonic()
    compiled = []
    with torch.inference_mode():
        for projection_name in ("fused13", "down"):
            for tier in backend.TIER_NAMES:
                projection = layer.projections[projection_name][tier]
                decode = torch.zeros(
                    (1, projection.k), device="cuda", dtype=torch.bfloat16
                )
                projection.forward_triton(decode)
                compiled.append({
                    "kernel": f"_{tier}_gemv" if tier != "qtip" else "_qtip_gemv",
                    "projection": projection_name,
                    "path": "decode",
                    "m": "runtime",
                    "n": projection.n,
                    "k": projection.k,
                })
                prefill = torch.zeros(
                    (64, projection.k), device="cuda", dtype=torch.bfloat16
                )
                projection.forward_dense(prefill)
                compiled.append({
                    "kernel": "streaming_dequant_bf16_dense_gemm",
                    "projection": projection_name,
                    "tier": tier,
                    "path": "prefill",
                    "m_threshold": 64,
                    "n": projection.n,
                    "k": projection.k,
                })
                if tier != "native_mxfp4":
                    projection.forward_mbatched(prefill)
                    compiled.append({
                        "kernel": "_vq_gemm_mbatched",
                        "projection": projection_name,
                        "tier": tier,
                        "path": "alternate_prefill",
                        "m": 64,
                        "n": projection.n,
                        "k": projection.k,
                    })
        torch.cuda.synchronize()

    from vllm import _custom_ops as custom_ops
    marlin_name = next(
        (
            name
            for name in ("marlin_gemm", "gptq_marlin_gemm")
            if callable(getattr(custom_ops, name, None))
        ),
        None,
    )
    if marlin_name is None:
        raise RuntimeError("vLLM MARLIN operator is not present in the frozen runtime")

    files = sorted(
        path for path in cache.rglob("*")
        if path.is_file() and path.name != "CACHE_MANIFEST.json"
    )
    if not files:
        raise RuntimeError("Triton warmup produced no cache files")
    manifest = {
        "schema": "banana_smasher-triton-cache-v1",
        "status": "PASS",
        "architecture": "sm_121",
        "additional_architectures": [],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "marlin_api": f"vllm._custom_ops.{marlin_name}",
        "decode_row_count_compile_key": False,
        "compiled": compiled,
        "elapsed_seconds": time.monotonic() - started,
        "cache_files": [
            {
                "path": path.relative_to(cache).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (cache / "CACHE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
