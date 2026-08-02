"""Public build entry point for exact CUDA producers."""
from __future__ import annotations

import hashlib
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def build_kernel(*, tier: str, bpw: float) -> dict[str, Any]:
    """Build the requested exact producer through its shipped source loader."""
    if tier != "qtip" or bpw != 2.0:
        raise ValueError(
            "the public exact QTIP producer is sealed for --tier qtip --bpw 2.00"
        )
    try:
        from .trellis_v2 import exact

        metadata = exact.geometry(SimpleNamespace(L=16, K=2, V=2))
        producer = Path(exact.trellis_v2_cuda_exact.__file__).resolve()
    except Exception as exc:
        raise RuntimeError(
            "public exact QTIP2 CUDA producer build failed; root build traceback:\n"
            + traceback.format_exc()
        ) from exc

    return {
        "schema": "banana-smasher-kernel-build-v1",
        "status": "PASS",
        "tier": tier,
        "bpw": bpw,
        "implementation": metadata["implementation"],
        "producer": str(producer),
        "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
        "cache_root": os.environ.get("TORCH_EXTENSIONS_DIR"),
    }
