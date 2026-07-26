#!/usr/bin/env python3
"""Seal the logical-rate reachability contract for QTIP bitshift codebooks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


TASK = "PUBLIC_TASK"
SHAPES = {
    "fused13": (4096, 4096),
    "down": (4096, 2048),
}


def logical_bytes(k_bits: int, shape: tuple[int, int]) -> int:
    m, n = shape
    if k_bits * m * n % 8:
        raise ValueError("trellis payload is not byte aligned")
    trellis = k_bits * m * n // 8
    # SU[n] fp16, SV[m] fp16, Wscale scalar fp32.
    return trellis + 2 * n + 2 * m + 4


def logical_bpw(k_bits: int, shape: tuple[int, int]) -> float:
    m, n = shape
    return logical_bytes(k_bits, shape) * 8.0 / (m * n)


def source_math_sha256() -> str:
    payload = (
        "bitshift.pack_trellis_bits=T*K*V;"
        "trellis_states=T=values/V;"
        "logical_trellis_bits=values*K;"
        "K,V,L-integer;decoder-R=K-integer;"
        "metadata=SU-fp16+SV-fp16+Wscale-fp32"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_contract(targets: list[float], tolerance: float) -> dict[str, Any]:
    rows = []
    for k_bits in (1, 2, 3, 4):
        projections = {
            name: {
                "shape": list(shape),
                "logical_bytes_excluding_shared_tlut": logical_bytes(k_bits, shape),
                "logical_bpw_excluding_shared_tlut": logical_bpw(k_bits, shape),
            }
            for name, shape in SHAPES.items()
        }
        values = sum(m * n for m, n in SHAPES.values())
        total_bytes = sum(
            logical_bytes(k_bits, shape) for shape in SHAPES.values()
        )
        rows.append({
            "L": "does_not_change_packed_rate",
            "K": k_bits,
            "V": 2,
            "projections": projections,
            "expert_pair_logical_bpw": total_bytes * 8.0 / values,
        })

    target_rows = []
    for target in targets:
        candidates = []
        for row in rows:
            projection_max_error = max(
                abs(v["logical_bpw_excluding_shared_tlut"] - target)
                for v in row["projections"].values()
            )
            candidates.append({
                "K": row["K"],
                "projection_max_abs_error_bpw": projection_max_error,
                "within_tolerance_all_projections": projection_max_error <= tolerance,
            })
        passing = [c for c in candidates if c["within_tolerance_all_projections"]]
        target_rows.append({
            "target_bpw": target,
            "tolerance_bpw": tolerance,
            "reachable_current_uniform_codebook": bool(passing),
            "passing_configs": passing,
            "candidates": candidates,
            "verdict": "PASS_CONFIG" if passing else "UNREACHABLE_CURRENT_OPTIONS",
        })

    return {
        "schema": "qtip-rate-reachability-v1",
        "status": "PASS",
        "task": TASK,
        "created_unix": time.time(),
        "math": {
            "transition_bits_per_state": "K*V",
            "states_per_values": "1/V",
            "packed_trellis_bpw": "K",
            "logical_bpw": "K + 8*(2*n + 2*m + 4)/(m*n)",
            "depth_effect": "L changes state count/quality only; pack_trellis truncates boundary bits, so L does not change payload rate",
            "current_option_domain": "integer K in 1..4, integer V; pinned CUDA-tensor decoder requires integer R=K",
            "source_math_sha256": source_math_sha256(),
        },
        "uniform_integer_configs": rows,
        "targets": target_rows,
        "no_silent_substitution": True,
        "hybrid_note": (
            "A 50/50 mixture of independently encoded K=1 and K=2 tiles could average "
            "1.5 trellis bpw, but no such mixed-tile codebook/packer/decoder exists in the "
            "pinned current quantizer. Per-unit K=1 and K=2 artifacts are not 1.5-bpw units."
        ),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.15)
    args = ap.parse_args()
    result = build_contract([1.5, 2.0], args.tolerance)
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
