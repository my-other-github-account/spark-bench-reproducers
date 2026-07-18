#!/usr/bin/env python3
"""Real-L42 correctness/performance gate for the VQ warp small-M extension."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("moe_vq_m4_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(candidate: torch.Tensor, oracle: torch.Tensor) -> dict[str, Any]:
    a, b = candidate.float(), oracle.float()
    diff = (a - b).abs()
    flat_a, flat_b = a.flatten(), b.flatten()
    cosine = float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0))
    result = {
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        "bit_equal": bool(torch.equal(candidate, oracle)),
        "equal_fraction": float((candidate == oracle).float().mean()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rms_abs": float(diff.square().mean().sqrt()),
        "cosine": cosine,
        "thresholds": {"max_abs": 0.5, "max_mean_abs": 0.01, "min_cosine": 0.9999},
    }
    result["pass"] = bool(
        result["finite"]
        and result["max_abs"] <= 0.5
        and result["mean_abs"] <= 0.01
        and result["cosine"] >= 0.9999
    )
    return result


def select_experts(arrays: dict[str, np.ndarray], kind: np.ndarray, limit: int = 8) -> list[int]:
    selected: list[int] = []
    seen: set[tuple[int, int]] = set()
    for expert in range(len(kind)):
        if int(kind[expert]) != 0:
            continue
        key = (int(arrays["dimension"][expert]), int(arrays["bits"][expert]))
        if key not in seen:
            selected.append(expert)
            seen.add(key)
        if len(selected) >= limit:
            return selected
    for expert in range(len(kind)):
        if int(kind[expert]) == 0 and expert not in selected:
            selected.append(expert)
        if len(selected) >= limit:
            break
    return selected


def time_call(callable_, *, warmup: int, reps: int) -> dict[str, Any]:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(reps):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "samples_ms": samples, "warmup": warmup, "reps": reps}


def projection(module, prefix: Path, which: str, warmup: int, reps: int) -> dict[str, Any]:
    names = [
        "codes", "scales", "codebooks", "code_offset", "scale_offset",
        "code_row_bytes", "dimension", "bits", "cb_offset",
    ]
    arrays = {
        name: np.load(f"{prefix}.vq{which}.{name}.npy", mmap_mode="r")
        for name in names
    }
    meta = json.loads(Path(f"{prefix}.meta.json").read_text())
    n, k, experts_total = int(meta[f"N{which}"]), int(meta[f"K{which}"]), int(meta["E"])
    kind_np = np.asarray(meta[f"kind{which}"], dtype=np.int32)
    arrays["n_outputs"] = n
    module.validate_projection_state(arrays, experts_total, k)

    state: dict[str, Any] = {"n_outputs": n, "layer_key": 42}
    for name in ["code_offset", "scale_offset", "code_row_bytes", "dimension", "bits", "cb_offset"]:
        state[name] = torch.tensor(np.asarray(arrays[name]), device="cuda")
    for name in ["codes", "scales", "codebooks"]:
        state[name] = torch.from_numpy(arrays[name])
    state["blob_ptrs"] = torch.tensor(
        [state[name].data_ptr() for name in ["codes", "scales", "codebooks"]],
        dtype=torch.int64,
        device="cuda",
    )

    selected = select_experts(arrays, kind_np)
    pairs, mblock = len(selected), 4
    torch.manual_seed(20260718 + int(which))
    x = torch.randn(pairs * mblock, k, dtype=torch.bfloat16, device="cuda")
    expert_blocks = torch.tensor(selected, dtype=torch.int32, device="cuda")
    num_post = torch.tensor(pairs * mblock, dtype=torch.int32, device="cuda")
    kind = torch.tensor(kind_np, dtype=torch.int32, device="cuda")
    os.environ["VLLM_MOE_VQ_FAST"] = "1"
    os.environ["VLLM_MOE_VQ_CUDA_WARP_MAX_M"] = "4"

    rows: dict[str, Any] = {}
    for valid_m in (1, 2, 3, 4):
        candidate = torch.zeros(pairs * mblock, n, dtype=torch.bfloat16, device="cuda")
        oracle = torch.zeros_like(candidate)

        os.environ["VLLM_MOE_VQ_CUDA_WARP"] = "1"
        module.vq_gemm(x, candidate, expert_blocks, num_post, kind, state,
                       n=n, k=k, mblock=mblock, valid_m=valid_m)
        os.environ["VLLM_MOE_VQ_CUDA_WARP"] = "0"
        module.vq_gemm(x, oracle, expert_blocks, num_post, kind, state,
                       n=n, k=k, mblock=mblock, valid_m=valid_m)
        torch.cuda.synchronize()

        active = torch.tensor(
            [pair * mblock + row for pair in range(pairs) for row in range(valid_m)],
            dtype=torch.long,
            device="cuda",
        )
        inactive = torch.tensor(
            [pair * mblock + row for pair in range(pairs) for row in range(valid_m, mblock)],
            dtype=torch.long,
            device="cuda",
        )
        correctness = stats(candidate.index_select(0, active), oracle.index_select(0, active))
        padded_untouched = bool(
            inactive.numel() == 0
            or (candidate.index_select(0, inactive) == 0).all()
        )

        def run_candidate() -> None:
            os.environ["VLLM_MOE_VQ_CUDA_WARP"] = "1"
            module.vq_gemm(x, candidate, expert_blocks, num_post, kind, state,
                           n=n, k=k, mblock=mblock, valid_m=valid_m)

        def run_oracle() -> None:
            os.environ["VLLM_MOE_VQ_CUDA_WARP"] = "0"
            module.vq_gemm(x, oracle, expert_blocks, num_post, kind, state,
                           n=n, k=k, mblock=mblock, valid_m=valid_m)

        candidate_timing = time_call(run_candidate, warmup=warmup, reps=reps)
        oracle_timing = time_call(run_oracle, warmup=warmup, reps=reps)
        speedup = oracle_timing["median_ms"] / candidate_timing["median_ms"]
        rows[str(valid_m)] = {
            "correctness": correctness,
            "padded_rows_untouched": padded_untouched,
            "candidate_timing": candidate_timing,
            "oracle_timing": oracle_timing,
            "speedup": speedup,
            "pass": bool(correctness["pass"] and padded_untouched and speedup > 1.0),
        }

    return {
        "projection": which,
        "shape": {"pairs": pairs, "mblock": mblock, "n": n, "k": k},
        "experts": selected,
        "dimensions": [int(arrays["dimension"][x]) for x in selected],
        "bits": [int(arrays["bits"][x]) for x in selected],
        "valid_m": rows,
        "pass": all(row["pass"] for row in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()

    import vq_warp_gemv

    module = load_module(args.module)
    projections = {
        which: projection(module, args.prefix, which, args.warmup, args.reps)
        for which in ("13", "2")
    }
    result = {
        "format": "vq-warp-real-plane-small-m-gate-v1",
        "status": "PASS" if all(value["pass"] for value in projections.values()) else "FAIL",
        "created_unix": time.time(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension": vq_warp_gemv.__file__,
        "module": str(args.module),
        "prefix": str(args.prefix),
        "contract": (
            "For T<=4, moe_align packs every expert's valid assignments before filler; "
            "each token routes to an expert at most once, so min(T,4) rows per compact "
            "expert block exactly covers every live row. The CUDA grid's z dimension "
            "computes those rows and leaves later filler rows untouched."
        ),
        "projections": projections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "summary": {
            which: {
                valid_m: {
                    "max_abs": row["correctness"]["max_abs"],
                    "mean_abs": row["correctness"]["mean_abs"],
                    "speedup": row["speedup"],
                    "pass": row["pass"],
                }
                for valid_m, row in value["valid_m"].items()
            }
            for which, value in projections.items()
        },
    }, sort_keys=True, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
