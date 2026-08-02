"""Full-branch paired-step CUDA Viterbi for public QTIP L16/K2/V2 solves.

The extension evaluates every parity-legal q in ascending order.  Eight
independent sequences share each LUT tile, and adjacent trellis steps are paired
to keep the intermediate axis tile on-chip.  Pair graph nodes carry pre-offset
x/backpointer pointers, and the odd step sinks directly to global cost and packed
backpointer storage.  There is no proposal, pruning,
approximation, warm start, or fallback route.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from . import trellis_v2_cuda_exact
except ImportError:
    # The solve wheel carries source rather than a machine-specific cubin.  JIT
    # compilation is cached by torch and guarded by its inter-process lock, so
    # concurrent public producers still converge on one canonical extension.
    try:
        from torch.utils.cpp_extension import load

        _CSRC = Path(__file__).with_name("csrc")
        trellis_v2_cuda_exact = load(
            name="banana_smasher_trellis_v2_b256_compactprev_exact_v38",
            sources=[
                str(_CSRC / "binding_exact.cpp"),
                str(_CSRC / "trellis_v2_exact.cu"),
            ],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "-lineinfo", "--fmad=false",
            ],
            verbose=False,
        )
    except Exception as exc:  # fail loudly; a silent slow path is forbidden
        raise RuntimeError(
            "public `smash solve --qtip-profile-config` could not build the "
            "required warp-argmin LUT/backpointer-tiled exact QTIP2 CUDA producer"
        ) from exc

STEPS = 128
PREFIXES = 4096
STATES = 65536
BRANCHES = 8
_LUT_AOS_CACHE: dict[
    tuple[int, int, int], tuple[torch.Tensor, int, torch.Tensor]
] = {}


def _lut_aos(lut: torch.Tensor) -> torch.Tensor:
    stream = torch.cuda.current_stream(lut.device)
    key = (int(lut.device.index), int(stream.cuda_stream), id(lut))
    version = int(lut._version)
    cached = _LUT_AOS_CACHE.get(key)
    if cached is not None and cached[0] is lut and cached[1] == version:
        return cached[2]
    aos = lut.T.contiguous()
    _LUT_AOS_CACHE[key] = (lut, version, aos)
    return aos


def geometry(cb: Any) -> dict[str, int | str | bool]:
    got = (int(cb.L), int(cb.K), int(cb.V))
    if got != (16, 2, 2):
        raise ValueError(f"trellis-v2 is sealed for L16/K2/V2, got {got}")
    return {
        "implementation": "qtip-trellis-v2-graph-replay-b256-chunked-batch-exact-v46",
        "L": 16,
        "K": 2,
        "V": 2,
        "steps": STEPS,
        "full_states": STATES,
        "retained_prefix_costs": PREFIXES,
        "branches_per_prefix": BRANCHES,
        "branch_sampling": "alternating-parity-full",
        "strict_tie_order": "ascending-q-strict-less-than",
        "argmin_update": "PTX ordered setp.lt.f32 plus selp.f32/selp.u32; no divergent winner branch",
        "lut_layout": "aligned-float2-aos-pairs-reused-across-two-independent-sequences-per-lane",
        "backpointer_dtype": "packed-u4-two-steps-per-byte",
        "graph_node_offsets": "pair-preoffset-x-and-packed-backpointer",
        "odd_step_sink": "direct-global-cost-and-packed-backpointer",
        "parity_q_staging": "eight-legal-q-register-stage-with-branch-local-float2-lut",
        "steady_transition_math": "sm100-componentwise-fadd2-rn-ffma2-rn-over-independent-row-pairs",
        "compile_time_specialization": "first-pair-has-overlap-no-previous-and-backtrack-overlap",
        "shared_previous_layout": "parity-legal-128-prefix-compact-float-planes-with-full-q-backpointer",
        "sequences_per_cta": 8,
        "batch_contract": "exact contiguous B=256 chunks through one production kernel, up to B=8192",
        "warps_per_cta": 16,
        "final_reduction": "one-warp-per-sequence-strict-lowest-prefix-argmin",
        "steps_per_launch": 2,
        "launch_submission": "one cudaGraphLaunch replay per exact 64-pair-plus-backtrack sequence",
        "graph_static_buffers": "device-local exact x/lut/overlap/cost/backpointer/state buffers with ordered D2D ingress/egress",
        "lut_aos_materialization": "version-and-stream-keyed exact cache of canonical LUT transpose",
        "ordering": (
            "captured 64 paired exact step kernels plus exact backtrack; batch-8 LUT reuse; "
            "pre-offset graph pointers; direct odd sink; packed u4 backpointer; "
            "no proposal/pruning/fallback"
        ),
        "production_default": True,
    }


def trellis_v2_exact(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    geometry(cb)
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"x must be CUDA [256,B], got {tuple(x.shape)} {x.device}")
    if x.dtype != torch.float16:
        raise ValueError(
            f"public smash solve must produce float16 trellis input; got {x.dtype}"
        )
    batch = int(x.shape[1])
    if batch < 256 or batch > 8192 or batch % 256:
        raise ValueError(
            f"exact public QTIP2 production path requires B in 256..8192 divisible by 256, got {batch}"
        )
    if not cb.lut.is_cuda or tuple(cb.lut.shape) != (2, STATES):
        raise ValueError(
            "canonical LUT must be CUDA [2,65536], got "
            f"{tuple(cb.lut.shape)} {cb.lut.device}"
        )
    if overlap is not None and (not overlap.is_cuda or overlap.numel() != batch):
        raise ValueError("overlap must be CUDA [B]")
    lut_aos = _lut_aos(cb.lut)
    outputs = []
    for start in range(0, batch, 256):
        overlap_arg = None
        if overlap is not None:
            overlap_arg = overlap[start : start + 256].to(
                dtype=torch.int32, device=x.device
            ).contiguous()
        outputs.append(
            trellis_v2_cuda_exact.viterbi(
                x[:, start : start + 256].contiguous(), lut_aos, overlap_arg
            )[0]
        )
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
