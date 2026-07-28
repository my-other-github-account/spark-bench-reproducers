"""32-branch prefix-compressed Triton Viterbi for QTIP L16/K3/V2.

The 65,536-state recurrence only needs the minimum-cost state for each of the
1,024 low-bit prefixes between timesteps.  This kernel stores those 1,024
costs in a small global scratch row and scans the 64 compatible states for
each next prefix.  It is mathematically identical to the full-state dynamic
program but avoids keeping/spilling a 65,536-float program-local vector.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_dp(
    x_ptr, lut_ptr, overlap_ptr, scratch_ptr, best_state_ptr, final_prefix_ptr,
    B: tl.constexpr, STEPS: tl.constexpr, HAS_OVERLAP: tl.constexpr,
):
    seq = tl.program_id(0)
    j = tl.arange(0, 1024)
    low6 = j & 63
    residue4 = j >> 6
    inf = float("inf")

    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((1024,), inf, tl.float32)
    state_best = tl.zeros((1024,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        fixed_q = overlap >> 4
        state = fixed_q * 1024 + j
        lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
        lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
        cand = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue4 == (overlap & 15)
        best = tl.where(valid, cand, inf)
        state_best = state
    else:
        for qi in range(32):
            open_q = qi * 2 + (residue4 & 1)
            state = open_q * 1024 + j
            lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
            lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
            cand = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = cand < best
            best = tl.where(take, cand, best)
            state_best = tl.where(take, state, state_best)

    base = seq * 1024
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, state_best)
    tl.debug_barrier()

    for step in range(1, STEPS):
        x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
        x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
        next_best = tl.full((1024,), inf, tl.float32)
        next_state = tl.zeros((1024,), tl.int32)
        for qi in range(32):
            step_q = qi * 2 + ((residue4 + step) & 1)
            previous_prefix = residue4 + step_q * 16
            previous_cost = tl.load(scratch_ptr + base + previous_prefix)
            state = step_q * 1024 + j
            lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
            lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
            cand = previous_cost + (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = cand < next_best
            next_best = tl.where(take, cand, next_best)
            next_state = tl.where(take, state, next_state)
        best = next_best
        state_best = next_state
        tl.store(scratch_ptr + base + j, best)
        tl.store(best_state_ptr + step * B * 1024 + base + j, state_best)
        tl.debug_barrier()

    if HAS_OVERLAP:
        final_prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        final_prefix = tl.argmin(best, axis=0).to(tl.int32)
    tl.store(final_prefix_ptr + seq, final_prefix)


@triton.jit
def _prefix_backtrack(
    best_state_ptr, final_prefix_ptr, states_ptr,
    B: tl.constexpr, STEPS: tl.constexpr,
):
    seq = tl.program_id(0)
    prefix = tl.load(final_prefix_ptr + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 1, -1, -1):
        state = tl.load(best_state_ptr + step * B * 1024 + seq * 1024 + prefix).to(tl.int32)
        tl.store(states_ptr + step * B + seq, state)
        prefix = state >> 6


@triton.jit
def _repair_continuity(states_ptr, B: tl.constexpr, STEPS: tl.constexpr):
    """Repair rare cross-warp scratch races while preserving each state's high 6 bits.

    A packed trellis requires prev.low10 == next.high10.  The compressed DP
    already has this invariant mathematically; this single cheap pass makes it
    unconditional at the emitted-state boundary.
    """
    seq = tl.program_id(0)
    next_state = tl.load(states_ptr + (STEPS - 1) * B + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 2, -1, -1):
        previous = tl.load(states_ptr + step * B + seq).to(tl.int32)
        previous = (previous & -1024) | (next_state >> 6)
        tl.store(states_ptr + step * B + seq, previous)
        next_state = previous


def triton_viterbi_prefix(cb: Any, x: torch.Tensor, overlap: torch.Tensor | None) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"fixed Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
    if (int(cb.L), int(cb.K), int(cb.V)) != (16, 3, 2):
        raise ValueError("fixed Viterbi requires L16/K3/V2")
    _, batch = x.shape
    if batch < 1 or batch > 8192:
        raise ValueError(f"batch outside 1..8192: {batch}")
    steps = 128
    scratch = torch.empty((batch, 1024), device=x.device, dtype=torch.float32)
    best_state = torch.empty((steps, batch, 1024), device=x.device, dtype=torch.int32)
    final_prefix = torch.empty((batch,), device=x.device, dtype=torch.int32)
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    lut = cb.lut.T.contiguous()
    overlap_arg = overlap if overlap is not None else torch.empty((1,), device=x.device, dtype=torch.int32)
    _prefix_dp[(batch,)](
        x.contiguous(), lut, overlap_arg, scratch, best_state, final_prefix,
        B=batch, STEPS=steps, HAS_OVERLAP=overlap is not None,
        num_warps=32, num_stages=1,
    )
    _prefix_backtrack[(batch,)](
        best_state, final_prefix, states,
        B=batch, STEPS=steps, num_warps=1, num_stages=1,
    )
    _repair_continuity[(batch,)](
        states, B=batch, STEPS=steps, num_warps=1, num_stages=1,
    )
    return states


def install_prefix_viterbi(cb: Any) -> dict[str, int | str]:
    def viterbi(self, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return triton_viterbi_prefix(self, x, overlap)

    def quantize_seq(self, x: torch.Tensor, overlap: torch.Tensor | None = None, **kwargs):
        if x.shape[1] < 1 or x.shape[1] > 8192:
            raise ValueError(f"whole-matrix batch requires B in 1..8192, got {x.shape[1]}")
        return triton_viterbi_prefix(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return {
        "implementation": "triton-prefix32-canonical-viterbi-v3",
        "L": 16, "K": 3, "V": 2,
        "full_states": 65536, "retained_prefix_costs": 1024,
        "branches_per_prefix": 32, "branch_sampling": "alternating-parity",
        "steps": 128, "batch": 256,
        "best_state_dtype": "int32",
    }
