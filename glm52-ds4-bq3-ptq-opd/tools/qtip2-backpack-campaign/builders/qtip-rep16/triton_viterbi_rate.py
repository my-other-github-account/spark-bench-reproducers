"""Rate-generic prefix-compressed Triton Viterbi for QTIP.

This preserves the deployed v3 alternating-parity recurrence while deriving the
prefix width from the sealed integer codebook geometry.  It is intentionally
limited to V=2 and L<=16, the configurations supported by the pinned QTIP
Python/CUDA-tensor decoder used by this task.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_dp(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    final_prefix_ptr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
    PREFIXES: tl.constexpr,
    SHIFT: tl.constexpr,
    RESIDUE_MASK: tl.constexpr,
    Q_FACTOR: tl.constexpr,
    BRANCH_SAMPLES: tl.constexpr,
    HAS_OVERLAP: tl.constexpr,
):
    seq = tl.program_id(0)
    j = tl.arange(0, PREFIXES)
    residue = j >> SHIFT
    inf = float("inf")

    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((PREFIXES,), inf, tl.float32)
    state_best = tl.zeros((PREFIXES,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q_overlap = overlap // Q_FACTOR
        state = q_overlap * PREFIXES + j
        lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
        lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
        cand = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue == (overlap & RESIDUE_MASK)
        best = tl.where(valid, cand, inf)
        state_best = state
    else:
        parity_initial = residue & 1
        for branch in tl.static_range(0, BRANCH_SAMPLES):
            q_initial = branch * 2 + parity_initial
            state = q_initial * PREFIXES + j
            lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
            lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
            cand = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = cand < best
            best = tl.where(take, cand, best)
            state_best = tl.where(take, state, state_best)

    base = seq * PREFIXES
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, state_best)
    tl.debug_barrier()

    for step in range(1, STEPS):
        x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
        x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
        next_best = tl.full((PREFIXES,), inf, tl.float32)
        next_state = tl.zeros((PREFIXES,), tl.int32)
        parity_step = (residue + step) & 1
        for branch in tl.static_range(0, BRANCH_SAMPLES):
            q_step = branch * 2 + parity_step
            previous_prefix = residue + q_step * Q_FACTOR
            previous_cost = tl.load(scratch_ptr + base + previous_prefix)
            state = q_step * PREFIXES + j
            lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
            lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
            cand = previous_cost + (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = cand < next_best
            next_best = tl.where(take, cand, next_best)
            next_state = tl.where(take, state, next_state)
        best = next_best
        state_best = next_state
        tl.store(scratch_ptr + base + j, best)
        tl.store(
            best_state_ptr + step * B * PREFIXES + base + j,
            state_best,
        )
        tl.debug_barrier()

    if HAS_OVERLAP:
        final_prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        final_prefix = tl.argmin(best, axis=0).to(tl.int32)
    tl.store(final_prefix_ptr + seq, final_prefix)


@triton.jit
def _prefix_backtrack(
    best_state_ptr,
    final_prefix_ptr,
    states_ptr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
    PREFIXES: tl.constexpr,
    SHIFT: tl.constexpr,
):
    seq = tl.program_id(0)
    prefix = tl.load(final_prefix_ptr + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 1, -1, -1):
        state = tl.load(
            best_state_ptr + step * B * PREFIXES + seq * PREFIXES + prefix
        ).to(tl.int32)
        tl.store(states_ptr + step * B + seq, state)
        prefix = state >> SHIFT


@triton.jit
def _repair_continuity(
    states_ptr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
    PREFIXES: tl.constexpr,
    SHIFT: tl.constexpr,
):
    """Make the packed transition invariant unconditional at emission."""
    seq = tl.program_id(0)
    next_state = tl.load(states_ptr + (STEPS - 1) * B + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 2, -1, -1):
        previous = tl.load(states_ptr + step * B + seq).to(tl.int32)
        previous = (previous & -PREFIXES) | (next_state >> SHIFT)
        tl.store(states_ptr + step * B + seq, previous)
        next_state = previous


def geometry(cb: Any) -> dict[str, int | str]:
    depth = int(cb.L)
    rate = int(cb.K)
    vector = int(cb.V)
    if vector != 2:
        raise ValueError(f"rate verifier requires V=2, got V={vector}")
    if depth < 2 or depth > 16:
        raise ValueError(f"rate verifier requires 2<=L<=16, got L={depth}")
    if rate < 1 or rate > 4:
        raise ValueError(f"rate verifier requires integer 1<=K<=4, got K={rate}")
    shift = rate * vector
    if depth <= 2 * shift:
        raise ValueError(
            f"alternating-prefix kernel requires L>2*K*V: L={depth} K={rate} V={vector}"
        )
    prefix_bits = depth - shift
    residue_bits = prefix_bits - shift
    prefixes = 1 << prefix_bits
    full_branches = 1 << shift
    return {
        "implementation": "triton-prefix-rate-generic-alternating-v1",
        "L": depth,
        "K": rate,
        "V": vector,
        "shift_bits": shift,
        "full_states": 1 << depth,
        "retained_prefix_costs": prefixes,
        "full_branches_per_prefix": full_branches,
        "branches_per_prefix": full_branches // 2,
        "branch_sampling": "alternating-parity",
        "residue_bits": residue_bits,
        "steps": 128,
        "best_state_dtype": "int32",
    }


def triton_viterbi_rate(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None,
) -> torch.Tensor:
    meta = geometry(cb)
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"fixed rate Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
    _, batch = x.shape
    if batch < 1 or batch > 8192:
        raise ValueError(f"batch outside 1..8192: {batch}")

    steps = 128
    prefixes = int(meta["retained_prefix_costs"])
    shift = int(meta["shift_bits"])
    residue_bits = int(meta["residue_bits"])
    branch_samples = int(meta["branches_per_prefix"])
    residue_mask = (1 << residue_bits) - 1
    q_factor = 1 << residue_bits

    scratch = torch.empty((batch, prefixes), device=x.device, dtype=torch.float32)
    best_state = torch.empty(
        (steps, batch, prefixes), device=x.device, dtype=torch.int32
    )
    final_prefix = torch.empty((batch,), device=x.device, dtype=torch.int32)
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    lut = cb.lut.T.contiguous()
    overlap_arg = (
        overlap
        if overlap is not None
        else torch.empty((1,), device=x.device, dtype=torch.int32)
    )
    _prefix_dp[(batch,)](
        x.contiguous(),
        lut,
        overlap_arg,
        scratch,
        best_state,
        final_prefix,
        B=batch,
        STEPS=steps,
        PREFIXES=prefixes,
        SHIFT=shift,
        RESIDUE_MASK=residue_mask,
        Q_FACTOR=q_factor,
        BRANCH_SAMPLES=branch_samples,
        HAS_OVERLAP=overlap is not None,
        num_warps=32,
        num_stages=1,
    )
    _prefix_backtrack[(batch,)](
        best_state,
        final_prefix,
        states,
        B=batch,
        STEPS=steps,
        PREFIXES=prefixes,
        SHIFT=shift,
        num_warps=1,
        num_stages=1,
    )
    _repair_continuity[(batch,)](
        states,
        B=batch,
        STEPS=steps,
        PREFIXES=prefixes,
        SHIFT=shift,
        num_warps=1,
        num_stages=1,
    )
    return states


def install_rate_viterbi(cb: Any) -> dict[str, int | str]:
    meta = geometry(cb)

    def viterbi(self, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return triton_viterbi_rate(self, x, overlap)

    def quantize_seq(
        self,
        x: torch.Tensor,
        overlap: torch.Tensor | None = None,
        **kwargs,
    ):
        return triton_viterbi_rate(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return meta
