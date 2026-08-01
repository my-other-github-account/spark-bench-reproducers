"""Exact prefix-compressed Triton Viterbi for QTIP L16/K3/V2.

The production kernel assigns one independent sequence to each CTA and keeps all
128 dynamic-programming steps in one launch.  The launch-by-step implementation
is retained only as an exact test/acceptance oracle.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _init_prefix_costs(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    B,
    HAS_OVERLAP: tl.constexpr,
):
    seq = tl.program_id(0)
    j = tl.arange(0, 1024)
    residue4 = j >> 6
    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((1024,), float("inf"), tl.float32)
    chosen = tl.zeros((1024,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q = overlap >> 4
        state = q * 1024 + j
        lut0 = tl.load(lut_ptr + state).to(tl.float32)
        lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
        candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue4 == (overlap & 15)
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        for q in range(64):
            state = q * 1024 + j
            lut0 = tl.load(lut_ptr + state).to(tl.float32)
            lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
            candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)

    base = seq * 1024
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, chosen)


@triton.jit
def _advance_prefix_costs(
    x_ptr,
    lut_ptr,
    previous_ptr,
    current_ptr,
    best_state_ptr,
    B,
    step,
):
    seq = tl.program_id(0)
    j = tl.arange(0, 1024)
    residue4 = j >> 6
    x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
    x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
    best = tl.full((1024,), float("inf"), tl.float32)
    chosen = tl.zeros((1024,), tl.int32)
    previous_base = seq * 1024

    for q in range(64):
        predecessor_prefix = q * 16 + residue4
        predecessor_cost = tl.load(previous_ptr + previous_base + predecessor_prefix)
        state = q * 1024 + j
        lut0 = tl.load(lut_ptr + state).to(tl.float32)
        lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
        candidate = predecessor_cost + (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        take = candidate < best
        best = tl.where(take, candidate, best)
        chosen = tl.where(take, state, chosen)

    base = seq * 1024
    tl.store(current_ptr + base + j, best)
    tl.store(best_state_ptr + step * B * 1024 + base + j, chosen)


@triton.jit
def _backtrack_reference(
    best_state_ptr,
    final_prefix_ptr,
    states_ptr,
    B,
    STEPS: tl.constexpr,
):
    seq = tl.program_id(0)
    prefix = tl.load(final_prefix_ptr + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 1, -1, -1):
        state = tl.load(best_state_ptr + step * B * 1024 + seq * 1024 + prefix).to(tl.int32)
        tl.store(states_ptr + step * B + seq, state)
        prefix = state >> 6


def exact_prefix_viterbi_reference(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch-by-step exact oracle retained for tests and acceptance only."""
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"exact prefix Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
    batch = int(x.shape[1])
    x = x.contiguous()
    # Keep the canonical codebook as two contiguous state planes.  Every
    # transition step consumes both planes for the same 1,024-state tile, so
    # structure-of-arrays loads are fully coalesced instead of stride-two.
    lut = cb.lut.contiguous()
    scratch_a = torch.empty((batch, 1024), device=x.device, dtype=torch.float32)
    scratch_b = torch.empty_like(scratch_a)
    best_state = torch.empty((128, batch, 1024), device=x.device, dtype=torch.int32)
    states = torch.empty((128, batch), device=x.device, dtype=torch.int32)
    overlap_arg = overlap if overlap is not None else torch.empty((1,), device=x.device, dtype=torch.int32)
    _init_prefix_costs[(batch,)](
        x, lut, overlap_arg, scratch_a, best_state,
        B=batch, HAS_OVERLAP=overlap is not None, num_warps=32, num_stages=1,
    )
    previous, current = scratch_a, scratch_b
    for step in range(1, 128):
        _advance_prefix_costs[(batch,)](
            x, lut, previous, current, best_state,
            B=batch, step=step, num_warps=32, num_stages=1,
        )
        previous, current = current, previous
    final_prefix = previous.argmin(dim=1).to(torch.int32) if overlap is None else overlap.to(torch.int32)
    _backtrack_reference[(batch,)](
        best_state, final_prefix, states,
        B=batch, STEPS=128, num_warps=1, num_stages=1,
    )
    return states


@triton.jit
def _persistent_prefix_viterbi(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    states_ptr,
    B,
    HAS_OVERLAP: tl.constexpr,
):
    """Solve one independent sequence per CTA with all timesteps resident.

    The two 1,024-entry cost rows ping-pong in task-local global scratch.  A CTA
    barrier replaces the 127 host launches while preserving the original q=0..63
    strict-< update order and exact int32 backpointer table.
    """
    seq = tl.program_id(0)
    j = tl.arange(0, 1024)
    residue4 = j >> 6
    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    best = tl.full((1024,), float("inf"), tl.float32)
    chosen = tl.zeros((1024,), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        q = overlap >> 4
        state = q * 1024 + j
        lut0 = tl.load(lut_ptr + state).to(tl.float32)
        lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
        candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
        valid = residue4 == (overlap & 15)
        best = tl.where(valid, candidate, best)
        chosen = state
    else:
        for q in range(64):
            state = q * 1024 + j
            lut0 = tl.load(lut_ptr + state).to(tl.float32)
            lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
            candidate = (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)

    base = seq * 1024
    tl.store(scratch_ptr + base + j, best)
    tl.store(best_state_ptr + base + j, chosen)
    tl.debug_barrier()

    step = 1
    while step < 128:
        previous_base = ((step - 1) & 1) * B * 1024 + base
        current_base = (step & 1) * B * 1024 + base
        x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
        x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
        best = tl.full((1024,), float("inf"), tl.float32)
        chosen = tl.zeros((1024,), tl.int32)
        for q in range(64):
            predecessor_prefix = q * 16 + residue4
            predecessor_cost = tl.load(scratch_ptr + previous_base + predecessor_prefix)
            state = q * 1024 + j
            lut0 = tl.load(lut_ptr + state).to(tl.float32)
            lut1 = tl.load(lut_ptr + 65536 + state).to(tl.float32)
            candidate = predecessor_cost + (lut0 - x0) * (lut0 - x0) + (lut1 - x1) * (lut1 - x1)
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, state, chosen)
        tl.store(scratch_ptr + current_base + j, best)
        tl.store(best_state_ptr + step * B * 1024 + base + j, chosen)
        tl.debug_barrier()
        step += 1

    if HAS_OVERLAP:
        prefix = tl.load(overlap_ptr + seq).to(tl.int32)
    else:
        prefix = tl.argmin(best, axis=0).to(tl.int32)
    for back_step in tl.static_range(127, -1, -1):
        state = tl.load(best_state_ptr + back_step * B * 1024 + base + prefix).to(tl.int32)
        tl.store(states_ptr + back_step * B + seq, state)
        prefix = state >> 6


@triton.jit
def _batched_prefix_viterbi(
    x_ptr,
    lut_ptr,
    overlap_ptr,
    scratch_ptr,
    best_state_ptr,
    B,
    HAS_OVERLAP: tl.constexpr,
    SEQS_PER_CTA: tl.constexpr,
):
    """Batch independent sequences in each CTA and reuse transition tiles.

    The q-major LUT tile is identical for every independent sequence.  Keeping
    SEQS_PER_CTA sequences in one program loads each state tile once, broadcasts
    it across the sequence rows, and preserves the scalar q=0..63 strict-<
    ordering for every sequence/prefix pair.  Backtracking remains a separate
    exact kernel because the final argmin spans the complete prefix row.
    """
    seq = tl.program_id(0) * SEQS_PER_CTA + tl.arange(0, SEQS_PER_CTA)[:, None]
    valid_seq = seq < B
    j = tl.arange(0, 1024)[None, :]
    residue4 = j >> 6
    x0 = tl.load(x_ptr + seq, mask=valid_seq, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq, mask=valid_seq, other=0.0).to(tl.float32)
    best = tl.full((SEQS_PER_CTA, 1024), float("inf"), tl.float32)
    chosen = tl.zeros((SEQS_PER_CTA, 1024), tl.int32)

    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq, mask=valid_seq, other=0).to(tl.int32)
        initial_q = overlap >> 4
        initial_state = initial_q * 1024 + j
        initial_lut0 = tl.load(lut_ptr + initial_state).to(tl.float32)
        initial_lut1 = tl.load(lut_ptr + 65536 + initial_state).to(tl.float32)
        candidate = (initial_lut0 - x0) * (initial_lut0 - x0) + (initial_lut1 - x1) * (initial_lut1 - x1)
        valid = residue4 == (overlap & 15)
        best = tl.where(valid, candidate, best)
        chosen = initial_state
    else:
        for q in range(64):
            initial_state = q * 1024 + j
            # These two contiguous tiles are shared by all sequence rows.
            initial_lut0 = tl.load(lut_ptr + initial_state).to(tl.float32)
            initial_lut1 = tl.load(lut_ptr + 65536 + initial_state).to(tl.float32)
            candidate = (
                (initial_lut0 - x0) * (initial_lut0 - x0)
                + (initial_lut1 - x1) * (initial_lut1 - x1)
            )
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, initial_state, chosen)

    base = seq * 1024
    tl.store(scratch_ptr + base + j, best, mask=valid_seq)
    tl.store(best_state_ptr + base + j, chosen, mask=valid_seq)
    tl.debug_barrier()

    step = 1
    while step < 128:
        previous_base = ((step - 1) & 1) * B * 1024 + base
        current_base = (step & 1) * B * 1024 + base
        x0 = tl.load(
            x_ptr + (step * 2) * B + seq, mask=valid_seq, other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + (step * 2 + 1) * B + seq, mask=valid_seq, other=0.0
        ).to(tl.float32)
        best = tl.full((SEQS_PER_CTA, 1024), float("inf"), tl.float32)
        chosen = tl.zeros((SEQS_PER_CTA, 1024), tl.int32)
        for q in range(64):
            predecessor_prefix = q * 16 + residue4
            predecessor_cost = tl.load(
                scratch_ptr + previous_base + predecessor_prefix,
                mask=valid_seq,
                other=float("inf"),
            )
            transition_state = q * 1024 + j
            transition_lut0 = tl.load(lut_ptr + transition_state).to(tl.float32)
            transition_lut1 = tl.load(
                lut_ptr + 65536 + transition_state
            ).to(tl.float32)
            candidate = (
                predecessor_cost
                + (transition_lut0 - x0) * (transition_lut0 - x0)
                + (transition_lut1 - x1) * (transition_lut1 - x1)
            )
            take = candidate < best
            best = tl.where(take, candidate, best)
            chosen = tl.where(take, transition_state, chosen)
        tl.store(scratch_ptr + current_base + j, best, mask=valid_seq)
        tl.store(
            best_state_ptr + step * B * 1024 + base + j,
            chosen,
            mask=valid_seq,
        )
        tl.debug_barrier()
        step += 1


def exact_prefix_viterbi(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return exact full-branch Viterbi states for CUDA input shaped [256, B]."""
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"exact prefix Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
    if (int(cb.L), int(cb.K), int(cb.V)) != (16, 3, 2):
        raise ValueError("exact prefix Viterbi requires L16/K3/V2")
    batch = int(x.shape[1])
    if batch < 1 or batch > 8192:
        raise ValueError(f"batch outside 1..8192: {batch}")
    if overlap is not None and (not overlap.is_cuda or overlap.numel() != batch):
        raise ValueError("overlap must be a CUDA tensor with one prefix per sequence")

    steps = 128
    x = x.contiguous()
    # Keep the canonical codebook as two contiguous state planes.  Every
    # transition step consumes both planes for the same 1,024-state tile, so
    # structure-of-arrays loads are fully coalesced instead of stride-two.
    lut = cb.lut.contiguous()
    scratch = torch.empty((2, batch, 1024), device=x.device, dtype=torch.float32)
    best_state = torch.empty((steps, batch, 1024), device=x.device, dtype=torch.int32)
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    overlap_arg = overlap if overlap is not None else torch.empty((1,), device=x.device, dtype=torch.int32)
    _persistent_prefix_viterbi[(batch,)](
        x,
        lut,
        overlap_arg,
        scratch,
        best_state,
        states,
        B=batch,
        HAS_OVERLAP=overlap is not None,
        # Sixteen warps preserve the exact j=0..1023 and q=0..63 order while
        # allowing two independent sequence CTAs to reside per 32-warp budget.
        num_warps=16,
        num_stages=1,
    )
    return states


def install_exact_prefix_viterbi(
    cb: Any,
    *,
    reference: bool = False,
) -> dict[str, int | str | float]:
    solve = exact_prefix_viterbi_reference if reference else exact_prefix_viterbi

    def viterbi(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return solve(self, x, overlap)

    def quantize_seq(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None, **_: Any):
        return solve(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return {
        "implementation": (
            "launch-by-step-exact-prefix-dp-reference-v1"
            if reference else "persistent-exact-prefix-dp-v1"
        ),
        "L": 16,
        "K": 3,
        "V": 2,
        "full_states": 65536,
        "retained_prefix_costs": 1024,
        "branches_per_prefix": 64,
        "branch_sampling": "full",
        "steps": 128,
        "min_exact_quality": 1.0,
        "ordering": (
            "separate-stream-ordered-kernel-launches"
            if reference else "one persistent launch per independent sequence batch"
        ),
        "best_state_dtype": "int32",
    }
