"""Test-only launch-by-step oracle for QTIP exact-path parity.

Production intentionally contains only the resident persistent kernel. This
module preserves the independently launched parity route exclusively under
``tests/``.
"""
from __future__ import annotations

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
        candidate = (
            predecessor_cost
            + (lut0 - x0) * (lut0 - x0)
            + (lut1 - x1) * (lut1 - x1)
        )
        take = candidate < best
        best = tl.where(take, candidate, best)
        chosen = tl.where(take, state, chosen)

    base = seq * 1024
    tl.store(current_ptr + base + j, best)
    tl.store(best_state_ptr + step * B * 1024 + base + j, chosen)


@triton.jit
def _backtrack_oracle(
    best_state_ptr,
    final_prefix_ptr,
    states_ptr,
    B,
    STEPS: tl.constexpr,
):
    seq = tl.program_id(0)
    prefix = tl.load(final_prefix_ptr + seq).to(tl.int32)
    for step in tl.static_range(STEPS - 1, -1, -1):
        state = tl.load(
            best_state_ptr + step * B * 1024 + seq * 1024 + prefix
        ).to(tl.int32)
        tl.store(states_ptr + step * B + seq, state)
        prefix = state >> 6


def exact_prefix_viterbi_oracle(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(
            f"exact prefix Viterbi expects CUDA [256,B], got {tuple(x.shape)}"
        )
    batch = int(x.shape[1])
    x = x.contiguous()
    lut = cb.lut.contiguous()
    scratch_a = torch.empty(
        (batch, 1024), device=x.device, dtype=torch.float32
    )
    scratch_b = torch.empty_like(scratch_a)
    best_state = torch.empty(
        (128, batch, 1024), device=x.device, dtype=torch.int32
    )
    states = torch.empty((128, batch), device=x.device, dtype=torch.int32)
    overlap_arg = (
        overlap
        if overlap is not None
        else torch.empty((1,), device=x.device, dtype=torch.int32)
    )
    _init_prefix_costs[(batch,)](
        x,
        lut,
        overlap_arg,
        scratch_a,
        best_state,
        B=batch,
        HAS_OVERLAP=overlap is not None,
        num_warps=32,
        num_stages=1,
    )
    previous, current = scratch_a, scratch_b
    for step in range(1, 128):
        _advance_prefix_costs[(batch,)](
            x,
            lut,
            previous,
            current,
            best_state,
            B=batch,
            step=step,
            num_warps=32,
            num_stages=1,
        )
        previous, current = current, previous
    final_prefix = (
        previous.argmin(dim=1).to(torch.int32)
        if overlap is None
        else overlap.to(torch.int32)
    )
    _backtrack_oracle[(batch,)](
        best_state,
        final_prefix,
        states,
        B=batch,
        STEPS=128,
        num_warps=1,
        num_stages=1,
    )
    return states
