"""Fused Triton Viterbi for QTIP's fixed L16/K3/V2 16x16 trellis.

One Triton program owns one independent sequence.  It keeps all 65,536 path
costs in program-local values across the 128 DP steps, eliminating the upstream
one-kernel-per-step launch/global-cost round trips.  Predecessors are emitted for
an exact separate backtrace.  The state transition and tie order are identical
to upstream bitshift_codebook.viterbi.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _viterbi_dp(
    x_ptr, lut_ptr, overlap_ptr, pred_ptr, final_ptr,
    B: tl.constexpr, STEPS: tl.constexpr,
    HAS_OVERLAP: tl.constexpr,
):
    seq = tl.program_id(0)
    state = tl.arange(0, 65536)
    prefix = state // 64
    lut0 = tl.load(lut_ptr + state * 2).to(tl.float32)
    lut1 = tl.load(lut_ptr + state * 2 + 1).to(tl.float32)
    x0 = tl.load(x_ptr + seq).to(tl.float32)
    x1 = tl.load(x_ptr + B + seq).to(tl.float32)
    d0 = lut0 - x0
    d1 = lut1 - x1
    cost = d0 * d0 + d1 * d1
    overlap = 0
    if HAS_OVERLAP:
        overlap = tl.load(overlap_ptr + seq).to(tl.int32)
        cost = tl.where(prefix == overlap, cost, float("inf"))

    pred_prefix = tl.arange(0, 1024)
    for step in range(1, STEPS):
        matrix = tl.reshape(cost, (64, 1024))
        best = tl.min(matrix, axis=0)
        high = tl.argmin(matrix, axis=0).to(tl.int32)
        previous = pred_prefix + high * 1024
        tl.store(pred_ptr + step * B * 1024 + seq * 1024 + pred_prefix, previous)
        best_by_state = tl.reshape(
            tl.broadcast_to(tl.reshape(best, (1024, 1)), (1024, 64)),
            (65536,),
        )
        x0 = tl.load(x_ptr + (step * 2) * B + seq).to(tl.float32)
        x1 = tl.load(x_ptr + (step * 2 + 1) * B + seq).to(tl.float32)
        d0 = lut0 - x0
        d1 = lut1 - x1
        cost = d0 * d0 + d1 * d1 + best_by_state

    if HAS_OVERLAP:
        cost = tl.where((state % 1024) == overlap, cost, float("inf"))
    final = tl.argmin(cost, axis=0).to(tl.int32)
    tl.store(final_ptr + seq, final)


@triton.jit
def _backtrack(pred_ptr, final_ptr, states_ptr, B: tl.constexpr, STEPS: tl.constexpr):
    seq = tl.program_id(0)
    current = tl.load(final_ptr + seq).to(tl.int32)
    tl.store(states_ptr + (STEPS - 1) * B + seq, current)
    for step in tl.static_range(STEPS - 1, 0, -1):
        prefix = current >> 6
        current = tl.load(pred_ptr + step * B * 1024 + seq * 1024 + prefix).to(tl.int32)
        tl.store(states_ptr + (step - 1) * B + seq, current)


def triton_viterbi(cb: Any, x: torch.Tensor, overlap: torch.Tensor | None) -> torch.Tensor:
    if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
        raise ValueError(f"fixed Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
    if (int(cb.L), int(cb.K), int(cb.V)) != (16, 3, 2):
        raise ValueError("fixed Viterbi requires L16/K3/V2")
    _, batch = x.shape
    if batch < 1 or batch > 256:
        raise ValueError(f"batch outside 1..256: {batch}")
    steps = 128
    pred = torch.empty((steps, batch, 1024), device=x.device, dtype=torch.int32)
    final = torch.empty((batch,), device=x.device, dtype=torch.int32)
    states = torch.empty((steps, batch), device=x.device, dtype=torch.int32)
    lut = cb.lut.T.contiguous()
    overlap_arg = overlap if overlap is not None else torch.empty((1,), device=x.device, dtype=torch.int32)
    _viterbi_dp[(batch,)](
        x.contiguous(), lut, overlap_arg, pred, final,
        B=batch, STEPS=steps, HAS_OVERLAP=overlap is not None,
        num_warps=32, num_stages=1,
    )
    _backtrack[(batch,)](pred, final, states, B=batch, STEPS=steps, num_warps=1, num_stages=1)
    return states


def install_fused_viterbi(cb: Any) -> dict[str, int | str]:
    def viterbi(self, x: torch.Tensor, overlap: torch.Tensor | None = None):
        return triton_viterbi(self, x, overlap)

    def quantize_seq(self, x: torch.Tensor, overlap: torch.Tensor | None = None, **kwargs):
        # The fit-receipt LDLQ path always presents exactly 256 independent tiles.
        if x.shape[1] != 256:
            raise ValueError(f"fit-receipt fast path requires B=256, got {x.shape[1]}")
        return triton_viterbi(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return {
        "implementation": "triton-sequence-fused-viterbi-v1",
        "L": 16,
        "K": 3,
        "V": 2,
        "states": 65536,
        "prefixes": 1024,
        "branches": 64,
        "steps": 128,
        "batch": 256,
    }
