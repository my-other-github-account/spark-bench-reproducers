"""P526 single candidate: explicit-M packed-QTIP/trueVQ gather-dequant-GEMM.

The candidate reconstructs only register tiles and writes one BF16 output tile. It
never materializes or retains a dense second weight copy.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qtip_gemm_mbatched(
    x_ptr, codes_ptr, scales_ptr, cb_ptr, expert_ptr, y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    r = tl.program_id(1)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn - pid_m * grid_n
    e = tl.load(expert_ptr + r).to(tl.int64)
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    m_mask = m < M
    n_mask = n < N
    acc = tl.zeros((BM, BN), tl.float32)
    codes_k = K // 8
    scales_k = K // 32
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        xv = tl.load(
            x_ptr + (r * M + m[:, None]) * K + k[None, :],
            mask=m_mask[:, None] & (k[None, :] < K), other=0.0,
        )
        code = tl.load(
            codes_ptr + e * N * codes_k + n[:, None] * codes_k + (k[None, :] // 8),
            mask=n_mask[:, None] & (k[None, :] < K), other=0,
        ).to(tl.int32)
        weight = tl.load(cb_ptr + code * 8 + (k[None, :] % 8)).to(tl.float32)
        scale_u8 = tl.load(
            scales_ptr + e * N * scales_k + n[:, None] * scales_k + (k[None, :] // 32),
            mask=n_mask[:, None] & (k[None, :] < K), other=127,
        ).to(tl.float32)
        weight *= tl.exp2(scale_u8 - 127.0)
        acc += tl.dot(xv.to(tl.bfloat16), tl.trans(weight.to(tl.bfloat16)))
    tl.store(
        y_ptr + (r * M + m[:, None]) * N + n[None, :],
        acc, mask=m_mask[:, None] & n_mask[None, :],
    )


def qtip_gemm_mbatched(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor,
                       codebook: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """Run the one candidate kernel on x[R,M,K] and packed experts[E,N,*]."""
    if x.ndim != 3 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("expected x[R,M,K], codes[E,N,K/8], scales[E,N,K/32]")
    r, m, k = x.shape
    e, n, codes_k = codes.shape
    if codes_k * 8 != k or scales.shape != (e, n, k // 32):
        raise ValueError("packed shape mismatch")
    x = x.to(torch.bfloat16).contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    y = torch.empty((r, m, n), device=x.device, dtype=torch.bfloat16)
    bm, bn, bk = 16, 32, 32
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), r)
    _qtip_gemm_mbatched[grid](
        x, codes, scales, codebook, expert_ids, y,
        M=m, N=n, K=k, BM=bm, BN=bn, BK=bk,
        num_warps=4, num_stages=2,
    )
    return y
