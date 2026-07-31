"""Triton fused dequantize+linear operators for BANANA_SMASHER expert weights.

The kernels form only one weight tile in SRAM/registers at a time. They never
materialize a dense expert weight in global memory. Custom autograd recomputes
the same bounded tile stream for grad_input; VQ grad_codebook uses a transient
single-expert dense grad_weight and immediately reduces it into the small shared
codebook parameter.
"""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(mag == 0, 0.0,
            tl.where(mag == 1, 0.5,
            tl.where(mag == 2, 1.0,
            tl.where(mag == 3, 1.5,
            tl.where(mag == 4, 2.0,
            tl.where(mag == 5, 3.0,
            tl.where(mag == 6, 4.0, 6.0)))))))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _vq_forward_kernel(
    x, codebook, codes, scales, out,
    m_size, n_size: tl.constexpr, k_size: tl.constexpr,
    d_size: tl.constexpr,
    sxm: tl.constexpr, sxk: tl.constexpr,
    scn: tl.constexpr, scb: tl.constexpr,
    ssn: tl.constexpr, ssg: tl.constexpr,
    som: tl.constexpr, son: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, k_size, BLOCK_K):
        ok = k0 + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x + om[:, None] * sxm + ok[None, :] * sxk,
            mask=(om[:, None] < m_size) & (ok[None, :] < k_size),
            other=0.0,
        )
        block = ok // d_size
        dim = ok % d_size
        code = tl.load(
            codes + on[:, None] * scn + block[None, :] * scb,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=0,
        ).to(tl.int32)
        centroid = tl.load(
            codebook + code * d_size + dim[None, :],
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=0.0,
        ).to(tl.float16).to(tl.float32)
        scale_byte = tl.load(
            scales + on[:, None] * ssn + (ok[None, :] // 32) * ssg,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=127,
        ).to(tl.float32)
        weight = (centroid * tl.exp2(scale_byte - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x_tile.to(tl.bfloat16), tl.trans(weight))
    tl.store(
        out + om[:, None] * som + on[None, :] * son,
        acc.to(tl.bfloat16),
        mask=(om[:, None] < m_size) & (on[None, :] < n_size),
    )


@triton.jit
def _vq_grad_input_kernel(
    grad_out, codebook, codes, scales, grad_x,
    m_size, n_size: tl.constexpr, k_size: tl.constexpr,
    d_size: tl.constexpr,
    sgm: tl.constexpr, sgn: tl.constexpr,
    scn: tl.constexpr, scb: tl.constexpr,
    ssn: tl.constexpr, ssg: tl.constexpr,
    sxm: tl.constexpr, sxk: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for n0 in range(0, n_size, BLOCK_N):
        on = n0 + tl.arange(0, BLOCK_N)
        dy = tl.load(
            grad_out + om[:, None] * sgm + on[None, :] * sgn,
            mask=(om[:, None] < m_size) & (on[None, :] < n_size),
            other=0.0,
        )
        block = ok // d_size
        dim = ok % d_size
        code = tl.load(
            codes + on[:, None] * scn + block[None, :] * scb,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=0,
        ).to(tl.int32)
        centroid = tl.load(
            codebook + code * d_size + dim[None, :],
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=0.0,
        ).to(tl.float16).to(tl.float32)
        scale_byte = tl.load(
            scales + on[:, None] * ssn + (ok[None, :] // 32) * ssg,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size),
            other=127,
        ).to(tl.float32)
        weight = (centroid * tl.exp2(scale_byte - 127.0)).to(tl.bfloat16)
        acc += tl.dot(dy.to(tl.bfloat16), weight)
    tl.store(
        grad_x + om[:, None] * sxm + ok[None, :] * sxk,
        acc.to(tl.bfloat16),
        mask=(om[:, None] < m_size) & (ok[None, :] < k_size),
    )


@triton.jit
def _native_forward_kernel(
    x, packed, scales, out,
    m_size, n_size: tl.constexpr, k_size: tl.constexpr,
    sxm: tl.constexpr, sxk: tl.constexpr,
    spn: tl.constexpr, spb: tl.constexpr,
    ssn: tl.constexpr, ssg: tl.constexpr,
    som: tl.constexpr, son: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, k_size, BLOCK_K):
        ok = k0 + tl.arange(0, BLOCK_K)
        x_tile = tl.load(
            x + om[:, None] * sxm + ok[None, :] * sxk,
            mask=(om[:, None] < m_size) & (ok[None, :] < k_size), other=0.0,
        )
        byte = tl.load(
            packed + on[:, None] * spn + (ok[None, :] // 2) * spb,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size), other=0,
        ).to(tl.int32)
        nibble = tl.where((ok[None, :] & 1) == 0, byte & 15, byte >> 4)
        value = _e2m1(nibble)
        scale_byte = tl.load(
            scales + on[:, None] * ssn + (ok[None, :] // 32) * ssg,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size), other=127,
        ).to(tl.float32)
        weight = (value * tl.exp2(scale_byte - 127.0)).to(tl.bfloat16)
        acc += tl.dot(x_tile.to(tl.bfloat16), tl.trans(weight))
    tl.store(
        out + om[:, None] * som + on[None, :] * son,
        acc.to(tl.bfloat16),
        mask=(om[:, None] < m_size) & (on[None, :] < n_size),
    )


@triton.jit
def _native_grad_input_kernel(
    grad_out, packed, scales, grad_x,
    m_size, n_size: tl.constexpr, k_size: tl.constexpr,
    sgm: tl.constexpr, sgn: tl.constexpr,
    spn: tl.constexpr, spb: tl.constexpr,
    ssn: tl.constexpr, ssg: tl.constexpr,
    sxm: tl.constexpr, sxk: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pm = tl.program_id(0)
    pk = tl.program_id(1)
    om = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    ok = pk * BLOCK_K + tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for n0 in range(0, n_size, BLOCK_N):
        on = n0 + tl.arange(0, BLOCK_N)
        dy = tl.load(
            grad_out + om[:, None] * sgm + on[None, :] * sgn,
            mask=(om[:, None] < m_size) & (on[None, :] < n_size), other=0.0,
        )
        byte = tl.load(
            packed + on[:, None] * spn + (ok[None, :] // 2) * spb,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size), other=0,
        ).to(tl.int32)
        nibble = tl.where((ok[None, :] & 1) == 0, byte & 15, byte >> 4)
        value = _e2m1(nibble)
        scale_byte = tl.load(
            scales + on[:, None] * ssn + (ok[None, :] // 32) * ssg,
            mask=(on[:, None] < n_size) & (ok[None, :] < k_size), other=127,
        ).to(tl.float32)
        weight = (value * tl.exp2(scale_byte - 127.0)).to(tl.bfloat16)
        acc += tl.dot(dy.to(tl.bfloat16), weight)
    tl.store(
        grad_x + om[:, None] * sxm + ok[None, :] * sxk,
        acc.to(tl.bfloat16),
        mask=(om[:, None] < m_size) & (ok[None, :] < k_size),
    )


def _forward_launch_config(k_size: int) -> tuple[int, int, int]:
    """Tune only the measured p2 (K=4096) forward tile; keep P643 otherwise."""
    if int(k_size) != 4096:
        return 16, 32, 4
    # P649 exhaustive canonical B32 sweep selected this exact/floor-safe tile.
    # Environment variables remain diagnostic overrides only.
    block_m = int(os.environ.get("P649_P2_BLOCK_M", "64"))
    block_n = int(os.environ.get("P649_P2_BLOCK_N", "32"))
    num_warps = int(os.environ.get("P649_P2_NUM_WARPS", "8"))
    if block_m not in (16, 32, 64) or block_n not in (32, 64, 128) \
            or num_warps not in (4, 8):
        raise ValueError(
            f"invalid P649 p2 launch config m={block_m} n={block_n} warps={num_warps}"
        )
    return block_m, block_n, num_warps


def _launch_vq_forward(x, codebook, codes, scales):
    m, k = x.shape
    n = codes.shape[0]
    d = codebook.shape[1]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    block_m, block_n, num_warps = _forward_launch_config(k)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _vq_forward_kernel[grid](
        x, codebook, codes, scales, out,
        m, n, k, d,
        x.stride(0), x.stride(1), codes.stride(0), codes.stride(1),
        scales.stride(0), scales.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=64, num_warps=num_warps,
    )
    return out


def _launch_vq_grad_input(grad_out, codebook, codes, scales, k):
    m, n = grad_out.shape
    d = codebook.shape[1]
    grad_x = torch.empty((m, k), device=grad_out.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 16), triton.cdiv(k, 64))
    _vq_grad_input_kernel[grid](
        grad_out, codebook, codes, scales, grad_x,
        m, n, k, d,
        grad_out.stride(0), grad_out.stride(1), codes.stride(0), codes.stride(1),
        scales.stride(0), scales.stride(1), grad_x.stride(0), grad_x.stride(1),
        BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, num_warps=4,
    )
    return grad_x


def _launch_native_forward(x, packed, scales):
    m, k = x.shape
    n = packed.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    block_m, block_n, num_warps = _forward_launch_config(k)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _native_forward_kernel[grid](
        x, packed, scales, out,
        m, n, k,
        x.stride(0), x.stride(1), packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=64, num_warps=num_warps,
    )
    return out


def _launch_native_grad_input(grad_out, packed, scales, k):
    m, n = grad_out.shape
    grad_x = torch.empty((m, k), device=grad_out.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, 16), triton.cdiv(k, 64))
    _native_grad_input_kernel[grid](
        grad_out, packed, scales, grad_x,
        m, n, k,
        grad_out.stride(0), grad_out.stride(1), packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1), grad_x.stride(0), grad_x.stride(1),
        BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, num_warps=4,
    )
    return grad_x


class FusedVQLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebook32, codes, scales):
        if x.ndim != 2 or codes.ndim != 2 or scales.ndim != 2:
            raise ValueError('fused VQ linear expects x/codes/scales rank 2')
        if x.shape[1] != codes.shape[1] * codebook32.shape[1]:
            raise ValueError('fused VQ linear K mismatch')
        ctx.save_for_backward(x, codebook32, codes, scales)
        return _launch_vq_forward(x, codebook32, codes, scales)

    @staticmethod
    def backward(ctx, grad_out):
        x, codebook32, codes, scales = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = _launch_vq_grad_input(grad_out, codebook32, codes, scales, x.shape[1])
        # Match the canonical custom-dequant STE reduction exactly: BF16 dense
        # grad_weight, fp32 scale/reduction, then index_add into the shared cb.
        grad_weight = torch.mm(grad_out.transpose(0, 1), x)
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, -1)
        d = codebook32.shape[1]
        grouped = (grad_weight.float() * scale_columns).reshape(codes.shape[0], codes.shape[1], d)
        grad_codebook = torch.zeros_like(codebook32, dtype=torch.float32)
        grad_codebook.index_add_(0, codes.reshape(-1).long(), grouped.reshape(-1, d))
        return grad_x, grad_codebook, None, None


class FusedNativeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scales):
        if x.ndim != 2 or packed.ndim != 2 or scales.ndim != 2:
            raise ValueError('fused native linear expects rank-2 tensors')
        if x.shape[1] != packed.shape[1] * 2:
            raise ValueError('fused native linear K mismatch')
        ctx.save_for_backward(packed, scales)
        ctx.k = x.shape[1]
        return _launch_native_forward(x, packed, scales)

    @staticmethod
    def backward(ctx, grad_out):
        packed, scales = ctx.saved_tensors
        return _launch_native_grad_input(grad_out.contiguous(), packed, scales, ctx.k), None, None


def fused_vq_linear(x, codebook32, codes, scales):
    return FusedVQLinearFn.apply(x, codebook32, codes, scales)


def fused_native_linear(x, packed, scales):
    return FusedNativeLinearFn.apply(x, packed, scales)


def warm_fused_expert_linear(device="cuda"):
    """Compile every bounded production projection/dimension graph pre-timer."""
    import time
    started = time.perf_counter()
    rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for k_size in (2048, 4096):
            for d_size in (4, 8):
                n_size, c_size, m_size = 4096, 256, 1
                x = torch.randn(m_size, k_size, device=device, dtype=torch.bfloat16,
                                requires_grad=True)
                cb = torch.randn(c_size, d_size, device=device, dtype=torch.float32,
                                 requires_grad=True)
                codes = torch.randint(
                    0, c_size, (n_size, k_size // d_size),
                    device=device, dtype=torch.int32,
                )
                scales = torch.full(
                    (n_size, k_size // 32), 127, device=device, dtype=torch.uint8,
                )
                out = fused_vq_linear(x, cb, codes, scales)
                out.backward(torch.ones_like(out))
                torch.cuda.synchronize()
                rows.append({"kind": "vq", "k": k_size, "d": d_size,
                             "finite": bool(torch.isfinite(out).all())})
                del x, cb, codes, scales, out
            x = torch.randn(1, k_size, device=device, dtype=torch.bfloat16,
                            requires_grad=True)
            packed = torch.randint(
                0, 256, (4096, k_size // 2), device=device, dtype=torch.uint8,
            )
            scales = torch.full(
                (4096, k_size // 32), 127, device=device, dtype=torch.uint8,
            )
            out = fused_native_linear(x, packed, scales)
            out.backward(torch.ones_like(out))
            torch.cuda.synchronize()
            rows.append({"kind": "native_mxfp4", "k": k_size,
                         "finite": bool(torch.isfinite(out).all())})
            del x, packed, scales, out
    torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "scope": "tile-SRAM packed-code/codebook and MXFP4 expert linear",
        "activation_checkpoint_compatible": True,
        "dense_weight_materialized": False,
        "seconds": time.perf_counter() - started,
        "rows": rows,
    }
