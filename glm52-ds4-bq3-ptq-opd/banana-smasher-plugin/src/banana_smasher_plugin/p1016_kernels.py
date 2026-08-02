from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qtip_raw_gemv(
    x_ptr,
    blocked_ptr_holder,
    lut_ptr,
    y_ptr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    RATE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    blocked_addr = tl.load(blocked_ptr_holder).to(tl.int64)
    blocked_ptr = tl.cast(blocked_addr, tl.pointer_type(tl.uint16))
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < K
        x = tl.load(
            x_ptr + r * K + k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        local_n = n[:, None] & 15
        local_k = k[None, :] & 15
        q = (
            (local_n & 7) * 32
            + (local_n // 8) * 2
            + ((local_k & 7) // 2) * 8
            + (local_k & 1)
            + (local_k // 8) * 4
        )
        pair = q // 2
        component = q & 1
        selected_bit_position = pair * (RATE << 1)
        word_index = selected_bit_position // 16
        shift = selected_bit_position & 15
        block = (n[:, None] // 16) * (K // 16) + (k[None, :] // 16)
        offset = block * (RATE * 16) + word_index
        curr = tl.load(
            blocked_ptr + offset,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        ).to(tl.int64)
        nxt = tl.load(
            blocked_ptr
            + block * (RATE * 16)
            + ((word_index + 1) % (RATE * 16)),
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        ).to(tl.int64)
        word = (curr << 16) | nxt
        index = (word >> (16 - shift)) & 65535
        weight = tl.load(lut_ptr + index * 2 + component).to(tl.float32)
        acc += tl.sum(weight * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _d4_gemv(
    x_ptr,
    codes_ptr_holder,
    scales_ptr_holder,
    codebook_ptr_holder,
    y_ptr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    codes_ptr = tl.cast(
        tl.load(codes_ptr_holder + r).to(tl.int64), tl.pointer_type(tl.int16)
    )
    scales_ptr = tl.cast(
        tl.load(scales_ptr_holder + r).to(tl.int64), tl.pointer_type(tl.uint8)
    )
    cb_ptr = tl.cast(
        tl.load(codebook_ptr_holder + r).to(tl.int64), tl.pointer_type(tl.float16)
    )
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        mask = n_mask[:, None] & (k[None, :] < K)
        x = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        code = tl.load(
            codes_ptr + n[:, None] * (K // D) + (k[None, :] // D),
            mask=mask,
            other=0,
        ).to(tl.int32)
        value = tl.load(cb_ptr + code * D + (k[None, :] % D), mask=mask, other=0.0)
        scale = tl.load(
            scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
            mask=mask,
            other=127,
        ).to(tl.float32)
        acc += tl.sum(value.to(tl.float32) * tl.exp2(scale - 127.0) * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _e2m1(code):
    mag = code & 7
    value = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _native_mxfp4_gemv(
    x_ptr,
    packed_ptr_holder,
    scales_ptr_holder,
    y_ptr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    packed_ptr = tl.cast(
        tl.load(packed_ptr_holder + r).to(tl.int64), tl.pointer_type(tl.uint8)
    )
    scales_ptr = tl.cast(
        tl.load(scales_ptr_holder + r).to(tl.int64), tl.pointer_type(tl.uint8)
    )
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        mask = n_mask[:, None] & (k[None, :] < K)
        x = tl.load(x_ptr + r * K + k, mask=k < K, other=0.0).to(tl.float32)
        byte = tl.load(
            packed_ptr + n[:, None] * (K // 2) + (k[None, :] // 2),
            mask=mask,
            other=0,
        ).to(tl.int32)
        code = tl.where((k[None, :] & 1) == 0, byte & 15, byte >> 4)
        scale = tl.load(
            scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32),
            mask=mask,
            other=127,
        ).to(tl.float32)
        acc += tl.sum(_e2m1(code) * tl.exp2(scale - 127.0) * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


def pointer_holder(*tensors: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.int64,
        device="cuda",
    )


def qtip_blocked(compressed: torch.Tensor, m: int, k: int, rate: int) -> torch.Tensor:
    compressed = compressed.contiguous().view(torch.uint16)
    blocked = (
        compressed.view(torch.uint8)
        .reshape(m // 32, k // 32, 32, 2, 2, rate)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, rate * 16, 2)
        .flip((-1,))
        .contiguous()
        .view(torch.uint16)
        .reshape(m // 16, k // 16, rate * 16)
    )
    return blocked


def qtip_raw_gemv(x: torch.Tensor, blocked: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    n = blocked.shape[0] * 16
    rate = blocked.shape[-1] // 16
    holder = pointer_holder(blocked)
    y = torch.empty((r, n), dtype=torch.float32, device=x.device)
    _qtip_raw_gemv[(triton.cdiv(n, 8), r)](
        x,
        holder,
        lut,
        y,
        R=r,
        N=n,
        K=k,
        RATE=rate,
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


def d4_gemv(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    n = codes.shape[0]
    holders = pointer_holder(codes, scales, codebook)
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    _d4_gemv[(triton.cdiv(n, 8), r)](
        x,
        holders[0:1],
        holders[1:2],
        holders[2:3],
        y,
        R=r,
        N=n,
        K=k,
        D=codebook.shape[-1],
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


def native_mxfp4_gemv(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    n = scales.shape[0]
    holders = pointer_holder(packed, scales)
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    _native_mxfp4_gemv[(triton.cdiv(n, 8), r)](
        x,
        holders[0:1],
        holders[1:2],
        y,
        R=r,
        N=n,
        K=k,
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


@triton.jit
def _qtip_raw_gemv_source(
    x_ptr,
    source_ptr_holders,
    offset_map,
    lut_ptr,
    y_ptr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    RATE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    source_addr = tl.load(source_ptr_holders + r).to(tl.int64)
    source_ptr = tl.cast(source_addr, tl.pointer_type(tl.uint8))
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < K
        x = tl.load(x_ptr + r * K + k, mask=k_mask, other=0.0).to(tl.float32)
        local_n = n[:, None] & 15
        local_k = k[None, :] & 15
        q = (
            (local_n & 7) * 32
            + (local_n // 8) * 2
            + ((local_k & 7) // 2) * 8
            + (local_k & 1)
            + (local_k // 8) * 4
        )
        pair = q // 2
        component = q & 1
        selected_bit_position = pair * (RATE << 1)
        word_index = selected_bit_position // 16
        shift = selected_bit_position & 15
        block_m = n[:, None] // 16
        block_k = k[None, :] // 16
        parity = ((block_m & 1) * 2 + (block_k & 1)).to(tl.int64)
        superblock = ((block_m // 2) * (K // 32) + (block_k // 2)).to(tl.int64)
        superblock_bytes = 32 * 2 * 2 * RATE
        map_base = (parity * (RATE * 16) + word_index) * 2
        curr_lo_off = tl.load(offset_map + map_base).to(tl.int64)
        curr_hi_off = tl.load(offset_map + map_base + 1).to(tl.int64)
        next_word = (word_index + 1) % (RATE * 16)
        next_base = (parity * (RATE * 16) + next_word) * 2
        next_lo_off = tl.load(offset_map + next_base).to(tl.int64)
        next_hi_off = tl.load(offset_map + next_base + 1).to(tl.int64)
        mask = n_mask[:, None] & k_mask[None, :]
        source_base = superblock * superblock_bytes
        curr = (
            tl.load(source_ptr + source_base + curr_lo_off, mask=mask, other=0).to(tl.int64)
            | (tl.load(source_ptr + source_base + curr_hi_off, mask=mask, other=0).to(tl.int64) << 8)
        )
        nxt = (
            tl.load(source_ptr + source_base + next_lo_off, mask=mask, other=0).to(tl.int64)
            | (tl.load(source_ptr + source_base + next_hi_off, mask=mask, other=0).to(tl.int64) << 8)
        )
        word = (curr << 16) | nxt
        index = (word >> (16 - shift)) & 65535
        weight = tl.load(lut_ptr + index * 2 + component).to(tl.float32)
        acc += tl.sum(weight * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


def qtip_offset_map(rate: int, device: str = "cuda") -> torch.Tensor:
    total = 32 * 2 * 2 * rate
    indices = torch.arange(total, dtype=torch.int64).reshape(1, 1, 32, 2, 2, rate)
    mapped = (
        indices.permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(2, 2, rate * 16, 2)
        .flip((-1,))
        .contiguous()
    )
    return mapped.to(device=device, dtype=torch.int32)


def qtip_raw_gemv_source(
    x: torch.Tensor,
    sources: list[torch.Tensor],
    rate: int,
    lut: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    if len(sources) != r:
        raise ValueError(f"source count {len(sources)} != rows {r}")
    n = 4096
    holders = pointer_holder(*sources)
    offsets = qtip_offset_map(rate) if offsets is None else offsets
    y = torch.empty((r, n), dtype=torch.float32, device=x.device)
    _qtip_raw_gemv_source[(triton.cdiv(n, 8), r)](
        x,
        holders,
        offsets,
        lut,
        y,
        R=r,
        N=n,
        K=k,
        RATE=rate,
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


def d4_gemv_batch(
    x: torch.Tensor,
    codes: list[torch.Tensor],
    scales: list[torch.Tensor],
    codebooks: list[torch.Tensor],
) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    if not (len(codes) == len(scales) == len(codebooks) == r):
        raise ValueError("D4 batch state count mismatch")
    n = codes[0].shape[0]
    code_holders = pointer_holder(*codes)
    scale_holders = pointer_holder(*scales)
    cb_holders = pointer_holder(*codebooks)
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    _d4_gemv[(triton.cdiv(n, 8), r)](
        x,
        code_holders,
        scale_holders,
        cb_holders,
        y,
        R=r,
        N=n,
        K=k,
        D=codebooks[0].shape[-1],
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


def native_mxfp4_gemv_batch(
    x: torch.Tensor,
    packed: list[torch.Tensor],
    scales: list[torch.Tensor],
) -> torch.Tensor:
    x = x.to(torch.bfloat16).contiguous()
    r, k = x.shape
    if not (len(packed) == len(scales) == r):
        raise ValueError("native MXFP4 batch state count mismatch")
    n = scales[0].shape[0]
    packed_holders = pointer_holder(*packed)
    scale_holders = pointer_holder(*scales)
    y = torch.empty((r, n), dtype=torch.bfloat16, device=x.device)
    _native_mxfp4_gemv[(triton.cdiv(n, 8), r)](
        x,
        packed_holders,
        scale_holders,
        y,
        R=r,
        N=n,
        K=k,
        BN=8,
        BK=256,
        num_warps=4,
        num_stages=2,
    )
    return y


# Graph-safe exact-P975 dispatch.  Unlike the reference path above, these
# kernels never copy routed expert ids to the CPU.  Every row indexes immutable
# per-expert pointer tables on device, so changing top-k ids remains correct
# under per-layer CUDA-graph replay.
@triton.jit
def _qtip_raw_gemv_dynamic(
    x_ptr,
    expert_ids_ptr,
    family_ptr,
    source_ptrs,
    offset_map,
    lut_ptr,
    y_ptr,
    EXPECTED_FAMILY: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    RATE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    expert = tl.load(expert_ids_ptr + r).to(tl.int64)
    active = tl.load(family_ptr + expert) == EXPECTED_FAMILY
    source_addr = tl.load(source_ptrs + expert).to(tl.int64)
    source_ptr = tl.cast(source_addr, tl.pointer_type(tl.uint8))
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < K
        mask = active & n_mask[:, None] & k_mask[None, :]
        x = tl.load(x_ptr + r * K + k, mask=active & k_mask, other=0.0).to(tl.float32)
        local_n = n[:, None] & 15
        local_k = k[None, :] & 15
        q = ((local_n & 7) * 32 + (local_n // 8) * 2
             + ((local_k & 7) // 2) * 8 + (local_k & 1)
             + (local_k // 8) * 4)
        pair = q // 2
        component = q & 1
        selected_bit_position = pair * (RATE << 1)
        word_index = selected_bit_position // 16
        shift = selected_bit_position & 15
        block_m = n[:, None] // 16
        block_k = k[None, :] // 16
        parity = ((block_m & 1) * 2 + (block_k & 1)).to(tl.int64)
        superblock = ((block_m // 2) * (K // 32) + (block_k // 2)).to(tl.int64)
        superblock_bytes = 32 * 2 * 2 * RATE
        map_base = (parity * (RATE * 16) + word_index) * 2
        curr_lo_off = tl.load(offset_map + map_base).to(tl.int64)
        curr_hi_off = tl.load(offset_map + map_base + 1).to(tl.int64)
        next_word = (word_index + 1) % (RATE * 16)
        next_base = (parity * (RATE * 16) + next_word) * 2
        next_lo_off = tl.load(offset_map + next_base).to(tl.int64)
        next_hi_off = tl.load(offset_map + next_base + 1).to(tl.int64)
        source_base = superblock * superblock_bytes
        curr = (tl.load(source_ptr + source_base + curr_lo_off, mask=mask, other=0).to(tl.int64)
                | (tl.load(source_ptr + source_base + curr_hi_off, mask=mask, other=0).to(tl.int64) << 8))
        nxt = (tl.load(source_ptr + source_base + next_lo_off, mask=mask, other=0).to(tl.int64)
               | (tl.load(source_ptr + source_base + next_hi_off, mask=mask, other=0).to(tl.int64) << 8))
        word = (curr << 16) | nxt
        index = (word >> (16 - shift)) & 65535
        weight = tl.load(lut_ptr + index * 2 + component, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(weight * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _d4_gemv_dynamic(
    x_ptr,
    expert_ids_ptr,
    family_ptr,
    codes_ptrs,
    scales_ptrs,
    codebook_ptrs,
    y_ptr,
    EXPECTED_FAMILY: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    expert = tl.load(expert_ids_ptr + r).to(tl.int64)
    active = tl.load(family_ptr + expert) == EXPECTED_FAMILY
    codes_ptr = tl.cast(tl.load(codes_ptrs + expert).to(tl.int64), tl.pointer_type(tl.int16))
    scales_ptr = tl.cast(tl.load(scales_ptrs + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    cb_ptr = tl.cast(tl.load(codebook_ptrs + expert).to(tl.int64), tl.pointer_type(tl.float16))
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        mask = active & n_mask[:, None] & (k[None, :] < K)
        x = tl.load(x_ptr + r * K + k, mask=active & (k < K), other=0.0).to(tl.float32)
        code = tl.load(codes_ptr + n[:, None] * (K // D) + (k[None, :] // D), mask=mask, other=0).to(tl.int32)
        value = tl.load(cb_ptr + code * D + (k[None, :] % D), mask=mask, other=0.0)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32), mask=mask, other=127).to(tl.float32)
        acc += tl.sum(value.to(tl.float32) * tl.exp2(scale - 127.0) * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


@triton.jit
def _native_mxfp4_gemv_dynamic(
    x_ptr,
    expert_ids_ptr,
    family_ptr,
    packed_ptrs,
    scales_ptrs,
    y_ptr,
    EXPECTED_FAMILY: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    expert = tl.load(expert_ids_ptr + r).to(tl.int64)
    active = tl.load(family_ptr + expert) == EXPECTED_FAMILY
    packed_ptr = tl.cast(tl.load(packed_ptrs + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    scales_ptr = tl.cast(tl.load(scales_ptrs + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        mask = active & n_mask[:, None] & (k[None, :] < K)
        x = tl.load(x_ptr + r * K + k, mask=active & (k < K), other=0.0).to(tl.float32)
        byte = tl.load(packed_ptr + n[:, None] * (K // 2) + (k[None, :] // 2), mask=mask, other=0).to(tl.int32)
        code = tl.where((k[None, :] & 1) == 0, byte & 15, byte >> 4)
        scale = tl.load(scales_ptr + n[:, None] * (K // 32) + (k[None, :] // 32), mask=mask, other=127).to(tl.float32)
        acc += tl.sum(_e2m1(code) * tl.exp2(scale - 127.0) * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


def qtip_raw_gemv_dynamic(x, expert_ids, family, source_ptrs, rate, lut, offsets):
    x = x.to(torch.bfloat16).contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.long).reshape(-1)
    r, k = x.shape
    y = torch.empty((r, 4096), dtype=torch.float32, device=x.device)
    expected = 0 if rate == 2 else 1
    _qtip_raw_gemv_dynamic[(triton.cdiv(4096, 8), r)](
        x, expert_ids, family, source_ptrs, offsets, lut, y,
        EXPECTED_FAMILY=expected, R=r, N=4096, K=k, RATE=rate,
        BN=8, BK=256, num_warps=4, num_stages=2)
    return y


def d4_gemv_dynamic(x, expert_ids, family, codes_ptrs, scales_ptrs, codebook_ptrs):
    x = x.to(torch.bfloat16).contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.long).reshape(-1)
    r, k = x.shape
    y = torch.empty((r, 4096), dtype=torch.bfloat16, device=x.device)
    _d4_gemv_dynamic[(triton.cdiv(4096, 8), r)](
        x, expert_ids, family, codes_ptrs, scales_ptrs, codebook_ptrs, y,
        EXPECTED_FAMILY=2, R=r, N=4096, K=k, D=4,
        BN=8, BK=256, num_warps=4, num_stages=2)
    return y


def native_mxfp4_gemv_dynamic(x, expert_ids, family, packed_ptrs, scales_ptrs):
    x = x.to(torch.bfloat16).contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.long).reshape(-1)
    r, k = x.shape
    y = torch.empty((r, 4096), dtype=torch.bfloat16, device=x.device)
    _native_mxfp4_gemv_dynamic[(triton.cdiv(4096, 8), r)](
        x, expert_ids, family, packed_ptrs, scales_ptrs, y,
        EXPECTED_FAMILY=3, R=r, N=4096, K=k,
        BN=8, BK=256, num_warps=4, num_stages=2)
    return y


@triton.jit
def _mixed_exact_gemv(
    x_ptr,
    expert_ids,
    family_codes,
    qtip_sources,
    d4_codes,
    d4_index_bits,
    d4_scales,
    d4_codebooks,
    native_packed,
    native_scales,
    qtip_offsets2,
    qtip_offsets3,
    lut_ptr,
    y_ptr,
    R: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """One exact expert-specific packed GEMV launch for all P975 families."""
    pid_n = tl.program_id(0)
    r = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_mask = n < N
    expert = tl.load(expert_ids + r).to(tl.int64)
    family = tl.load(family_codes + expert).to(tl.int32)
    is_q2 = family == 0
    is_q3 = family == 1
    is_qtip = is_q2 | is_q3
    is_d4 = family == 2
    is_native = family == 3

    qtip_ptr = tl.cast(tl.load(qtip_sources + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    d4_code_ptr = tl.cast(tl.load(d4_codes + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    d4_bits = tl.load(d4_index_bits + expert).to(tl.int64)
    d4_scale_ptr = tl.cast(tl.load(d4_scales + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    d4_cb_ptr = tl.cast(tl.load(d4_codebooks + expert).to(tl.int64), tl.pointer_type(tl.float16))
    native_packed_ptr = tl.cast(tl.load(native_packed + expert).to(tl.int64), tl.pointer_type(tl.uint8))
    native_scale_ptr = tl.cast(tl.load(native_scales + expert).to(tl.int64), tl.pointer_type(tl.uint8))

    rate = tl.where(is_q2, 2, 3).to(tl.int64)
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        k = k0 + tl.arange(0, BK)
        k_mask = k < K
        common_mask = n_mask[:, None] & k_mask[None, :]
        x = tl.load(x_ptr + r * K + k, mask=k_mask, other=0.0).to(tl.float32)

        local_n = n[:, None] & 15
        local_k = k[None, :] & 15
        q = ((local_n & 7) * 32 + (local_n // 8) * 2
             + ((local_k & 7) // 2) * 8 + (local_k & 1)
             + (local_k // 8) * 4)
        pair = q // 2
        component = q & 1
        selected_bit_position = pair * (rate << 1)
        word_index = selected_bit_position // 16
        shift = selected_bit_position & 15
        block_m = n[:, None] // 16
        block_k = k[None, :] // 16
        parity = ((block_m & 1) * 2 + (block_k & 1)).to(tl.int64)
        superblock = ((block_m // 2) * (K // 32) + (block_k // 2)).to(tl.int64)
        map_base2 = (parity * 32 + word_index) * 2
        map_base3 = (parity * 48 + word_index) * 2
        next_word = (word_index + 1) % (rate * 16)
        next_base2 = (parity * 32 + next_word) * 2
        next_base3 = (parity * 48 + next_word) * 2
        curr_lo = tl.where(is_q2, tl.load(qtip_offsets2 + map_base2), tl.load(qtip_offsets3 + map_base3)).to(tl.int64)
        curr_hi = tl.where(is_q2, tl.load(qtip_offsets2 + map_base2 + 1), tl.load(qtip_offsets3 + map_base3 + 1)).to(tl.int64)
        next_lo = tl.where(is_q2, tl.load(qtip_offsets2 + next_base2), tl.load(qtip_offsets3 + next_base3)).to(tl.int64)
        next_hi = tl.where(is_q2, tl.load(qtip_offsets2 + next_base2 + 1), tl.load(qtip_offsets3 + next_base3 + 1)).to(tl.int64)
        qmask = common_mask & is_qtip
        source_base = superblock * (128 * rate)
        curr = (tl.load(qtip_ptr + source_base + curr_lo, mask=qmask, other=0).to(tl.int64)
                | (tl.load(qtip_ptr + source_base + curr_hi, mask=qmask, other=0).to(tl.int64) << 8))
        nxt = (tl.load(qtip_ptr + source_base + next_lo, mask=qmask, other=0).to(tl.int64)
               | (tl.load(qtip_ptr + source_base + next_hi, mask=qmask, other=0).to(tl.int64) << 8))
        word = (curr << 16) | nxt
        index = (word >> (16 - shift)) & 65535
        qtip_weight = tl.load(lut_ptr + index * 2 + component, mask=qmask, other=0.0).to(tl.float32)

        d4mask = common_mask & is_d4
        d4_row_bytes = ((K // 4) * d4_bits + 7) // 8
        d4_bit = (k[None, :] // 4) * d4_bits
        d4_byte = d4_bit // 8
        d4_shift = d4_bit & 7
        d4_base = n[:, None] * d4_row_bytes + d4_byte
        d4_word = (
            tl.load(d4_code_ptr + d4_base, mask=d4mask, other=0).to(tl.int32)
            | (
                tl.load(
                    d4_code_ptr + d4_base + 1,
                    mask=d4mask & (d4_byte + 1 < d4_row_bytes),
                    other=0,
                ).to(tl.int32)
                << 8
            )
            | (
                tl.load(
                    d4_code_ptr + d4_base + 2,
                    mask=d4mask & (d4_byte + 2 < d4_row_bytes),
                    other=0,
                ).to(tl.int32)
                << 16
            )
        )
        d4_code = (d4_word >> d4_shift) & ((1 << d4_bits) - 1)
        d4_value = tl.load(d4_cb_ptr + d4_code * 4 + (k[None, :] & 3), mask=d4mask, other=0.0).to(tl.float32)
        d4_scale = tl.load(d4_scale_ptr + n[:, None] * (K // 32) + (k[None, :] // 32), mask=d4mask, other=127).to(tl.float32)
        d4_weight = d4_value * tl.exp2(d4_scale - 127.0)

        native_mask = common_mask & is_native
        packed = tl.load(native_packed_ptr + n[:, None] * (K // 2) + (k[None, :] // 2), mask=native_mask, other=0).to(tl.int32)
        native_code = tl.where((k[None, :] & 1) == 0, packed & 15, packed >> 4)
        native_scale = tl.load(native_scale_ptr + n[:, None] * (K // 32) + (k[None, :] // 32), mask=native_mask, other=127).to(tl.float32)
        native_weight = _e2m1(native_code) * tl.exp2(native_scale - 127.0)

        weight = tl.where(is_qtip, qtip_weight, tl.where(is_d4, d4_weight, native_weight))
        acc += tl.sum(weight * x[None, :], axis=1)
    tl.store(y_ptr + r * N + n, acc, mask=n_mask)


def mixed_exact_gemv(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    family_codes: torch.Tensor,
    pointer_tables: dict[str, torch.Tensor],
    offsets2: torch.Tensor,
    offsets3: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    x = x.float().contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int64).contiguous()
    r, k = x.shape
    y = torch.empty((r, 4096), dtype=torch.float32, device=x.device)
    _mixed_exact_gemv[(triton.cdiv(4096, 8), r)](
        x, expert_ids, family_codes,
        pointer_tables["qtip_sources"],
        pointer_tables["d4_codes"], pointer_tables["d4_index_bits"],
        pointer_tables["d4_scales"],
        pointer_tables["d4_codebooks"], pointer_tables["native_packed"],
        pointer_tables["native_scales"], offsets2, offsets3, lut, y,
        R=r, N=4096, K=k, BN=8, BK=256,
        num_warps=4, num_stages=2,
    )
    return y
