# SPDX-License-Identifier: Apache-2.0
"""Direct d4/d8 VQ MoE GEMM for the IQ3 campaign wire format.

The kernel reconstructs each block-32-scaled codebook vector in registers and
feeds BF16 tensor-core dot products. Codes, scales, and codebooks may remain in
pageable mmap-backed host memory on coherent GB10 systems; their raw UVA/ATS
pointers are passed as integers so Triton's launcher does not reject CPU tensor
objects. Small routing/offset arrays live on CUDA.
"""
from __future__ import annotations

import functools
import os
from typing import Any

import numpy as np
import torch


_ARRAY_KEYS = (
    "codes",
    "scales",
    "codebooks",
    "code_offset",
    "scale_offset",
    "code_row_bytes",
    "dimension",
    "bits",
    "cb_offset",
)
_META_KEYS = (
    "code_offset",
    "scale_offset",
    "code_row_bytes",
    "dimension",
    "bits",
    "cb_offset",
)


def validate_projection_state(state: dict[str, Any], experts: int, width: int) -> None:
    """Fail closed on malformed offsets before a GPU kernel sees the pack."""
    if width % 32:
        raise ValueError("width must be a multiple of 32")
    missing = [key for key in _ARRAY_KEYS if key not in state]
    if missing:
        raise ValueError(f"projection state missing keys: {missing}")
    n_outputs = int(state.get("n_outputs", 0))
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive")
    for key in _META_KEYS:
        if len(state[key]) != experts:
            raise ValueError(f"{key} has {len(state[key])} rows; expected {experts}")

    codes_size = int(np.asarray(state["codes"]).size)
    scales_size = int(np.asarray(state["scales"]).size)
    codebooks_size = int(np.asarray(state["codebooks"]).size)
    for expert in range(experts):
        dimension = int(state["dimension"][expert])
        bits = int(state["bits"][expert])
        if dimension not in (4, 8) or bits not in (8, 9, 10, 11, 12):
            raise ValueError(f"expert {expert} unsupported d/bits: {dimension}/{bits}")
        row_bytes = int(state["code_row_bytes"][expert])
        expected_row_bytes = ((width // dimension) * bits + 7) // 8
        if row_bytes != expected_row_bytes:
            raise ValueError(
                f"expert {expert} packed row bytes {row_bytes}; "
                f"expected {expected_row_bytes}"
            )
        code_begin = int(state["code_offset"][expert])
        code_end = code_begin + n_outputs * row_bytes
        if code_begin < 0 or code_end > codes_size:
            raise ValueError(f"expert {expert} code range [{code_begin}, {code_end}) invalid")
        scale_begin = int(state["scale_offset"][expert])
        scale_end = scale_begin + n_outputs * (width // 32)
        if scale_begin < 0 or scale_end > scales_size:
            raise ValueError(f"expert {expert} scale range [{scale_begin}, {scale_end}) invalid")
        cb_begin = int(state["cb_offset"][expert])
        cb_end = cb_begin + (1 << bits) * dimension
        if cb_begin < 0 or cb_end > codebooks_size:
            raise ValueError(f"expert {expert} codebook range [{cb_begin}, {cb_end}) invalid")


def projection_to_torch(
    state: dict[str, Any], device: torch.device, *, blobs_on_device: bool = True
) -> dict[str, Any]:
    """Convert a NumPy projection pack to stable tensors for serving/tests."""
    experts = len(state["code_offset"])
    width = int(state["scales"].size // (experts * int(state["n_outputs"])) * 32)
    validate_projection_state(state, experts, width)
    result: dict[str, Any] = {
        "n_outputs": int(state["n_outputs"]),
        "all_d4": bool(np.all(np.asarray(state["dimension"]) == 4)),
    }
    for key in _META_KEYS:
        result[key] = torch.as_tensor(np.asarray(state[key]), device=device)
    blob_device = device if blobs_on_device else torch.device("cpu")
    for key in ("codes", "scales", "codebooks"):
        result[key] = torch.as_tensor(np.asarray(state[key]), device=blob_device)
    result["blob_ptrs"] = torch.tensor(
        [result[key].data_ptr() for key in ("codes", "scales", "codebooks")],
        dtype=torch.int64,
        device=device,
    )
    return result


@functools.cache
def _compiled_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        a_ptr,
        out_ptr,
        expert_blocks_ptr,
        num_post_ptr,
        kind_ptr,
        code_offset_ptr,
        scale_offset_ptr,
        code_row_bytes_ptr,
        dimension_ptr,
        bits_ptr,
        cb_offset_ptr,
        blob_ptrs_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pair = tl.program_id(0)
        n_block = tl.program_id(1)
        expert = tl.load(expert_blocks_ptr + pair).to(tl.int64)
        live = pair < (tl.load(num_post_ptr).to(tl.int64) // MBLOCK)
        is_vq = tl.load(kind_ptr + expert).to(tl.int32) == 0
        active = live & is_vq

        code_base = tl.load(code_offset_ptr + expert).to(tl.int64)
        scale_base = tl.load(scale_offset_ptr + expert).to(tl.int64)
        row_bytes = tl.load(code_row_bytes_ptr + expert).to(tl.int64)
        dimension = tl.load(dimension_ptr + expert).to(tl.int32)
        index_width = tl.load(bits_ptr + expert).to(tl.int32)
        cb_base = tl.load(cb_offset_ptr + expert).to(tl.int64)
        codes_ptr = tl.cast(tl.load(blob_ptrs_ptr), tl.pointer_type(tl.uint8))
        scales_ptr = tl.cast(tl.load(blob_ptrs_ptr + 1), tl.pointer_type(tl.uint8))
        codebooks_ptr = tl.cast(
            tl.load(blob_ptrs_ptr + 2), tl.pointer_type(tl.float16)
        )

        rows_i = tl.arange(0, 16)
        rows = pair * MBLOCK + rows_i
        row_mask = rows_i < MBLOCK
        cols = n_block * BN + tl.arange(0, BN)
        col_mask = cols < N
        acc = tl.zeros((16, BN), dtype=tl.float32)
        index_mask = (1 << index_width) - 1

        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            a = tl.load(
                a_ptr + rows[:, None] * K + ks[None, :],
                mask=active & row_mask[:, None],
                other=0.0,
            )
            groups = ks // dimension
            components = ks - groups * dimension
            bit_positions = groups * index_width
            byte_positions = bit_positions // 8
            shifts = bit_positions - byte_positions * 8
            addresses = (
                code_base
                + cols[:, None].to(tl.int64) * row_bytes
                + byte_positions[None, :].to(tl.int64)
            )
            load_mask = active & col_mask[:, None]
            b0 = tl.load(codes_ptr + addresses, mask=load_mask, other=0).to(tl.int32)
            b1 = tl.load(
                codes_ptr + addresses + 1,
                mask=load_mask & (byte_positions[None, :] + 1 < row_bytes),
                other=0,
            ).to(tl.int32)
            b2 = tl.load(
                codes_ptr + addresses + 2,
                mask=load_mask & (byte_positions[None, :] + 2 < row_bytes),
                other=0,
            ).to(tl.int32)
            words = b0 | (b1 << 8) | (b2 << 16)
            indices = (words >> shifts[None, :]) & index_mask
            cb_addresses = (
                cb_base
                + indices.to(tl.int64) * dimension
                + components[None, :].to(tl.int64)
            )
            cb = tl.load(codebooks_ptr + cb_addresses, mask=load_mask, other=0.0)
            scale_addresses = (
                scale_base
                + cols[:, None].to(tl.int64) * (K // 32)
                + (ks // 32)[None, :].to(tl.int64)
            )
            scale_byte = tl.load(
                scales_ptr + scale_addresses, mask=load_mask, other=127
            ).to(tl.float32)
            weights = (cb.to(tl.float32) * tl.exp2(scale_byte - 127.0)).to(tl.bfloat16)
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(weights))

        tl.store(
            out_ptr + rows[:, None] * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=active & row_mask[:, None] & col_mask[None, :],
        )

    return triton, kernel


@functools.cache
def _compiled_kernel_d4():
    """Dimension-4 specialization for the shipping IQ3 arm4 artifact.

    The generic kernel loads each packed VQ index and block scale once per
    reconstructed scalar.  A d4 index represents four adjacent scalars, so
    that path performs four redundant packed-index loads and 32 redundant
    scale loads per output block.  This specialization loads eight indices
    and one scale per 32-wide K tile, expands each index through its exact
    four-value FP16 codebook row in registers, then feeds the same BF16 MMA.
    Quantized codes/scales/codebooks and accumulation order are unchanged.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(
        a_ptr,
        out_ptr,
        expert_blocks_ptr,
        num_post_ptr,
        kind_ptr,
        code_offset_ptr,
        scale_offset_ptr,
        code_row_bytes_ptr,
        dimension_ptr,
        bits_ptr,
        cb_offset_ptr,
        blob_ptrs_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pair = tl.program_id(0)
        n_block = tl.program_id(1)
        expert = tl.load(expert_blocks_ptr + pair).to(tl.int64)
        live = pair < (tl.load(num_post_ptr).to(tl.int64) // MBLOCK)
        is_vq = tl.load(kind_ptr + expert).to(tl.int32) == 0
        active = live & is_vq

        code_base = tl.load(code_offset_ptr + expert).to(tl.int64)
        scale_base = tl.load(scale_offset_ptr + expert).to(tl.int64)
        row_bytes = tl.load(code_row_bytes_ptr + expert).to(tl.int64)
        index_width = tl.load(bits_ptr + expert).to(tl.int32)
        cb_base = tl.load(cb_offset_ptr + expert).to(tl.int64)
        codes_ptr = tl.cast(tl.load(blob_ptrs_ptr), tl.pointer_type(tl.uint8))
        scales_ptr = tl.cast(tl.load(blob_ptrs_ptr + 1), tl.pointer_type(tl.uint8))
        codebooks_ptr = tl.cast(
            tl.load(blob_ptrs_ptr + 2), tl.pointer_type(tl.float16)
        )

        rows_i = tl.arange(0, 16)
        rows = pair * MBLOCK + rows_i
        row_mask = rows_i < MBLOCK
        cols = n_block * BN + tl.arange(0, BN)
        col_mask = cols < N
        acc = tl.zeros((16, BN), dtype=tl.float32)
        index_mask = (1 << index_width) - 1
        groups_i = tl.arange(0, BK // 4)
        components = tl.arange(0, 4)

        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            a = tl.load(
                a_ptr + rows[:, None] * K + ks[None, :],
                mask=active & row_mask[:, None],
                other=0.0,
            )

            groups = (k0 // 4) + groups_i
            bit_positions = groups * index_width
            byte_positions = bit_positions // 8
            shifts = bit_positions - byte_positions * 8
            addresses = (
                code_base
                + cols[:, None].to(tl.int64) * row_bytes
                + byte_positions[None, :].to(tl.int64)
            )
            load_mask = active & col_mask[:, None]
            b0 = tl.load(codes_ptr + addresses, mask=load_mask, other=0).to(tl.int32)
            b1 = tl.load(
                codes_ptr + addresses + 1,
                mask=load_mask & (byte_positions[None, :] + 1 < row_bytes),
                other=0,
            ).to(tl.int32)
            b2 = tl.load(
                codes_ptr + addresses + 2,
                mask=load_mask & (byte_positions[None, :] + 2 < row_bytes),
                other=0,
            ).to(tl.int32)
            words = b0 | (b1 << 8) | (b2 << 16)
            indices = (words >> shifts[None, :]) & index_mask
            cb_addresses = (
                cb_base
                + indices[:, :, None].to(tl.int64) * 4
                + components[None, None, :].to(tl.int64)
            )
            cb = tl.load(
                codebooks_ptr + cb_addresses,
                mask=load_mask[:, :, None],
                other=0.0,
            )
            scale_addresses = (
                scale_base
                + cols.to(tl.int64) * (K // 32)
                + (k0 // 32)
            )
            scale_byte = tl.load(
                scales_ptr + scale_addresses, mask=active & col_mask, other=127
            ).to(tl.float32)
            weights = (
                cb.to(tl.float32)
                * tl.exp2(scale_byte[:, None, None] - 127.0)
            ).reshape((BN, BK)).to(tl.bfloat16)
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(weights))

        tl.store(
            out_ptr + rows[:, None] * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=active & row_mask[:, None] & col_mask[None, :],
        )

    return triton, kernel


@functools.cache
def _compiled_kernel_grouped():
    """Runtime d4/d8 specialization for heterogeneous shipping projections.

    Most arm4 projections contain only a handful of d8 experts, which made the
    projection-wide ``all_d4`` gate route every d4 expert through the scalar
    generic decoder.  This kernel keeps one launch and selects an exact grouped
    index expansion per routed expert: 8 indices per K=32 tile for d4, 4 for d8.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def kernel_branchy(
        a_ptr,
        out_ptr,
        expert_blocks_ptr,
        num_post_ptr,
        kind_ptr,
        code_offset_ptr,
        scale_offset_ptr,
        code_row_bytes_ptr,
        dimension_ptr,
        bits_ptr,
        cb_offset_ptr,
        blob_ptrs_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pair = tl.program_id(0)
        n_block = tl.program_id(1)
        expert = tl.load(expert_blocks_ptr + pair).to(tl.int64)
        live = pair < (tl.load(num_post_ptr).to(tl.int64) // MBLOCK)
        is_vq = tl.load(kind_ptr + expert).to(tl.int32) == 0
        active = live & is_vq

        code_base = tl.load(code_offset_ptr + expert).to(tl.int64)
        scale_base = tl.load(scale_offset_ptr + expert).to(tl.int64)
        row_bytes = tl.load(code_row_bytes_ptr + expert).to(tl.int64)
        dimension = tl.load(dimension_ptr + expert).to(tl.int32)
        index_width = tl.load(bits_ptr + expert).to(tl.int32)
        cb_base = tl.load(cb_offset_ptr + expert).to(tl.int64)
        codes_ptr = tl.cast(tl.load(blob_ptrs_ptr), tl.pointer_type(tl.uint8))
        scales_ptr = tl.cast(tl.load(blob_ptrs_ptr + 1), tl.pointer_type(tl.uint8))
        codebooks_ptr = tl.cast(
            tl.load(blob_ptrs_ptr + 2), tl.pointer_type(tl.float16)
        )

        rows_i = tl.arange(0, 16)
        rows = pair * MBLOCK + rows_i
        row_mask = rows_i < MBLOCK
        cols = n_block * BN + tl.arange(0, BN)
        col_mask = cols < N
        index_mask = (1 << index_width) - 1

        if dimension == 4:
            acc = tl.zeros((16, BN), dtype=tl.float32)
            groups_i = tl.arange(0, BK // 4)
            components = tl.arange(0, 4)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                a = tl.load(
                    a_ptr + rows[:, None] * K + ks[None, :],
                    mask=active & row_mask[:, None],
                    other=0.0,
                )
                groups = (k0 // 4) + groups_i
                bit_positions = groups * index_width
                byte_positions = bit_positions // 8
                shifts = bit_positions - byte_positions * 8
                addresses = (
                    code_base
                    + cols[:, None].to(tl.int64) * row_bytes
                    + byte_positions[None, :].to(tl.int64)
                )
                load_mask = active & col_mask[:, None]
                b0 = tl.load(
                    codes_ptr + addresses, mask=load_mask, other=0
                ).to(tl.int32)
                b1 = tl.load(
                    codes_ptr + addresses + 1,
                    mask=load_mask & (byte_positions[None, :] + 1 < row_bytes),
                    other=0,
                ).to(tl.int32)
                b2 = tl.load(
                    codes_ptr + addresses + 2,
                    mask=load_mask & (byte_positions[None, :] + 2 < row_bytes),
                    other=0,
                ).to(tl.int32)
                words = b0 | (b1 << 8) | (b2 << 16)
                indices = (words >> shifts[None, :]) & index_mask
                cb_addresses = (
                    cb_base
                    + indices[:, :, None].to(tl.int64) * 4
                    + components[None, None, :].to(tl.int64)
                )
                cb = tl.load(
                    codebooks_ptr + cb_addresses,
                    mask=load_mask[:, :, None],
                    other=0.0,
                )
                scale_addresses = (
                    scale_base
                    + cols.to(tl.int64) * (K // 32)
                    + (k0 // 32)
                )
                scale_byte = tl.load(
                    scales_ptr + scale_addresses,
                    mask=active & col_mask,
                    other=127,
                ).to(tl.float32)
                weights = (
                    cb.to(tl.float32)
                    * tl.exp2(scale_byte[:, None, None] - 127.0)
                ).reshape((BN, BK)).to(tl.bfloat16)
                acc += tl.dot(a.to(tl.bfloat16), tl.trans(weights))
            tl.store(
                out_ptr + rows[:, None] * N + cols[None, :],
                acc.to(tl.bfloat16),
                mask=active & row_mask[:, None] & col_mask[None, :],
            )
        else:
            acc = tl.zeros((16, BN), dtype=tl.float32)
            groups_i = tl.arange(0, BK // 8)
            components = tl.arange(0, 8)
            for k0 in range(0, K, BK):
                ks = k0 + tl.arange(0, BK)
                a = tl.load(
                    a_ptr + rows[:, None] * K + ks[None, :],
                    mask=active & row_mask[:, None],
                    other=0.0,
                )
                groups = (k0 // 8) + groups_i
                bit_positions = groups * index_width
                byte_positions = bit_positions // 8
                shifts = bit_positions - byte_positions * 8
                addresses = (
                    code_base
                    + cols[:, None].to(tl.int64) * row_bytes
                    + byte_positions[None, :].to(tl.int64)
                )
                load_mask = active & col_mask[:, None]
                b0 = tl.load(
                    codes_ptr + addresses, mask=load_mask, other=0
                ).to(tl.int32)
                b1 = tl.load(
                    codes_ptr + addresses + 1,
                    mask=load_mask & (byte_positions[None, :] + 1 < row_bytes),
                    other=0,
                ).to(tl.int32)
                b2 = tl.load(
                    codes_ptr + addresses + 2,
                    mask=load_mask & (byte_positions[None, :] + 2 < row_bytes),
                    other=0,
                ).to(tl.int32)
                words = b0 | (b1 << 8) | (b2 << 16)
                indices = (words >> shifts[None, :]) & index_mask
                cb_addresses = (
                    cb_base
                    + indices[:, :, None].to(tl.int64) * 8
                    + components[None, None, :].to(tl.int64)
                )
                cb = tl.load(
                    codebooks_ptr + cb_addresses,
                    mask=load_mask[:, :, None],
                    other=0.0,
                )
                scale_addresses = (
                    scale_base
                    + cols.to(tl.int64) * (K // 32)
                    + (k0 // 32)
                )
                scale_byte = tl.load(
                    scales_ptr + scale_addresses,
                    mask=active & col_mask,
                    other=127,
                ).to(tl.float32)
                weights = (
                    cb.to(tl.float32)
                    * tl.exp2(scale_byte[:, None, None] - 127.0)
                ).reshape((BN, BK)).to(tl.bfloat16)
                acc += tl.dot(a.to(tl.bfloat16), tl.trans(weights))
            tl.store(
                out_ptr + rows[:, None] * N + cols[None, :],
                acc.to(tl.bfloat16),
                mask=active & row_mask[:, None] & col_mask[None, :],
            )

    @triton.jit
    def kernel(
        a_ptr,
        out_ptr,
        expert_blocks_ptr,
        num_post_ptr,
        kind_ptr,
        code_offset_ptr,
        scale_offset_ptr,
        code_row_bytes_ptr,
        dimension_ptr,
        bits_ptr,
        cb_offset_ptr,
        blob_ptrs_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
        MBLOCK: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pair = tl.program_id(0)
        n_block = tl.program_id(1)
        expert = tl.load(expert_blocks_ptr + pair).to(tl.int64)
        live = pair < (tl.load(num_post_ptr).to(tl.int64) // MBLOCK)
        is_vq = tl.load(kind_ptr + expert).to(tl.int32) == 0
        active = live & is_vq

        code_base = tl.load(code_offset_ptr + expert).to(tl.int64)
        scale_base = tl.load(scale_offset_ptr + expert).to(tl.int64)
        row_bytes = tl.load(code_row_bytes_ptr + expert).to(tl.int64)
        dimension = tl.load(dimension_ptr + expert).to(tl.int32)
        index_width = tl.load(bits_ptr + expert).to(tl.int32)
        cb_base = tl.load(cb_offset_ptr + expert).to(tl.int64)
        codes_ptr = tl.cast(tl.load(blob_ptrs_ptr), tl.pointer_type(tl.uint8))
        scales_ptr = tl.cast(tl.load(blob_ptrs_ptr + 1), tl.pointer_type(tl.uint8))
        codebooks_ptr = tl.cast(
            tl.load(blob_ptrs_ptr + 2), tl.pointer_type(tl.float16)
        )

        rows_i = tl.arange(0, 16)
        rows = pair * MBLOCK + rows_i
        row_mask = rows_i < MBLOCK
        cols = n_block * BN + tl.arange(0, BN)
        col_mask = cols < N
        acc = tl.zeros((16, BN), dtype=tl.float32)
        index_mask = (1 << index_width) - 1
        group_lanes4 = tl.arange(0, BK // 4)
        group_lanes8 = tl.arange(0, BK // 8)
        scalar_lanes = tl.arange(0, BK)

        for k0 in range(0, K, BK):
            ks = k0 + scalar_lanes
            a = tl.load(
                a_ptr + rows[:, None] * K + ks[None, :],
                mask=active & row_mask[:, None],
                other=0.0,
            )
            is_d4 = dimension == 4

            groups4 = (k0 // 4) + group_lanes4
            bit_positions4 = groups4 * index_width
            byte_positions4 = bit_positions4 // 8
            shifts4 = bit_positions4 - byte_positions4 * 8
            addresses4 = (
                code_base
                + cols[:, None].to(tl.int64) * row_bytes
                + byte_positions4[None, :].to(tl.int64)
            )
            load_mask4 = active & is_d4 & col_mask[:, None]
            b04 = tl.load(codes_ptr + addresses4, mask=load_mask4, other=0).to(tl.int32)
            b14 = tl.load(
                codes_ptr + addresses4 + 1,
                mask=load_mask4 & (byte_positions4[None, :] + 1 < row_bytes),
                other=0,
            ).to(tl.int32)
            b24 = tl.load(
                codes_ptr + addresses4 + 2,
                mask=load_mask4 & (byte_positions4[None, :] + 2 < row_bytes),
                other=0,
            ).to(tl.int32)
            words4 = b04 | (b14 << 8) | (b24 << 16)
            indices4 = (words4 >> shifts4[None, :]) & index_mask
            scalar_indices4 = tl.broadcast_to(
                indices4[:, :, None], (BN, BK // 4, 4)
            ).reshape((BN, BK))

            groups8 = (k0 // 8) + group_lanes8
            bit_positions8 = groups8 * index_width
            byte_positions8 = bit_positions8 // 8
            shifts8 = bit_positions8 - byte_positions8 * 8
            addresses8 = (
                code_base
                + cols[:, None].to(tl.int64) * row_bytes
                + byte_positions8[None, :].to(tl.int64)
            )
            load_mask8 = active & (dimension == 8) & col_mask[:, None]
            b08 = tl.load(codes_ptr + addresses8, mask=load_mask8, other=0).to(tl.int32)
            b18 = tl.load(
                codes_ptr + addresses8 + 1,
                mask=load_mask8 & (byte_positions8[None, :] + 1 < row_bytes),
                other=0,
            ).to(tl.int32)
            b28 = tl.load(
                codes_ptr + addresses8 + 2,
                mask=load_mask8 & (byte_positions8[None, :] + 2 < row_bytes),
                other=0,
            ).to(tl.int32)
            words8 = b08 | (b18 << 8) | (b28 << 16)
            indices8 = (words8 >> shifts8[None, :]) & index_mask
            scalar_indices8 = tl.broadcast_to(
                indices8[:, :, None], (BN, BK // 8, 8)
            ).reshape((BN, BK))

            scalar_indices = tl.where(is_d4, scalar_indices4, scalar_indices8)
            components = tl.where(is_d4, scalar_lanes % 4, scalar_lanes % 8)
            cb_addresses = (
                cb_base
                + scalar_indices.to(tl.int64) * dimension
                + components[None, :].to(tl.int64)
            )
            scalar_mask = active & col_mask[:, None]
            cb = tl.load(
                codebooks_ptr + cb_addresses, mask=scalar_mask, other=0.0
            )
            scale_addresses = (
                scale_base
                + cols.to(tl.int64) * (K // 32)
                + (k0 // 32)
            )
            scale_byte = tl.load(
                scales_ptr + scale_addresses,
                mask=active & col_mask,
                other=127,
            ).to(tl.float32)
            weights = (
                cb.to(tl.float32) * tl.exp2(scale_byte[:, None] - 127.0)
            ).to(tl.bfloat16)
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(weights))

        tl.store(
            out_ptr + rows[:, None] * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=active & row_mask[:, None] & col_mask[None, :],
        )

    return triton, kernel


def fast_enabled() -> bool:
    """Single opt-in/rollback gate for every VQ decode optimization."""
    return os.getenv("VLLM_MOE_VQ_FAST", "0") == "1"


@functools.cache
def _cuda_warp_module():
    """Load the packageable CUDA warp-GEMV only when its explicit gate is on."""
    import vq_warp_gemv

    return vq_warp_gemv


def cuda_warp_enabled(
    state: dict[str, Any], mblock: int, valid_m: int
) -> bool:
    """Small-M decode gate with optional row and quality layer cutoffs.

    The scalar-FP32 warp reduction is extremely close to grouped Triton for one
    projection but not bit-identical; that error compounds through all 43 MoE
    layers.  ``VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER`` keeps the fast implementation
    only through an inclusive model-layer index and falls back to the exact
    grouped-Triton path above it. ``VLLM_MOE_VQ_CUDA_WARP_MAX_M`` defaults to 1
    for exact backward compatibility; the product MTP rail opts into rows 2-4
    after the real-L42 multi-row correctness gate passes.
    """
    max_layer = int(os.getenv("VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER", "2147483647"))
    max_m = int(os.getenv("VLLM_MOE_VQ_CUDA_WARP_MAX_M", "1"))
    layer_key = int(state.get("layer_key", -1))
    return (
        fast_enabled()
        and mblock == 4
        and 1 <= valid_m <= min(4, max_m)
        and os.getenv("VLLM_MOE_VQ_CUDA_WARP", "0") == "1"
        and (layer_key < 0 or layer_key <= max_layer)
        and state.get("scales") is not None
        and state["scales"].dtype == torch.uint8
        and state.get("codebooks") is not None
        and state["codebooks"].dtype == torch.float16
    )


def dispatch_probe_label(
    state: dict[str, Any], mblock: int, valid_m: int
) -> str:
    """Return a stable label for the branch that ``vq_gemm`` will take."""
    branch = "cuda_warp" if cuda_warp_enabled(state, mblock, valid_m) else "fallback"
    return f"{branch}_m{valid_m}"


def group_fast_enabled(state: dict[str, Any]) -> bool:
    """Whether mixed d4/d8 decode should use grouped-index reconstruction."""
    return (
        fast_enabled()
        and "dimension" in state
        and os.getenv("VLLM_MOE_VQ_GROUP_FAST", "1") == "1"
    )


def vq_gemm(
    a: torch.Tensor,
    out: torch.Tensor,
    expert_blocks: torch.Tensor,
    num_post: torch.Tensor,
    kind: torch.Tensor,
    state: dict[str, Any],
    *,
    n: int,
    k: int,
    mblock: int,
    valid_m: int | None = None,
) -> None:
    """Launch direct VQ GEMM for VQ-routed pairs; non-VQ rows are untouched."""
    if valid_m is None:
        valid_m = mblock
    if not 1 <= valid_m <= mblock:
        raise ValueError(f"invalid valid_m={valid_m} for mblock={mblock}")
    if a.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("VQ GEMM requires BF16 activations and output")
    if a.shape[1] != k or out.shape[1] != n:
        raise ValueError((a.shape, out.shape, n, k))
    if mblock not in (4, 16):
        raise ValueError(f"unsupported mblock: {mblock}")
    if int(state["n_outputs"]) != n:
        raise ValueError(f"wire outputs {state['n_outputs']} != GEMM outputs {n}")
    if "blob_ptrs" not in state:
        raise ValueError("VQ projection state is not bound to stable blob pointers")
    if cuda_warp_enabled(state, mblock, valid_m):
        # Exact incumbent call contract, compact pair grid, current stream, and
        # canonical mmap-backed row pack are preserved. This branch is decode
        # only; flag-off falls through to the proven grouped Triton path.
        _cuda_warp_module().vq_gemm(
            a,
            out,
            expert_blocks,
            num_post,
            kind,
            state,
            n=n,
            k=k,
            mblock=mblock,
            valid_m=valid_m,
        )
        return
    use_group = mblock == 4 and group_fast_enabled(state)
    use_d4 = (
        fast_enabled()
        and (not use_group)
        and bool(state.get("all_d4", False))
        and os.getenv("VLLM_MOE_VQ_D4_FAST", "1") == "1"
    )
    if use_group:
        triton, kernel = _compiled_kernel_grouped()
    elif use_d4:
        triton, kernel = _compiled_kernel_d4()
    else:
        triton, kernel = _compiled_kernel()
    pairs = expert_blocks.numel()
    default_bn = "128" if (use_group or use_d4) and mblock == 4 else "32"
    bn = int(os.getenv("VLLM_MOE_VQ_BN", default_bn))
    if bn not in (32, 64, 128, 256):
        raise ValueError(f"unsupported VLLM_MOE_VQ_BN: {bn}")
    kernel[(pairs, triton.cdiv(n, bn))](
        a,
        out,
        expert_blocks,
        num_post,
        kind,
        state["code_offset"],
        state["scale_offset"],
        state["code_row_bytes"],
        state["dimension"],
        state["bits"],
        state["cb_offset"],
        state["blob_ptrs"],
        N=n,
        K=k,
        MBLOCK=mblock,
        BN=bn,
        BK=32,
        num_warps=4,
    )
