#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Codec and schema helpers for exact d4/d8 IQ3 VQ serving packs.

Each code row is independently bit-packed so a runtime kernel can address an
expert/output row without scanning preceding rows. Weight scales retain the
campaign's block-32 UE8M0 byte representation verbatim.
"""
from __future__ import annotations

import math

import numpy as np


_VQ_TIERS = {
    "vqa": (4, 256),
    "d4_k1024": (4, 1024),
    "d4_k2048": (4, 2048),
    "d4_k4096": (4, 4096),
    "d8_k256": (8, 256),
    "d8_k512": (8, 512),
    "d8_k1024": (8, 1024),
    "d8_k2048": (8, 2048),
    "d8_k4096": (8, 4096),
}


def parse_vq_tier(name: str) -> tuple[int, int]:
    """Return ``(vector_dimension, codebook_size)`` for a wire VQ tier."""
    try:
        return _VQ_TIERS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported VQ tier: {name}") from exc


def index_bits(codebook_size: int) -> int:
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise ValueError(f"codebook size must be a power of two: {codebook_size}")
    return int(math.log2(codebook_size))


def packed_row_bytes(n_indices: int, bits: int) -> int:
    if n_indices < 0 or not 1 <= bits <= 16:
        raise ValueError((n_indices, bits))
    return (n_indices * bits + 7) // 8


def pack_index_rows(values: np.ndarray, bits: int) -> np.ndarray:
    """Pack a 2-D unsigned index array into independent little-endian rows."""
    src = np.asarray(values)
    if src.ndim != 2 or src.dtype.kind != "u":
        raise ValueError("indices must be a 2-D unsigned integer array")
    if not 1 <= bits <= 16:
        raise ValueError(f"unsupported index width: {bits}")
    limit = 1 << bits
    if src.size and int(src.max()) >= limit:
        raise ValueError(f"index does not fit in {bits}-bit wire representation")

    rows, columns = src.shape
    out = np.zeros((rows, packed_row_bytes(columns, bits)), dtype=np.uint8)
    src64 = src.astype(np.uint64, copy=False)
    for column in range(columns):
        bit = column * bits
        byte, shift = divmod(bit, 8)
        word = src64[:, column] << shift
        out[:, byte] |= (word & 0xFF).astype(np.uint8)
        if byte + 1 < out.shape[1]:
            out[:, byte + 1] |= ((word >> 8) & 0xFF).astype(np.uint8)
        if byte + 2 < out.shape[1]:
            out[:, byte + 2] |= ((word >> 16) & 0xFF).astype(np.uint8)
    return out


def unpack_index_rows(packed: np.ndarray, bits: int, n_indices: int) -> np.ndarray:
    """Inverse of :func:`pack_index_rows`, returning uint16 indices."""
    src = np.asarray(packed, dtype=np.uint8)
    if src.ndim != 2:
        raise ValueError("packed indices must be a 2-D byte array")
    expected = packed_row_bytes(n_indices, bits)
    if src.shape[1] != expected:
        raise ValueError(f"packed row has {src.shape[1]} bytes; expected {expected}")

    padded = np.pad(src, ((0, 0), (0, 2)))
    out = np.empty((src.shape[0], n_indices), dtype=np.uint16)
    mask = (1 << bits) - 1
    for column in range(n_indices):
        bit = column * bits
        byte, shift = divmod(bit, 8)
        word = (
            padded[:, byte].astype(np.uint32)
            | (padded[:, byte + 1].astype(np.uint32) << 8)
            | (padded[:, byte + 2].astype(np.uint32) << 16)
        )
        out[:, column] = ((word >> shift) & mask).astype(np.uint16)
    return out


def decode_vq_rows(
    packed_codes: np.ndarray,
    scales: np.ndarray,
    codebook: np.ndarray,
    width: int,
) -> np.ndarray:
    """Reference wire decode, matching the campaign's FP16 stored codebooks.

    The return is FP16 for portable byte-exact tests. The serving kernel casts
    the reconstructed values to BF16 before tensor-core accumulation, matching
    the offline evaluator's final weight conversion.
    """
    if width % 32:
        raise ValueError("weight width must be a multiple of 32")
    cb = np.asarray(codebook, dtype=np.float16)
    if cb.ndim != 2 or cb.shape[1] not in (4, 8):
        raise ValueError(f"unsupported codebook shape: {cb.shape}")
    scale = np.asarray(scales, dtype=np.uint8)
    if scale.ndim != 2 or scale.shape[1] != width // 32:
        raise ValueError(f"scale shape {scale.shape} does not match width {width}")
    bits = index_bits(cb.shape[0])
    codes = unpack_index_rows(packed_codes, bits, width // cb.shape[1])
    if codes.shape[0] != scale.shape[0]:
        raise ValueError("code/scale row count mismatch")
    weights = cb[codes].reshape(codes.shape[0], width).astype(np.float32)
    weights *= np.exp2(scale.astype(np.float32) - 127).repeat(32, axis=1)
    return weights.astype(np.float16)


def pack_vq_projection(
    assignments: list[str],
    source_rows: dict[str, np.ndarray],
    sources: dict[str, dict[str, np.ndarray]],
    width: int,
    codebook_overrides: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Flatten one projection's heterogeneous VQ rows for direct serving."""
    if width % 32:
        raise ValueError("weight width must be a multiple of 32")
    overrides = codebook_overrides or {}
    tier_names = sorted(set(assignments))
    tier_info: dict[str, dict[str, int]] = {}
    codebook_parts = []
    cb_cursor = 0
    for tier in tier_names:
        d, k = parse_vq_tier(tier)
        if tier not in sources:
            raise ValueError(f"missing source for VQ tier: {tier}")
        cb = np.asarray(overrides.get(tier, sources[tier]["codebook"]), dtype=np.float16)
        if cb.shape != (k, d):
            raise ValueError(f"{tier} codebook shape {cb.shape}; expected {(k, d)}")
        tier_info[tier] = {
            "dimension": d,
            "codebook_size": k,
            "bits": index_bits(k),
            "cb_offset": cb_cursor,
        }
        codebook_parts.append(cb.reshape(-1))
        cb_cursor += cb.size

    n = len(assignments)
    code_offset = np.empty(n, dtype=np.int64)
    scale_offset = np.empty(n, dtype=np.int64)
    code_row_bytes = np.empty(n, dtype=np.int32)
    dimension = np.empty(n, dtype=np.uint8)
    bits_arr = np.empty(n, dtype=np.uint8)
    cb_offset = np.empty(n, dtype=np.int64)
    code_parts = []
    scale_parts = []
    code_cursor = scale_cursor = 0
    used = {tier: 0 for tier in tier_names}
    n_outputs = None

    for expert, tier in enumerate(assignments):
        info = tier_info[tier]
        pos = used[tier]
        rows = np.asarray(source_rows[tier])
        if pos >= rows.size:
            raise ValueError(f"not enough source rows for {tier}")
        source_row = int(rows[pos])
        used[tier] = pos + 1
        source = sources[tier]
        codes = np.asarray(source["codes"])
        scales = np.asarray(source["scales"], dtype=np.uint8)
        if not 0 <= source_row < codes.shape[0] or source_row >= scales.shape[0]:
            raise ValueError(f"invalid source row {source_row} for {tier}")
        if codes.ndim != 3 or scales.ndim != 3:
            raise ValueError(f"{tier} codes/scales must be expert-by-output matrices")
        expected_codes = width // info["dimension"]
        if codes.shape[2] != expected_codes:
            raise ValueError(
                f"{tier} code width {codes.shape[2]}; expected {expected_codes}"
            )
        if scales.shape[2] != width // 32:
            raise ValueError(f"{tier} scale width {scales.shape[2]}; expected {width // 32}")
        if codes.shape[1] != scales.shape[1]:
            raise ValueError(f"{tier} code/scale output row mismatch")
        if n_outputs is None:
            n_outputs = codes.shape[1]
        elif codes.shape[1] != n_outputs:
            raise ValueError(f"{tier} output rows {codes.shape[1]}; expected {n_outputs}")
        packed = pack_index_rows(codes[source_row], info["bits"])
        scale = np.ascontiguousarray(scales[source_row])

        code_offset[expert] = code_cursor
        scale_offset[expert] = scale_cursor
        code_row_bytes[expert] = packed.shape[1]
        dimension[expert] = info["dimension"]
        bits_arr[expert] = info["bits"]
        cb_offset[expert] = info["cb_offset"]
        code_parts.append(packed.reshape(-1))
        scale_parts.append(scale.reshape(-1))
        code_cursor += packed.size
        scale_cursor += scale.size

    for tier, count in used.items():
        total = np.asarray(source_rows[tier]).size
        if count != total:
            raise ValueError(f"unused source rows for {tier}: {total - count}")

    return {
        "codes": np.concatenate(code_parts) if code_parts else np.empty(0, dtype=np.uint8),
        "scales": np.concatenate(scale_parts) if scale_parts else np.empty(0, dtype=np.uint8),
        "codebooks": (
            np.concatenate(codebook_parts) if codebook_parts else np.empty(0, dtype=np.float16)
        ),
        "code_offset": code_offset,
        "scale_offset": scale_offset,
        "code_row_bytes": code_row_bytes,
        "dimension": dimension,
        "bits": bits_arr,
        "cb_offset": cb_offset,
        "tier_names": tier_names,
        "tier_info": tier_info,
        "n_outputs": int(n_outputs or 0),
    }


def decode_projection_expert(
    packed: dict[str, object], expert: int, width: int
) -> np.ndarray:
    """Reference-decode one expert from :func:`pack_vq_projection` output."""
    code_offset = int(packed["code_offset"][expert])
    row_bytes = int(packed["code_row_bytes"][expert])
    scale_offset = int(packed["scale_offset"][expert])
    d = int(packed["dimension"][expert])
    bits = int(packed["bits"][expert])
    cb_offset = int(packed["cb_offset"][expert])
    n_outputs = int(packed["n_outputs"])
    k = 1 << bits
    codebook = packed["codebooks"][cb_offset : cb_offset + k * d].reshape(k, d)
    codes = packed["codes"][
        code_offset : code_offset + n_outputs * row_bytes
    ].reshape(n_outputs, row_bytes)
    scales = packed["scales"][
        scale_offset : scale_offset + n_outputs * (width // 32)
    ].reshape(n_outputs, width // 32)
    return decode_vq_rows(codes, scales, codebook, width)
