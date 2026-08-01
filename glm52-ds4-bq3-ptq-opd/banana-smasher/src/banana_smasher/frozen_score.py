"""Triton-fused and reference frozen-plane weighted-SSE scoring."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - package-only CPU environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _frozen_weighted_sse_kernel(
        w_ptr,
        h_ptr,
        codes_ptr,
        scales_ptr,
        codebook_ptr,
        codebook_offsets_ptr,
        partial_ptr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        GROUPS: tl.constexpr,
        SCALE_BLOCKS: tl.constexpr,
        D: tl.constexpr,
        N_TILES: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tile = pid % N_TILES
        block_row = (pid // N_TILES) % tl.cdiv(ROWS, BLOCK_M)
        spec = pid // (N_TILES * tl.cdiv(ROWS, BLOCK_M))

        rows = block_row * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < ROWS
        col_mask = cols < COLS
        mask = row_mask[:, None] & col_mask[None, :]

        weights = tl.load(
            w_ptr + rows[:, None] * COLS + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        h = tl.load(h_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        groups = cols // D
        code_index = spec * ROWS * GROUPS + rows[:, None] * GROUPS + groups[None, :]
        codes = tl.load(codes_ptr + code_index, mask=mask, other=0).to(tl.int32)
        cb_offset = tl.load(codebook_offsets_ptr + spec).to(tl.int32)
        dims = cols % D
        codewords = tl.load(
            codebook_ptr + (codes + cb_offset) * D + dims[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        scale_index = (
            spec * ROWS * SCALE_BLOCKS
            + rows[:, None] * SCALE_BLOCKS
            + (cols // 32)[None, :]
        )
        scales = tl.load(scales_ptr + scale_index, mask=mask, other=127).to(tl.float32)
        dequant = (codewords * tl.exp2(scales - 127.0)).to(tl.bfloat16).to(tl.float32)
        error = (weights - dequant) * (weights - dequant) * h[None, :]
        partial = tl.sum(error, axis=1)
        out_index = (spec * ROWS + rows) * N_TILES + tile
        tl.store(partial_ptr + out_index, partial, mask=row_mask)


def _validate_frozen_inputs(
    w: torch.Tensor,
    h: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    flat_codebook: torch.Tensor,
    codebook_offsets: torch.Tensor,
    *,
    vector_width: int,
) -> tuple[int, int, int]:
    if not isinstance(vector_width, int) or vector_width < 1:
        raise ValueError("vector_width must be a positive integer")
    if w.ndim != 2 or h.ndim != 1 or codes.ndim != 3 or scales.ndim != 3:
        raise ValueError("unexpected frozen scoring tensor rank")
    if flat_codebook.ndim != 2 or codebook_offsets.ndim != 1:
        raise ValueError("unexpected frozen codebook tensor rank")
    specs, rows, groups = map(int, codes.shape)
    cols = int(w.shape[1])
    if tuple(w.shape) != (rows, cols) or tuple(h.shape) != (cols,):
        raise ValueError("frozen scoring weight/H geometry mismatch")
    if tuple(scales.shape[:2]) != (specs, rows):
        raise ValueError("frozen scoring batch geometry mismatch")
    if groups * vector_width != cols or int(scales.shape[2]) * 32 != cols:
        raise ValueError("frozen scoring vector/scale geometry mismatch")
    if tuple(flat_codebook.shape[1:]) != (vector_width,):
        raise ValueError("frozen scoring codebook vector width mismatch")
    if int(codebook_offsets.numel()) != specs:
        raise ValueError("frozen scoring codebook offset count mismatch")
    if codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("frozen scoring codes must use an integer dtype")
    if scales.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("frozen scoring scales must use an integer dtype")
    if codebook_offsets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("frozen scoring codebook offsets must use an integer dtype")
    for name, tensor in (("weights", w), ("H", h), ("codebook", flat_codebook)):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"frozen scoring {name} must be finite")

    offsets = [int(value) for value in codebook_offsets.detach().cpu().tolist()]
    total_codewords = int(flat_codebook.shape[0])
    if not offsets or offsets[0] < 0 or offsets != sorted(offsets):
        raise ValueError("frozen scoring codebook offsets must be ordered and non-negative")
    ends = offsets[1:] + [total_codewords]
    for spec, (start, end) in enumerate(zip(offsets, ends, strict=True)):
        if start >= end or end > total_codewords:
            raise ValueError("frozen scoring codebook offsets define an empty or invalid slice")
        spec_codes = codes[spec]
        minimum = int(spec_codes.min().item())
        maximum = int(spec_codes.max().item())
        if minimum < 0 or maximum >= end - start:
            raise ValueError(
                f"frozen scoring code range for spec {spec} is outside [0, {end - start})"
            )
    return specs, rows, cols


def reference_frozen_weighted_errors(
    w: torch.Tensor,
    h: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    flat_codebook: torch.Tensor,
    codebook_offsets: torch.Tensor,
    *,
    vector_width: int,
) -> torch.Tensor:
    """Return exact eager weighted-SSE scores for every declared option."""

    specs, rows, cols = _validate_frozen_inputs(
        w,
        h,
        codes,
        scales,
        flat_codebook,
        codebook_offsets,
        vector_width=vector_width,
    )
    group_scales = scales.to(torch.float32).sub(127.0).exp2().repeat_interleave(32, dim=2)
    scores: list[torch.Tensor] = []
    offsets = [int(value) for value in codebook_offsets.detach().cpu().tolist()]
    for spec in range(specs):
        selected = flat_codebook.index_select(0, codes[spec].to(torch.int64).reshape(-1) + offsets[spec])
        dequant = selected.reshape(rows, cols).to(torch.float32) * group_scales[spec]
        dequant = dequant.to(torch.bfloat16).to(torch.float32)
        error = (w.to(torch.float32) - dequant).square() * h.to(torch.float32).unsqueeze(0)
        scores.append(error.sum())
    return torch.stack(scores)


def fused_frozen_weighted_errors(
    w: torch.Tensor,
    h: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    flat_codebook: torch.Tensor,
    codebook_offsets: torch.Tensor,
    *,
    vector_width: int,
    block_m: int = 8,
    block_n: int = 256,
) -> torch.Tensor:
    """Return one weighted-SSE scalar per frozen tier/variant."""

    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not all(
        tensor.is_cuda
        for tensor in (w, h, codes, scales, flat_codebook, codebook_offsets)
    ):
        raise ValueError("fused frozen scoring requires CUDA tensors")
    specs, rows, cols = _validate_frozen_inputs(
        w,
        h,
        codes,
        scales,
        flat_codebook,
        codebook_offsets,
        vector_width=vector_width,
    )
    groups = int(codes.shape[2])
    n_tiles = triton.cdiv(cols, block_n)
    row_blocks = triton.cdiv(rows, block_m)
    partial = torch.empty(
        (specs, rows, n_tiles), device=w.device, dtype=torch.float32
    )
    _frozen_weighted_sse_kernel[(specs * row_blocks * n_tiles,)](
        w.contiguous(),
        h.contiguous(),
        codes.contiguous(),
        scales.contiguous(),
        flat_codebook.contiguous(),
        codebook_offsets.to(torch.int32).contiguous(),
        partial,
        ROWS=rows,
        COLS=cols,
        GROUPS=groups,
        SCALE_BLOCKS=int(scales.shape[2]),
        D=vector_width,
        N_TILES=n_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=3,
    )
    return partial.sum(dim=(1, 2))
