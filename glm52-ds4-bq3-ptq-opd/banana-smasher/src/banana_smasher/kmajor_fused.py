from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on CPU-only package hosts
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _codebook_vjp_kernel(
        grad_weight,
        codes,
        scales,
        output,
        elements: tl.constexpr,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        row = offsets // in_features
        column = offsets - row * in_features
        scale_stride = in_features // 32
        code_stride = in_features // code_dim
        scale = tl.load(
            scales + row * scale_stride + column // 32, mask=mask, other=127
        ).to(tl.float32)
        code = tl.load(
            codes + row * code_stride + column // code_dim, mask=mask, other=0
        ).to(tl.int32)
        component = column % code_dim
        value = tl.load(grad_weight + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.atomic_add(
            output + code * code_dim + component,
            value * tl.exp2(scale - 127.0),
            mask=mask,
        )


    @triton.jit
    def _grouped_codebook_vjp_kernel(
        grad_weight,
        codes,
        scales,
        partial,
        elements: tl.constexpr,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        num_codes: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        expert = tl.program_id(1)
        # The production grad-weight slab has exactly 2**32 elements. Force
        # the slab base calculation to int64 while retaining per-expert row/
        # column math in the faster bounded int32 domain.
        expert_grad_weight_offset = expert.to(tl.int64) * elements
        row = offsets // in_features
        column = offsets - row * in_features
        scale_stride = in_features // 32
        code_stride = in_features // code_dim
        scale = tl.load(
            scales
            + expert * rows * scale_stride
            + row * scale_stride
            + column // 32,
            mask=mask,
            other=127,
        ).to(tl.float32)
        code = tl.load(
            codes
            + expert * rows * code_stride
            + row * code_stride
            + column // code_dim,
            mask=mask,
            other=0,
        ).to(tl.int32)
        component = column % code_dim
        value = tl.load(
            grad_weight + expert_grad_weight_offset + offsets.to(tl.int64),
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            partial
            + expert * num_codes * code_dim
            + code * code_dim
            + component,
            value * tl.exp2(scale - 127.0),
            mask=mask,
        )


    @triton.jit
    def _codebook_vjp_from_inputs_kernel(
        grad_output,
        activation,
        codes,
        scales,
        output,
        elements: tl.constexpr,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        microbatch: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        row = offsets // in_features
        column = offsets - row * in_features
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
        for batch_index in range(microbatch):
            grad_value = tl.load(
                grad_output + batch_index * rows + row,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            activation_value = tl.load(
                activation + batch_index * in_features + column,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += grad_value * activation_value
        scale_stride = in_features // 32
        code_stride = in_features // code_dim
        scale = tl.load(
            scales + row * scale_stride + column // 32, mask=mask, other=127
        ).to(tl.float32)
        code = tl.load(
            codes + row * code_stride + column // code_dim, mask=mask, other=0
        ).to(tl.int32)
        component = column % code_dim
        # Legacy torch.mm returns BF16 for BF16 inputs before the FP32 scale
        # multiply. Preserve that numerical contract while eliminating the
        # materialized grad-weight tensor.
        rounded_grad_weight = accumulator.to(tl.bfloat16).to(tl.float32)
        tl.atomic_add(
            output + code * code_dim + component,
            rounded_grad_weight * tl.exp2(scale - 127.0),
            mask=mask,
        )


    @triton.jit
    def _grouped_codebook_vjp_from_inputs_kernel(
        grad_output,
        activation,
        codes,
        scales,
        output,
        elements: tl.constexpr,
        rows: tl.constexpr,
        in_features: tl.constexpr,
        microbatch: tl.constexpr,
        num_codes: tl.constexpr,
        code_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Keep every program's logical offset within one expert. The production
        # projection has 256 * 4096 * 4096 == 2**32 elements, so flattening the
        # expert axis into program_id(0) overflows Triton's 32-bit offset math.
        # A two-dimensional launch keeps the largest offset below 2**24 while
        # still issuing exactly one grouped reduction kernel per projection.
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elements
        expert = tl.program_id(1)
        row = offsets // in_features
        column = offsets - row * in_features
        accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
        for batch_index in range(microbatch):
            grad_value = tl.load(
                grad_output
                + expert * microbatch * rows
                + batch_index * rows
                + row,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            activation_value = tl.load(
                activation
                + expert * microbatch * in_features
                + batch_index * in_features
                + column,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            accumulator += grad_value * activation_value
        scale_stride = in_features // 32
        code_stride = in_features // code_dim
        scale = tl.load(
            scales
            + expert * rows * scale_stride
            + row * scale_stride
            + column // 32,
            mask=mask,
            other=127,
        ).to(tl.float32)
        code = tl.load(
            codes
            + expert * rows * code_stride
            + row * code_stride
            + column // code_dim,
            mask=mask,
            other=0,
        ).to(tl.int32)
        component = column % code_dim
        rounded_grad_weight = accumulator.to(tl.bfloat16).to(tl.float32)
        tl.atomic_add(
            output
            + expert * num_codes * code_dim
            + code * code_dim
            + component,
            rounded_grad_weight * tl.exp2(scale - 127.0),
            mask=mask,
        )


def _validate_common(
    codes: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    in_features: int,
    num_codes: int,
    code_dim: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("fused K-major VJP requires Triton")
    if not codes.is_cuda or not scales.is_cuda:
        raise ValueError("fused K-major VJP requires CUDA codes and scales")
    if codes.dtype != torch.int32:
        raise ValueError(f"codes must be int32, got {codes.dtype}")
    if scales.dtype != torch.uint8:
        raise ValueError(f"scales must be uint8, got {scales.dtype}")
    if code_dim <= 0 or in_features % code_dim:
        raise ValueError(
            f"invalid code dimension {code_dim} for in_features={in_features}"
        )
    if in_features % 32:
        raise ValueError(f"in_features must be divisible by 32, got {in_features}")
    if tuple(codes.shape) != (rows, in_features // code_dim):
        raise ValueError(
            f"codes shape drift {tuple(codes.shape)} != {(rows, in_features // code_dim)}"
        )
    if tuple(scales.shape) != (rows, in_features // 32):
        raise ValueError(
            f"scales shape drift {tuple(scales.shape)} != {(rows, in_features // 32)}"
        )
    if output is None:
        output = torch.zeros(
            (num_codes, code_dim), device=codes.device, dtype=torch.float32
        )
    elif (
        not output.is_cuda
        or output.dtype != torch.float32
        or tuple(output.shape) != (num_codes, code_dim)
        or output.device != codes.device
        or not output.is_contiguous()
    ):
        raise ValueError("output must be contiguous CUDA float32 [num_codes, code_dim]")
    return output


def fused_codebook_vjp(
    grad_weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_codes: int,
    code_dim: int,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse scale decode, multiply, code gather and FP32 codebook reduction."""
    if not grad_weight.is_cuda or grad_weight.ndim != 2:
        raise ValueError("grad_weight must be a rank-2 CUDA tensor")
    grad_weight = grad_weight.contiguous()
    rows, in_features = map(int, grad_weight.shape)
    output = _validate_common(
        codes,
        scales,
        rows,
        in_features,
        int(num_codes),
        int(code_dim),
        output,
    )
    elements = rows * in_features
    block = 256
    _codebook_vjp_kernel[(triton.cdiv(elements, block),)](
        grad_weight,
        codes,
        scales,
        output,
        elements=elements,
        rows=rows,
        in_features=in_features,
        code_dim=int(code_dim),
        BLOCK=block,
        num_warps=4,
    )
    return output


def fused_codebook_vjp_from_inputs(
    grad_output: torch.Tensor,
    activation: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_codes: int,
    code_dim: int,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse grad-weight dot products and packed codebook-gradient reduction."""
    if (
        not grad_output.is_cuda
        or not activation.is_cuda
        or grad_output.ndim != 2
        or activation.ndim != 2
    ):
        raise ValueError("grad_output and activation must be rank-2 CUDA tensors")
    grad_output = grad_output.contiguous()
    activation = activation.contiguous()
    microbatch, rows = map(int, grad_output.shape)
    activation_batch, in_features = map(int, activation.shape)
    if activation_batch != microbatch:
        raise ValueError(
            f"microbatch mismatch {microbatch} != {activation_batch}"
        )
    output = _validate_common(
        codes,
        scales,
        rows,
        in_features,
        int(num_codes),
        int(code_dim),
        output,
    )
    elements = rows * in_features
    block = 256
    _codebook_vjp_from_inputs_kernel[(triton.cdiv(elements, block),)](
        grad_output,
        activation,
        codes,
        scales,
        output,
        elements=elements,
        rows=rows,
        in_features=in_features,
        microbatch=microbatch,
        code_dim=int(code_dim),
        BLOCK=block,
        num_warps=4,
    )
    return output


def _grouped_vjp_launch_grid(
    *, experts: int, rows: int, in_features: int, block: int
) -> tuple[int, int]:
    if min(experts, rows, in_features, block) <= 0:
        raise ValueError("grouped K-major VJP launch dimensions must be positive")
    elements_per_expert = rows * in_features
    return ((elements_per_expert + block - 1) // block, experts)


def fused_grouped_codebook_vjp(
    grad_weight: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_codes: int,
    code_dim: int,
) -> torch.Tensor:
    # Reduce all expert grad-weight matrices with one grouped Triton launch.
    if not grad_weight.is_cuda or grad_weight.ndim != 3:
        raise ValueError("grouped grad_weight must be a rank-3 CUDA tensor")
    grad_weight = grad_weight.contiguous()
    codes = codes.contiguous()
    scales = scales.contiguous()
    experts, rows, in_features = map(int, grad_weight.shape)
    if tuple(codes.shape) != (experts, rows, in_features // int(code_dim)):
        raise ValueError("grouped codes shape drift")
    if tuple(scales.shape) != (experts, rows, in_features // 32):
        raise ValueError("grouped scales shape drift")
    if codes.dtype != torch.int32 or scales.dtype != torch.uint8:
        raise ValueError("grouped packed planes must be int32 codes and uint8 scales")
    block = 256
    partial = torch.zeros(
        (experts, int(num_codes), int(code_dim)),
        device=grad_weight.device,
        dtype=torch.float32,
    )
    elements = rows * in_features
    grid = _grouped_vjp_launch_grid(
        experts=experts,
        rows=rows,
        in_features=in_features,
        block=block,
    )
    _grouped_codebook_vjp_kernel[grid](
        grad_weight,
        codes,
        scales,
        partial,
        elements=elements,
        rows=rows,
        in_features=in_features,
        num_codes=int(num_codes),
        code_dim=int(code_dim),
        BLOCK=block,
        num_warps=4,
    )
    return partial.sum(dim=0)


def fused_grouped_codebook_vjp_from_inputs(
    grad_output: torch.Tensor,
    activation: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    num_codes: int,
    code_dim: int,
) -> torch.Tensor:
    """Fuse every expert in one projection into one partial-reduction launch."""
    if (
        not grad_output.is_cuda
        or not activation.is_cuda
        or grad_output.ndim != 3
        or activation.ndim != 3
    ):
        raise ValueError(
            "grouped grad_output and activation must be rank-3 CUDA tensors"
        )
    grad_output = grad_output.contiguous()
    activation = activation.contiguous()
    codes = codes.contiguous()
    scales = scales.contiguous()
    experts, microbatch, rows = map(int, grad_output.shape)
    activation_experts, activation_batch, in_features = map(int, activation.shape)
    if (activation_experts, activation_batch) != (experts, microbatch):
        raise ValueError("grouped expert/microbatch shape drift")
    if tuple(codes.shape) != (experts, rows, in_features // int(code_dim)):
        raise ValueError("grouped codes shape drift")
    if tuple(scales.shape) != (experts, rows, in_features // 32):
        raise ValueError("grouped scales shape drift")
    if codes.dtype != torch.int32 or scales.dtype != torch.uint8:
        raise ValueError("grouped packed planes must be int32 codes and uint8 scales")
    block = 256
    output = torch.zeros(
        (experts, int(num_codes), int(code_dim)),
        device=grad_output.device,
        dtype=torch.float32,
    )
    elements = rows * in_features
    grid = _grouped_vjp_launch_grid(
        experts=experts,
        rows=rows,
        in_features=in_features,
        block=block,
    )
    _grouped_codebook_vjp_from_inputs_kernel[grid](
        grad_output,
        activation,
        codes,
        scales,
        output,
        elements=elements,
        rows=rows,
        in_features=in_features,
        microbatch=microbatch,
        num_codes=int(num_codes),
        code_dim=int(code_dim),
        BLOCK=block,
        num_warps=4,
    )
    return output.sum(dim=0)
