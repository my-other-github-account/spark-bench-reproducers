from __future__ import annotations

import copy
import os
import threading
from typing import Any

import torch

from .kmajor_fused import fused_codebook_vjp_from_inputs

_LOCK = threading.Lock()
_GROUPS: dict[tuple[Any, ...], dict[str, Any]] = {}
_SEEN_GROUP_KEYS: set[tuple[Any, ...]] = set()
_TARGET_MODULE: Any | None = None
_BATCH_SIZE = 16
_STATS: dict[str, int] = {
    "forward_calls": 0,
    "backward_calls": 0,
    "batch_flushes": 0,
    "max_pending": 0,
}


def _group_key(
    codebook32: torch.Tensor, x: torch.Tensor, codes: torch.Tensor
) -> tuple[Any, ...]:
    return (
        codebook32.device.type,
        codebook32.device.index,
        int(codebook32.data_ptr()),
        int(codebook32._version),
        tuple(x.shape),
        tuple(codes.shape),
    )


def reset_batched_kmajor_vjp(*, batch_size: int | None = None) -> None:
    global _BATCH_SIZE
    selected = (
        int(os.environ.get("BANANA_SMASHER_KMAJOR_VJP_BATCH", "16"))
        if batch_size is None
        else int(batch_size)
    )
    if selected <= 0 or selected > 256:
        raise ValueError(f"invalid batched K-major VJP size {selected}")
    with _LOCK:
        if _GROUPS:
            raise RuntimeError(
                f"cannot reset batched K-major VJP with undrained tails: {len(_GROUPS)}"
            )
        _BATCH_SIZE = selected
        _SEEN_GROUP_KEYS.clear()
        for key in _STATS:
            _STATS[key] = 0


def batched_kmajor_vjp_stats() -> dict[str, int]:
    with _LOCK:
        return {
            **copy.deepcopy(_STATS),
            "batch_size": _BATCH_SIZE,
            "unique_groups": len(_SEEN_GROUP_KEYS),
            "active_groups": len(_GROUPS),
        }


def _flush_pending(
    pending: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    codebook_shape: tuple[int, ...],
) -> torch.Tensor:
    num_codes, code_dim = map(int, codebook_shape)
    if not pending[0][0].is_cuda:
        grad_outputs = torch.stack([row[0] for row in pending])
        activations = torch.stack([row[1] for row in pending])
        codes = torch.stack([row[2] for row in pending])
        scales = torch.stack([row[3] for row in pending])
        grad_weight_nk = torch.bmm(
            grad_outputs.transpose(1, 2), activations
        ).float()
        scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(
            32, -1
        )
        grouped = (grad_weight_nk * scale_columns).reshape(
            codes.shape[0], codes.shape[1], codes.shape[2], code_dim
        )
        grad_codebook = torch.zeros(
            codebook_shape, dtype=torch.float32, device=grad_outputs.device
        )
        grad_codebook.index_add_(
            0, codes.reshape(-1).long(), grouped.reshape(-1, code_dim)
        )
        return grad_codebook
    grad_codebook = torch.zeros(
        codebook_shape,
        dtype=torch.float32,
        device=pending[0][0].device,
    )
    for grad_output, activation, codes, scales in pending:
        fused_codebook_vjp_from_inputs(
            grad_output,
            activation,
            codes,
            scales,
            num_codes,
            code_dim,
            output=grad_codebook,
        )
    return grad_codebook


class BatchedKMajorVQLinearFn(torch.autograd.Function):
    """K-major linear with chunk-batched shared-codebook VJP reduction.

    Activation checkpointing executes an initial forward whose custom nodes are
    not traversed, then a recompute forward that is traversed. Groups therefore
    form lazily from actual backward calls; every full chunk returns one summed
    codebook gradient for normal leaf accumulation.
    """

    @staticmethod
    def forward(ctx, x, codebook32, codes, scales, dense_kn):
        if x.ndim != 2 or dense_kn.ndim != 2:
            raise ValueError("K-major VQ linear expects rank-2 activation/tile")
        if int(x.shape[1]) != int(dense_kn.shape[0]):
            raise ValueError(
                f"K-major activation/tile mismatch: {tuple(x.shape)} x {tuple(dense_kn.shape)}"
            )
        if codes.requires_grad or scales.requires_grad or dense_kn.requires_grad:
            raise ValueError("packed planes and detached K-major tile must remain frozen")
        x = x.contiguous()
        ctx.save_for_backward(x, codes, scales, dense_kn)
        ctx.group_key = _group_key(codebook32, x, codes)
        ctx.codebook_shape = tuple(codebook32.shape)
        output = torch.bmm(x.unsqueeze(0), dense_kn.unsqueeze(0)).squeeze(0)
        with _LOCK:
            _STATS["forward_calls"] += 1
        if _TARGET_MODULE is not None:
            _TARGET_MODULE._SENTINEL["bmm_launches"] += 1
        return output

    @staticmethod
    def backward(ctx, grad_out):
        x, codes, scales, dense_kn = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.mm(grad_out, dense_kn.transpose(0, 1))
        with _LOCK:
            _SEEN_GROUP_KEYS.add(ctx.group_key)
            group = _GROUPS.setdefault(
                ctx.group_key,
                {
                    "pending": [],
                    "codebook_shape": ctx.codebook_shape,
                },
            )
            if tuple(group["codebook_shape"]) != tuple(ctx.codebook_shape):
                raise RuntimeError("batched K-major VJP codebook shape drift")
            group["pending"].append((grad_out, x, codes, scales))
            _STATS["backward_calls"] += 1
            _STATS["max_pending"] = max(
                _STATS["max_pending"], len(group["pending"])
            )
            if len(group["pending"]) == _BATCH_SIZE:
                grad_codebook = _flush_pending(
                    group["pending"], tuple(group["codebook_shape"])
                )
                del _GROUPS[ctx.group_key]
                _STATS["batch_flushes"] += 1
            elif len(group["pending"]) > _BATCH_SIZE:
                raise RuntimeError("batched K-major VJP pending count exceeded batch")
            else:
                grad_codebook = None
        if _TARGET_MODULE is not None:
            _TARGET_MODULE._SENTINEL["backward_calls"] += 1
        return grad_x, grad_codebook, None, None, None


def install_batched_kmajor_vjp(kmajor_module: Any) -> dict[str, Any]:
    """Install the grouped VJP behind the existing public update runtime module."""
    global _TARGET_MODULE
    if getattr(kmajor_module, "_banana_smasher_batched_vjp_installed", False):
        return batched_kmajor_vjp_stats()
    _TARGET_MODULE = kmajor_module
    original_reset = kmajor_module.reset_kmajor_sentinel
    original_sentinel = kmajor_module.kmajor_sentinel

    def reset_kmajor_sentinel(*, clear_cache: bool = False) -> None:
        original_reset(clear_cache=clear_cache)
        reset_batched_kmajor_vjp()

    def kmajor_sentinel() -> dict[str, Any]:
        value = original_sentinel()
        value["batched_vjp"] = batched_kmajor_vjp_stats()
        return value

    kmajor_module.KMajorVQLinearFn = BatchedKMajorVQLinearFn
    kmajor_module.reset_kmajor_sentinel = reset_kmajor_sentinel
    kmajor_module.kmajor_sentinel = kmajor_sentinel
    kmajor_module._banana_smasher_batched_vjp_installed = True
    reset_batched_kmajor_vjp()
    return batched_kmajor_vjp_stats()
