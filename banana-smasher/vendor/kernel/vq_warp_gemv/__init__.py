# SPDX-License-Identifier: Apache-2.0
"""Thin torch.ops wrapper for the decode-only VQ warp GEMV."""
from __future__ import annotations

import torch

from . import _C  # noqa: F401


def vq_gemm(
    a: torch.Tensor,
    out: torch.Tensor,
    expert_blocks: torch.Tensor,
    num_post: torch.Tensor,
    kind: torch.Tensor,
    state: dict,
    *,
    n: int,
    k: int,
    mblock: int = 4,
) -> None:
    """Write VQ-routed rows to ``out``; leave non-VQ/inactive rows untouched."""
    torch.ops.vq_warp_gemv.gemm(
        a,
        out,
        expert_blocks,
        num_post,
        kind,
        state["codes"],
        state["scales"],
        state["codebooks"],
        state["code_offset"],
        state["scale_offset"],
        state["code_row_bytes"],
        state["dimension"],
        state["bits"],
        state["cb_offset"],
        n,
        k,
        mblock,
    )


__all__ = ["vq_gemm"]
