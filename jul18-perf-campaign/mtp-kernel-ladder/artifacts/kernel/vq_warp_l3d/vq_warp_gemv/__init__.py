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
    valid_m: int = 1,
) -> None:
    """Write the first ``valid_m`` routed rows of every compact expert block."""
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
        valid_m,
    )


__all__ = ["vq_gemm"]
