"""Public QTIP L16/K2/V2 persistent exact Viterbi integration."""
from __future__ import annotations

import types
from typing import Any

import torch

from .exact import BRANCHES, PREFIXES, STEPS, geometry, trellis_v2_exact


def trellis_v2(
    cb: Any,
    x: torch.Tensor,
    overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    states = trellis_v2_exact(cb, x, overlap)
    if bool(getattr(cb, "_trellis_v2_collect_stats", True)):
        batch = int(x.shape[1])
        initial_branches = 1 if overlap is not None else BRANCHES
        scored = batch * PREFIXES * (initial_branches + (STEPS - 1) * BRANCHES)
        last = {
            "mma_tiles": 0,
            "certified_transitions": 0,
            "fallback_transitions": 0,
            "exact_transitions": scored,
            "scored_transitions": scored,
            "certified_fraction": 0.0,
            "fallback_fraction": 0.0,
            "exact_fraction": 1.0,
        }
        cb._trellis_v2_last_stats = last
        total = dict(getattr(cb, "_trellis_v2_total_stats", {}))
        for key in ("mma_tiles", "certified_transitions", "fallback_transitions", "exact_transitions", "scored_transitions"):
            total[key] = int(total.get(key, 0)) + int(last[key])
        cb._trellis_v2_total_stats = total
    return states


def install_trellis_v2(cb: Any) -> dict[str, int | str | bool]:
    meta = geometry(cb)

    def viterbi(
        self: Any,
        x: torch.Tensor,
        overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return trellis_v2(self, x, overlap)

    def quantize_seq(
        self: Any,
        x: torch.Tensor,
        overlap: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        return trellis_v2(self, x, overlap)

    cb.viterbi = types.MethodType(viterbi, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return meta


__all__ = ["geometry", "trellis_v2", "install_trellis_v2"]
