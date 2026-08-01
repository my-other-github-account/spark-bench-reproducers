"""Full-codebook exact nearest-neighbour search.

The CUDA fast pass evaluates every codebook row with TF32x3 tensor cores and
keeps only fused top-2 scores.  A conservative row-wise error envelope certifies
separated winners; every ambiguous row is recomputed by a fused IEEE-FP32
full-codebook pass.  Candidate pruning is not part of this module's API.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU/package-only environments
    triton = None
    tl = None


@dataclass(frozen=True)
class FastTop2:
    ids: torch.Tensor
    best_scores: torch.Tensor
    second_scores: torch.Tensor


@dataclass(frozen=True)
class ExactCodebookPlan:
    codebook: torch.Tensor
    codebook_tf32x3: torch.Tensor
    codebook_bf16: torch.Tensor
    score_bias: torch.Tensor
    score_bias_bf16: torch.Tensor
    max_ec: torch.Tensor
    max_abs_c: torch.Tensor
    max_abs_c_ec: torch.Tensor
    max_ec_sq: torch.Tensor
    max_abs_cbb: torch.Tensor
    norm_abs: torch.Tensor
    norm_abs_bf16: torch.Tensor
    source_data_ptr: int
    source_version: int
    source_device: torch.device
    source_dtype: torch.dtype
    source_shape: tuple[int, ...]


def prepare_exact_codebook(codebook: torch.Tensor) -> ExactCodebookPlan:
    """Precompute all codebook-only fast-score, norm, and error-bound terms."""
    if codebook.ndim != 2 or codebook.shape[1] != 4:
        raise ValueError("the production codebook must have shape [C, 4]")
    if codebook.shape[0] < 2:
        raise ValueError("top-2 certification requires at least two candidates")
    if not codebook.is_floating_point():
        raise TypeError("codebook must be floating point")
    cb = codebook.float().contiguous()
    cbb = cb.to(torch.bfloat16).contiguous()
    cbb_float = cbb.float()
    ec = (cbb_float - cb).abs()
    return ExactCodebookPlan(
        codebook=cb,
        codebook_tf32x3=cb,
        codebook_bf16=cbb,
        score_bias=-cb.square().sum(dim=1),
        score_bias_bf16=-cbb_float.square().sum(dim=1),
        max_ec=ec.max(dim=0).values,
        max_abs_c=cb.abs().max(dim=0).values,
        max_abs_c_ec=(cb.abs() * ec).max(dim=0).values,
        max_ec_sq=ec.square().max(dim=0).values,
        max_abs_cbb=cbb_float.abs().max(dim=0).values,
        norm_abs=cb.square().sum(dim=1).max(),
        norm_abs_bf16=cbb_float.square().sum(dim=1).max(),
        source_data_ptr=codebook.data_ptr(),
        source_version=int(codebook._version),
        source_device=codebook.device,
        source_dtype=codebook.dtype,
        source_shape=tuple(codebook.shape),
    )


def _validate_plan(plan: ExactCodebookPlan, codebook: torch.Tensor) -> None:
    identity = (
        codebook.data_ptr(),
        int(codebook._version),
        codebook.device,
        codebook.dtype,
        tuple(codebook.shape),
    )
    expected = (
        plan.source_data_ptr,
        plan.source_version,
        plan.source_device,
        plan.source_dtype,
        plan.source_shape,
    )
    if identity != expected:
        raise ValueError("codebook plan does not match the active codebook identity")


if triton is not None:

    @triton.jit
    def _full_fp32_fused_top2_kernel(
        x_ptr,
        cb_ptr,
        cb_bias_ptr,
        winner_ptr,
        best_ptr,
        second_ptr,
        rows,
        CANDIDATES: tl.constexpr,
        D: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        """All-candidate FP32-input sweep with register-resident top-2."""
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, K_PAD)
        row_mask = row_offsets < rows
        x_offsets = row_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(
            x_ptr + x_offsets,
            mask=row_mask[:, None] & (k_offsets[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        best_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        second_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        best_id = tl.full((BLOCK_M,), CANDIDATES, tl.int32)

        for candidate_start in tl.static_range(0, CANDIDATES, BLOCK_N):
            candidate_offsets = candidate_start + tl.arange(0, BLOCK_N)
            candidate_mask = candidate_offsets < CANDIDATES
            cb_offsets = candidate_offsets[:, None] * D + k_offsets[None, :]
            cb_tile = tl.load(
                cb_ptr + cb_offsets,
                mask=candidate_mask[:, None] & (k_offsets[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            dots = tl.dot(
                x_tile,
                tl.trans(cb_tile),
                input_precision=INPUT_PRECISION,
                out_dtype=tl.float32,
            )
            bias = tl.load(
                cb_bias_ptr + candidate_offsets,
                mask=candidate_mask,
                other=-float("inf"),
            )
            scores = 2.0 * dots + bias[None, :]

            tile_best = tl.max(scores, axis=1)
            id_grid = tl.broadcast_to(candidate_offsets[None, :], (BLOCK_M, BLOCK_N))
            tile_best_id = tl.min(
                tl.where(scores == tile_best[:, None], id_grid, CANDIDATES),
                axis=1,
            )
            scores_without_best = tl.where(
                id_grid == tile_best_id[:, None], -float("inf"), scores
            )
            tile_second = tl.max(scores_without_best, axis=1)

            old_best = best_score
            old_second = second_score
            old_id = best_id
            tile_wins = (tile_best > old_best) | (
                (tile_best == old_best) & (tile_best_id < old_id)
            )
            best_score = tl.maximum(old_best, tile_best)
            best_id = tl.where(tile_wins, tile_best_id, old_id)
            second_score = tl.maximum(
                tl.maximum(old_second, tile_second),
                tl.minimum(old_best, tile_best),
            )

        tl.store(winner_ptr + row_offsets, best_id, mask=row_mask)
        tl.store(best_ptr + row_offsets, best_score, mask=row_mask)
        tl.store(second_ptr + row_offsets, second_score, mask=row_mask)


def _triton_full_top2(
    x: torch.Tensor,
    codebook: torch.Tensor,
    plan: ExactCodebookPlan | None = None,
    *,
    input_precision: str = "tf32x3",
) -> FastTop2:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not x.is_cuda or not codebook.is_cuda:
        raise ValueError("Triton fast path requires CUDA tensors")
    if input_precision not in {"tf32x3", "ieee"}:
        raise ValueError("input_precision must be tf32x3 or ieee")
    rows = int(x.shape[0])
    candidates = int(codebook.shape[0])
    if candidates % 64:
        raise ValueError("Triton fast path requires candidate count divisible by 64")
    if plan is None:
        plan = prepare_exact_codebook(codebook)
    _validate_plan(plan, codebook)
    winners = torch.empty((rows,), device=x.device, dtype=torch.int32)
    best = torch.empty((rows,), device=x.device, dtype=torch.float32)
    second = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block_m = 32
    _full_fp32_fused_top2_kernel[(triton.cdiv(rows, block_m),)](
        x,
        plan.codebook_tf32x3,
        plan.score_bias,
        winners,
        best,
        second,
        rows,
        CANDIDATES=candidates,
        D=4,
        K_PAD=16,
        BLOCK_M=block_m,
        BLOCK_N=64,
        INPUT_PRECISION=input_precision,
        num_warps=4,
        num_stages=3,
    )
    return FastTop2(winners.to(torch.int64), best, second)


def _validate_inputs(x: torch.Tensor, codebook: torch.Tensor) -> None:
    if x.ndim != 2 or codebook.ndim != 2:
        raise ValueError("x and codebook must be rank-2")
    if x.shape[1] != codebook.shape[1]:
        raise ValueError("x and codebook dimensions differ")
    if x.shape[1] != 4:
        raise ValueError("the production d4 codebook path requires D=4")
    if codebook.shape[0] < 2:
        raise ValueError("top-2 certification requires at least two candidates")
    if x.device != codebook.device:
        raise ValueError("x and codebook must be on the same device")
    if not x.is_floating_point() or not codebook.is_floating_point():
        raise TypeError("x and codebook must be floating point")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(codebook).all()):
        raise ValueError("x and codebook must contain only finite values")


def exhaustive_reference_winners(
    x: torch.Tensor,
    codebook: torch.Tensor,
    *,
    row_chunk: int = 65_536,
) -> torch.Tensor:
    """Return the exhaustive FP32 winner, preserving lowest-index ties.

    This mirrors the solver's reference expression: `-2*x@cb.T + ||cb||^2`,
    followed by PyTorch `argmin`.
    """
    _validate_inputs(x, codebook)
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    xf = x.float()
    cb = codebook.float()
    cb_norm = cb.square().sum(dim=1)
    parts: list[torch.Tensor] = []
    for start in range(0, xf.shape[0], row_chunk):
        rows = xf[start : start + row_chunk]
        distance_without_x_norm = rows @ cb.t()
        distance_without_x_norm.mul_(-2.0).add_(cb_norm.unsqueeze(0))
        parts.append(distance_without_x_norm.argmin(dim=1))
    if not parts:
        return torch.empty((0,), device=x.device, dtype=torch.int64)
    return torch.cat(parts)


def _lowest_index_top2(scores: torch.Tensor) -> FastTop2:
    """Top-2 scores with deterministic lowest-index selection for best ties."""
    best_scores = scores.max(dim=1).values
    candidate_ids = torch.arange(scores.shape[1], device=scores.device)
    best_ids = torch.where(
        scores == best_scores[:, None],
        candidate_ids[None, :],
        scores.shape[1],
    ).min(dim=1).values
    without_selected = scores.clone()
    without_selected.scatter_(1, best_ids[:, None], -math.inf)
    second_scores = without_selected.max(dim=1).values
    return FastTop2(best_ids, best_scores, second_scores)


def _bf16_full_top2(
    x: torch.Tensor,
    codebook: torch.Tensor,
    *,
    row_chunk: int,
    plan: ExactCodebookPlan | None = None,
) -> FastTop2:
    """Portable full-codebook BF16 fast pass used by CPU tests and fallback."""
    if plan is None:
        plan = prepare_exact_codebook(codebook)
    else:
        _validate_plan(plan, codebook)
    if x.is_cuda and codebook.is_cuda and triton is not None:
        return _triton_full_top2(x, codebook, plan)
    xb = x.to(torch.bfloat16).float()
    cbb = plan.codebook_bf16.float()
    cb_score_bias = plan.score_bias_bf16
    ids: list[torch.Tensor] = []
    best: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    for start in range(0, xb.shape[0], row_chunk):
        rows = xb[start : start + row_chunk]
        scores = 2.0 * (rows @ cbb.t()) + cb_score_bias.unsqueeze(0)
        top2 = _lowest_index_top2(scores)
        ids.append(top2.ids)
        best.append(top2.best_scores)
        second.append(top2.second_scores)
    if not ids:
        empty_i = torch.empty((0,), device=x.device, dtype=torch.int64)
        empty_f = torch.empty((0,), device=x.device, dtype=torch.float32)
        return FastTop2(empty_i, empty_f, empty_f)
    return FastTop2(torch.cat(ids), torch.cat(best), torch.cat(second))


def tf32x3_score_error_bound(
    x: torch.Tensor,
    codebook: torch.Tensor,
    plan: ExactCodebookPlan | None = None,
) -> torch.Tensor:
    """Conservative score-error envelope for Triton's TF32x3 tensor-core dot.

    Triton's TF32x3 decomposition uses three TF32 tensor-core products to
    recover FP32 product precision.  A factor of 128 FP32 unit roundoffs covers
    the three padded K=16 dot accumulations, decomposition/recombination,
    multiply-by-two, bias addition, and the separately evaluated exhaustive
    FP32 reference.  The additive one keeps the bound nonzero near the origin.
    """
    _validate_inputs(x, codebook)
    if plan is None:
        plan = prepare_exact_codebook(codebook)
    xf = x.float()
    dot_abs = (xf.abs() * plan.max_abs_c).sum(dim=1)
    unit_roundoff = 2.0**-24
    return 128.0 * unit_roundoff * (1.0 + 2.0 * dot_abs + plan.norm_abs)


def bf16_score_error_bound(
    x: torch.Tensor,
    codebook: torch.Tensor,
    plan: ExactCodebookPlan | None = None,
) -> torch.Tensor:
    """Conservative per-row absolute score-error bound for the BF16 fast pass.

    The quantization term bounds `2*x.c - ||c||^2` after independently rounding
    x and c to BF16.  The rounding term assumes a padded K=16 tensor-core dot
    with FP32 accumulation and is inflated to cover both fast and reference
    FP32 score construction.
    """
    _validate_inputs(x, codebook)
    if plan is None:
        plan = prepare_exact_codebook(codebook)
    xf = x.float()
    xb = xf.to(torch.bfloat16).float()
    ex = (xb - xf).abs()

    quant_bound = (
        2.0 * (xf.abs() * plan.max_ec).sum(dim=1)
        + 2.0 * (ex * plan.max_abs_c).sum(dim=1)
        + 2.0 * (ex * plan.max_ec).sum(dim=1)
        + (2.0 * plan.max_abs_c_ec + plan.max_ec_sq).sum()
    )

    # Higham gamma_n bound, deliberately using padded K=16 rather than D=4.
    unit_roundoff = 2.0**-24
    gamma = (16.0 * unit_roundoff) / (1.0 - 16.0 * unit_roundoff)
    dot_abs = (xb.abs() * plan.max_abs_cbb).sum(dim=1)
    norm_abs = plan.norm_abs_bf16
    dot_round = gamma * dot_abs
    norm_round = gamma * norm_abs
    final_sub_round = unit_roundoff * (
        2.0 * (dot_abs + dot_round) + norm_abs + norm_round
    )
    # Two-sided inflation covers tensor-core order, score subtraction, and the
    # separately evaluated FP32 exhaustive reference expression.
    round_bound = 4.0 * (2.0 * dot_round + norm_round + final_sub_round)
    finite_guard = 8.0 * unit_roundoff * (
        1.0 + 2.0 * dot_abs + norm_abs
    )
    return quant_bound + round_bound + finite_guard


def exact_codebook_winners(
    x: torch.Tensor,
    codebook: torch.Tensor,
    *,
    row_chunk: int = 65_536,
    plan: ExactCodebookPlan | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return exact winners using the fast full-codebook algorithm.

    The TF32x3/BF16 pass and bound-ambiguous IEEE recomputation are one exact
    algorithm. Independent reference comparisons belong in CI, not this path.
    """
    _validate_inputs(x, codebook)
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")

    stage_seconds = {
        "fast_full_codebook": 0.0,
        "error_bound": 0.0,
        "ambiguous_reference_verification": 0.0,
    }

    def begin_stage() -> float:
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        return time.perf_counter()

    def end_stage(started: float) -> float:
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        return time.perf_counter() - started

    precomputed = plan is not None
    if plan is None:
        plan = prepare_exact_codebook(codebook)
    cuda_tensorcore = x.is_cuda and codebook.is_cuda and triton is not None
    stage_started = begin_stage()
    fast = _bf16_full_top2(x, codebook, row_chunk=row_chunk, plan=plan)
    stage_seconds["fast_full_codebook"] = end_stage(stage_started)
    stage_started = begin_stage()
    bounds = (
        tf32x3_score_error_bound(x, codebook, plan)
        if cuda_tensorcore
        else bf16_score_error_bound(x, codebook, plan)
    )
    margins = fast.best_scores - fast.second_scores
    certified = torch.isfinite(margins) & (margins > 2.0 * bounds)
    verify = ~certified
    stage_seconds["error_bound"] = end_stage(stage_started)

    winners = fast.ids.to(torch.int64).clone()
    fast_changes = 0
    if bool(verify.any()):
        stage_started = begin_stage()
        if cuda_tensorcore:
            verified = _triton_full_top2(
                x[verify], codebook, plan, input_precision="ieee"
            ).ids
        else:
            verified = exhaustive_reference_winners(
                x[verify], codebook, row_chunk=row_chunk
            )
        fast_changes = int((winners[verify] != verified).sum().item())
        winners[verify] = verified
        stage_seconds["ambiguous_reference_verification"] = end_stage(stage_started)

    rows = int(x.shape[0])
    certified_rows = int(certified.sum().item())
    verified_rows = rows - certified_rows
    meta: dict[str, Any] = {
        "schema": "exact-codebook-search-v1",
        "search_mode": "full-codebook",
        "fast_pass": (
            "tf32x3-full-score-top2"
            if cuda_tensorcore
            else "bf16-full-score-top2"
        ),
        "fast_backend": (
            "triton-fused-tf32x3-tensorcore"
            if cuda_tensorcore
            else "torch-portable"
        ),
        "precomputed_codebook": precomputed,
        "verification": "full-codebook-fp32-on-bound-ambiguous-rows",
        "verification_backend": (
            "triton-fused-fp32-ieee"
            if cuda_tensorcore
            else "torch-materialized-fp32"
        ),
        "rows": rows,
        "candidates_per_row": int(codebook.shape[0]),
        "candidate_evaluations": rows * int(codebook.shape[0]),
        "verification_candidate_evaluations": verified_rows
        * int(codebook.shape[0]),
        "certified_rows": certified_rows,
        "verified_rows": verified_rows,
        "fast_winner_changes_after_verification": fast_changes,
        "backend_selected": (
            "triton-fused-tf32x3-tensorcore"
            if cuda_tensorcore
            else "torch-portable-exact"
        ),
        "stage_seconds": stage_seconds,
        "max_error_bound": float(bounds.max().item()) if rows else 0.0,
        "min_fast_margin": float(margins.min().item()) if rows else math.inf,
    }
    return winners, meta
