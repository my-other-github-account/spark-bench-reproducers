from __future__ import annotations

import inspect

import pytest
import torch

from banana_smasher.exact_codebook import (
    exhaustive_reference_winners,
    exact_codebook_winners,
    prepare_exact_codebook,
    tf32x3_score_error_bound,
)


def test_exhaustive_reference_uses_lowest_index_for_ties() -> None:
    x = torch.zeros((3, 4), dtype=torch.float32)
    cb = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert exhaustive_reference_winners(x, cb).tolist() == [0, 0, 0]


def test_exact_path_has_no_candidate_pruning_knob() -> None:
    parameters = inspect.signature(exact_codebook_winners).parameters
    assert "candidate_count" not in parameters
    assert "self_check_rows" not in parameters
    assert "reference_fallback" not in parameters
    x = torch.randn((32, 4), generator=torch.Generator().manual_seed(7))
    cb = torch.randn((80, 4), generator=torch.Generator().manual_seed(8))
    _, meta = exact_codebook_winners(x, cb)
    assert meta["candidates_per_row"] == 80
    assert meta["candidate_evaluations"] == 32 * 80
    assert meta["search_mode"] == "full-codebook"


def test_midpoint_tie_is_verified_and_preserves_reference_winner() -> None:
    cb = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.zeros((1, 4), dtype=torch.float32)
    winners, meta = exact_codebook_winners(x, cb)
    assert winners.tolist() == [0]
    assert meta["verified_rows"] == 1
    assert meta["certified_rows"] == 0


def test_well_separated_row_is_bound_certified() -> None:
    cb = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [16.0, 0.0, 0.0, 0.0],
            [-16.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor([[0.125, 0.0, 0.0, 0.0]], dtype=torch.float32)
    winners, meta = exact_codebook_winners(x, cb)
    assert winners.tolist() == [0]
    assert meta["verified_rows"] == 0
    assert meta["certified_rows"] == 1


def test_precomputed_codebook_plan_is_reused() -> None:
    cb = torch.randn((128, 4), generator=torch.Generator().manual_seed(33))
    x = torch.randn((256, 4), generator=torch.Generator().manual_seed(34))
    plan = prepare_exact_codebook(cb)
    first, first_meta = exact_codebook_winners(x, cb, plan=plan)
    second, second_meta = exact_codebook_winners(x, cb, plan=plan)
    assert torch.equal(first, second)
    assert plan.score_bias.data_ptr() == plan.score_bias.data_ptr()
    assert first_meta["precomputed_codebook"] is True
    assert second_meta["precomputed_codebook"] is True


def test_tf32x3_plan_keeps_fp32_codebook_and_fp32_norm_bias() -> None:
    cb = torch.tensor(
        [
            [1.001, -0.499, 3.14159, -2.71828],
            [-0.3333, 0.142857, 1.41421, 0.57721],
        ],
        dtype=torch.float32,
    )
    plan = prepare_exact_codebook(cb)
    assert plan.codebook_tf32x3.dtype == torch.float32
    assert torch.equal(plan.codebook_tf32x3, cb)
    assert torch.equal(plan.score_bias, -cb.square().sum(dim=1))


def test_tf32x3_bound_uses_conservative_fp32_tensorcore_envelope() -> None:
    cb = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-4.0, 3.0, -2.0, 1.0]],
        dtype=torch.float32,
    )
    x = torch.tensor([[0.5, -1.5, 2.5, -3.5]], dtype=torch.float32)
    bound = tf32x3_score_error_bound(x, cb)
    dot_abs = (x.abs() * cb.abs().max(dim=0).values).sum(dim=1)
    norm_abs = cb.square().sum(dim=1).max()
    expected = 128.0 * (2.0**-24) * (1.0 + 2.0 * dot_abs + norm_abs)
    assert torch.equal(bound, expected)
    assert bool(torch.isfinite(bound).all())
    assert bool((bound > 0).all())


def test_randomized_cpu_winners_match_exhaustive_reference() -> None:
    gen = torch.Generator().manual_seed(20260731)
    cb = torch.randn((257, 4), generator=gen, dtype=torch.float32)
    x = torch.randn((4096, 4), generator=gen, dtype=torch.float32)
    got, meta = exact_codebook_winners(x, cb, row_chunk=512)
    expected = exhaustive_reference_winners(x, cb, row_chunk=512)
    assert torch.equal(got, expected)
    assert "winner_diffs_vs_verified_rows" not in meta
    assert meta["certified_rows"] + meta["verified_rows"] == x.shape[0]


def test_same_shaped_plan_for_another_codebook_is_rejected() -> None:
    first = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0]], dtype=torch.float32
    )
    second = torch.tensor(
        [[4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    plan = prepare_exact_codebook(first)

    with pytest.raises(ValueError, match="active codebook"):
        exact_codebook_winners(torch.zeros((2, 4)), second, plan=plan)


def test_nonfinite_inputs_are_rejected_before_scoring() -> None:
    codebook = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    with pytest.raises(ValueError, match="finite"):
        exact_codebook_winners(torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), codebook)
    with pytest.raises(ValueError, match="finite"):
        exact_codebook_winners(torch.zeros((1, 4)), codebook.fill_(float("inf")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_full_codebook_winners_match_exhaustive_reference() -> None:
    gen = torch.Generator().manual_seed(731)
    cb = torch.randn((2048, 4), generator=gen, dtype=torch.float32).cuda()
    x = torch.randn((8192, 4), generator=gen, dtype=torch.float32).cuda()
    got, meta = exact_codebook_winners(x, cb, row_chunk=2048)
    expected = exhaustive_reference_winners(x, cb, row_chunk=2048)
    assert torch.equal(got, expected)
    assert meta["candidates_per_row"] == 2048
    assert meta["candidate_evaluations"] == 8192 * 2048
    assert meta["fast_pass"] == "tf32x3-full-score-top2"
    assert meta["fast_backend"] == "triton-fused-tf32x3-tensorcore"
    assert meta["certified_rows"] >= 8192 * 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_tie_canary_routes_to_fp32_verification() -> None:
    cb = torch.randn((2048, 4), generator=torch.Generator().manual_seed(99))
    cb[1] = cb[0]
    x = cb[0].repeat(64, 1).cuda()
    cb = cb.cuda()
    got, meta = exact_codebook_winners(x, cb)
    assert got.tolist() == [0] * 64
    assert meta["verified_rows"] == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_ambiguous_rows_use_fused_fp32_full_codebook_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import banana_smasher.exact_codebook as exact_module

    cb = torch.randn((2048, 4), generator=torch.Generator().manual_seed(199))
    cb[1] = cb[0]
    x = cb[0].repeat(128, 1).cuda()
    cb = cb.cuda()

    def forbidden_materialized_reference(*args, **kwargs):
        raise AssertionError("GPU ambiguity must use the fused FP32 full-codebook verifier")

    monkeypatch.setattr(
        exact_module,
        "exhaustive_reference_winners",
        forbidden_materialized_reference,
    )
    got, meta = exact_module.exact_codebook_winners(x, cb)
    assert got.tolist() == [0] * 128
    assert meta["verified_rows"] == 128
    assert meta["verification_backend"] == "triton-fused-fp32-ieee"
    assert meta["verification_candidate_evaluations"] == 128 * 2048
