from __future__ import annotations

import pytest

from banana_smasher.update import (
    _logical_segment_bounds,
    _logical_source_plan,
    _set_logical_training_extent,
)


def test_logical_8192_window_is_exactly_eight_contiguous_1024_segments() -> None:
    bounds = _logical_segment_bounds(1024, 8)
    assert bounds == [(index * 1024, (index + 1) * 1024) for index in range(8)]
    assert bounds[0] == (0, 1024)
    assert bounds[-1] == (7168, 8192)


@pytest.mark.parametrize("tokens,segments", [(0, 8), (1025, 8), (1024, 0), (1024, 9)])
def test_logical_segment_geometry_fails_closed(tokens: int, segments: int) -> None:
    with pytest.raises(ValueError):
        _logical_segment_bounds(tokens, segments)


def test_logical_window_expands_legacy_1024_loader_extent() -> None:
    class TrainingModule:
        T_TRAIN = 1024

    module = TrainingModule()
    assert _set_logical_training_extent(module, 8192) == 1024
    assert module.T_TRAIN == 8192


def test_logical_8192_extent_spans_explicit_sparse_source_windows_exactly() -> None:
    corpus = [
        {"real_len": length, "token_ids": list(range(length))}
        for length in [2048, 7, 2048, 9, 2044, 11, 2048, 13, 2048]
    ]
    plan = _logical_source_plan(corpus, [0, 2, 4, 6, 8], 8192)
    assert plan == [(0, 2048), (2, 2048), (4, 2044), (6, 2048), (8, 4)]
    assert sum(take for _, take in plan) == 8192


def test_logical_source_plan_fails_closed_when_explicit_assets_are_short() -> None:
    corpus = [{"real_len": 2048, "token_ids": list(range(2048))} for _ in range(4)]
    with pytest.raises(RuntimeError, match="provide 4096 tokens"):
        _logical_source_plan(corpus, [0, 3], 8192)