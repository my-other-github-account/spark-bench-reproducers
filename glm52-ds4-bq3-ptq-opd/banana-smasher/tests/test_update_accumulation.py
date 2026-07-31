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


def test_logical_8192_extent_spans_consecutive_corpus_windows_exactly() -> None:
    corpus = [
        {"real_len": length, "token_ids": list(range(length))}
        for length in [2048, 2048, 2044, 2048, 2048]
    ]
    plan = _logical_source_plan(corpus, 0, 8192)
    assert plan == [(0, 2048), (1, 2048), (2, 2044), (3, 2048), (4, 4)]
    assert sum(take for _, take in plan) == 8192