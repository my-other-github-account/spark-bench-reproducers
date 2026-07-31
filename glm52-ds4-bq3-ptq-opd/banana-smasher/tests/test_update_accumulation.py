from __future__ import annotations

import pytest

from banana_smasher.update import _logical_segment_bounds


def test_logical_8192_window_is_exactly_eight_contiguous_1024_segments() -> None:
    bounds = _logical_segment_bounds(1024, 8)
    assert bounds == [(index * 1024, (index + 1) * 1024) for index in range(8)]
    assert bounds[0] == (0, 1024)
    assert bounds[-1] == (7168, 8192)


@pytest.mark.parametrize("tokens,segments", [(0, 8), (1025, 8), (1024, 0), (1024, 9)])
def test_logical_segment_geometry_fails_closed(tokens: int, segments: int) -> None:
    with pytest.raises(ValueError):
        _logical_segment_bounds(tokens, segments)