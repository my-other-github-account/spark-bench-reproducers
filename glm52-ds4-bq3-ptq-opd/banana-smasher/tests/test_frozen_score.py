from __future__ import annotations

import pytest
import torch

from banana_smasher.frozen_score import reference_frozen_weighted_errors


def _fixture() -> tuple[torch.Tensor, ...]:
    weights = torch.tensor([[0.0] * 32, [1.0] * 32], dtype=torch.bfloat16)
    h = torch.ones((32,), dtype=torch.float32)
    codes = torch.zeros((2, 2, 8), dtype=torch.int16)
    scales = torch.full((2, 2, 1), 127, dtype=torch.uint8)
    codebooks = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, 1], dtype=torch.int64)
    return weights, h, codes, scales, codebooks, offsets


def test_reference_frozen_bucket_scores_full_declared_options() -> None:
    scores = reference_frozen_weighted_errors(*_fixture(), vector_width=4)

    assert scores.tolist() == [32.0, 32.0]
    assert int(scores.argmin()) == 0


def test_frozen_bucket_rejects_out_of_range_native_reads() -> None:
    values = list(_fixture())
    values[2][1, 0, 0] = 1

    with pytest.raises(ValueError, match="code range"):
        reference_frozen_weighted_errors(*values, vector_width=4)


def test_frozen_bucket_rejects_nonfinite_semantic_inputs() -> None:
    values = list(_fixture())
    values[1][0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        reference_frozen_weighted_errors(*values, vector_width=4)
