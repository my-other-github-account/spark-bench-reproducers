from __future__ import annotations

import pytest
import torch

from banana_smasher.accumulation import (
    audit_split_vs_unsplit,
    exact_accumulation_step,
)


def test_eight_way_split_matches_unsplit_loss_gradient_and_step() -> None:
    receipt = audit_split_vs_unsplit(segments=8, items_per_segment=7)
    assert receipt["status"] == "PASS"
    assert receipt["optimizer_steps"] == 1
    assert receipt["loss_abs_error"] <= receipt["tolerance"]
    assert receipt["gradient_max_abs_error"] <= receipt["tolerance"]
    assert receipt["post_step_parameter_max_abs_error"] <= receipt["tolerance"]


def test_unequal_final_segment_uses_total_item_weighting() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    segments = [torch.tensor([1.0, 2.0]), torch.tensor([4.0])]
    receipt = exact_accumulation_step(
        optimizer=optimizer,
        segments=segments,
        item_count=lambda values: values.numel(),
        loss_sum=lambda values: ((parameter - values) ** 2).sum(),
    )
    # Mean gradient at p=2 is 2 * mean([1,0,-2]) = -2/3.
    assert parameter.item() == pytest.approx(2.0 + 0.1 * 2.0 / 3.0)
    assert receipt["segment_items"] == [2, 1]
    assert receipt["optimizer_steps"] == 1


def test_accumulation_rejects_empty_segments() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(ValueError, match="non-empty"):
        exact_accumulation_step(
            optimizer=optimizer,
            segments=[torch.empty(0)],
            item_count=lambda values: values.numel(),
            loss_sum=lambda values: parameter * values.sum(),
        )
