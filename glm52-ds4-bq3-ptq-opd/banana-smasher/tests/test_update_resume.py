from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest
import torch

from banana_smasher.update_engine import run_segmented_update


def _segments() -> list[torch.Tensor]:
    return [
        torch.tensor([float(index), float(index) + 0.5], dtype=torch.float64)
        for index in range(8)
    ]


def _run_update(output: Path, *, stop_after: int | None = None) -> dict:
    parameter = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1e-2)

    def loss_sum(values: torch.Tensor) -> torch.Tensor:
        return ((parameter - values) ** 2).sum()

    def committed(index: int, _manifest: dict) -> None:
        if stop_after == index:
            os._exit(91)

    return run_segmented_update(
        parameters=[parameter],
        optimizer=optimizer,
        segments=_segments(),
        item_count=lambda values: int(values.numel()),
        loss_sum=loss_sum,
        output=output,
        receipt=output.with_suffix(".json"),
        identity={
            "fixture": "eight-segment-real-interrupt-v1",
            "source_plan": [(27, 16)],
        },
        on_segment_committed=committed,
    )


def _interrupted_child(output: str) -> None:
    _run_update(Path(output), stop_after=3)


def test_real_process_interrupt_resumes_eight_segments_with_one_optimizer_step(
    tmp_path: Path,
) -> None:
    interrupted_output = tmp_path / "resumed.pt"
    process = multiprocessing.get_context("spawn").Process(
        target=_interrupted_child, args=(str(interrupted_output),)
    )
    process.start()
    process.join(60)
    assert process.exitcode == 91
    assert not interrupted_output.exists()

    resumed = _run_update(interrupted_output)
    _run_update(tmp_path / "uninterrupted.pt")

    assert resumed["segments"] == 8
    assert resumed["completed_segments"] == 8
    assert resumed["resumed_segments"] == 4
    assert resumed["optimizer_steps"] == 1
    resumed_artifact = torch.load(interrupted_output, weights_only=False)
    uninterrupted_artifact = torch.load(tmp_path / "uninterrupted.pt", weights_only=False)
    torch.testing.assert_close(
        resumed_artifact["parameters"][0],
        uninterrupted_artifact["parameters"][0],
        rtol=0,
        atol=1e-12,
    )
    assert int(resumed_artifact["optimizer_state"]["state"][0]["step"]) == 1


def test_completed_update_is_idempotent_and_does_not_run_forward_again(
    tmp_path: Path,
) -> None:
    output = tmp_path / "complete.pt"
    first = _run_update(output)

    parameter = torch.nn.Parameter(torch.tensor(99.0, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1e-2)

    def forbidden(_values: torch.Tensor) -> torch.Tensor:
        raise AssertionError("idempotent replay executed forward")

    replay = run_segmented_update(
        parameters=[parameter],
        optimizer=optimizer,
        segments=_segments(),
        item_count=lambda values: int(values.numel()),
        loss_sum=forbidden,
        output=output,
        receipt=output.with_suffix(".json"),
        identity={
            "fixture": "eight-segment-real-interrupt-v1",
            "source_plan": [(27, 16)],
        },
    )
    assert replay == first
    assert replay["durable_completion"] is True


def test_resume_rejects_identity_drift(tmp_path: Path) -> None:
    output = tmp_path / "drift.pt"
    process = multiprocessing.get_context("spawn").Process(
        target=_interrupted_child, args=(str(output),)
    )
    process.start()
    process.join(60)
    assert process.exitcode == 91

    parameter = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1e-2)
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=_segments(),
            item_count=lambda values: int(values.numel()),
            loss_sum=lambda values: ((parameter - values) ** 2).sum(),
            output=output,
            identity={"fixture": "changed"},
        )
