from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch

from banana_smasher.cli import _parser
from banana_smasher.update import (
    _backend_environment,
    _validate_runtime_components,
    run_minimal_update,
)
from banana_smasher.update_engine import run_segmented_update


def _run_fixture(root: Path, backend: str) -> tuple[dict, torch.Tensor, list[torch.Tensor]]:
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=3e-4)
    segments = [
        torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64),
        torch.tensor([[0.5, 0.25]], dtype=torch.float64),
    ]
    selected: list[torch.Tensor] = []

    def loss_sum(values: torch.Tensor) -> torch.Tensor:
        expected = "1" if backend == "accelerated" else "0"
        assert os.environ["GENESIS_REPAIR_KMAJOR_10X"] == expected
        output = values @ parameter
        selected.append(output.detach().clone())
        return output.square().sum()

    with _backend_environment(backend):
        receipt = run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=segments,
            item_count=lambda values: int(values.shape[0]),
            loss_sum=loss_sum,
            backend=backend,
            output=root / f"{backend}.pt",
            receipt=root / f"{backend}.json",
            identity={"fixture": "backend-parity-v1"},
        )
    return receipt, parameter.detach().clone(), selected


def test_accelerated_is_the_shipped_default_and_reference_is_hidden() -> None:
    assert inspect.signature(run_minimal_update).parameters["backend"].default == "accelerated"
    parser = _parser()
    args = parser.parse_args(
        ["update", "--model-root", "/model", "--aot", "/aot.so", "--output", "/out.pt"]
    )
    assert args.backend == "accelerated"
    assert args.layers == 43
    assert args.segments == 8
    assert args.tokens == 1024
    assert "--backend" not in parser.format_help()


def test_accelerated_and_reference_backends_match_one_exact_step(tmp_path: Path) -> None:
    accelerated, accelerated_parameter, accelerated_outputs = _run_fixture(
        tmp_path, "accelerated"
    )
    reference, reference_parameter, reference_outputs = _run_fixture(tmp_path, "reference")

    assert accelerated["backend"] == {
        "requested": "accelerated",
        "used": "accelerated",
    }
    assert accelerated["fallback"] == {"used": False, "reason": None}
    assert reference["backend"]["used"] == "reference"
    assert accelerated["optimizer_steps"] == reference["optimizer_steps"] == 1
    assert accelerated["logical_items"] == reference["logical_items"] == 3
    assert len(accelerated_outputs) == len(reference_outputs) == 2
    for left, right in zip(accelerated_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        accelerated_parameter, reference_parameter, rtol=0, atol=1e-12
    )


def test_backend_environment_is_scoped_and_never_falls_back() -> None:
    os.environ["GENESIS_REPAIR_KMAJOR_10X"] = "outer"
    with _backend_environment("accelerated"):
        assert os.environ["GENESIS_REPAIR_KMAJOR_10X"] == "1"
    assert os.environ["GENESIS_REPAIR_KMAJOR_10X"] == "outer"
    with pytest.raises(ValueError, match="unsupported update backend"):
        with _backend_environment("automatic"):
            pass


def test_missing_acceleration_runtime_fails_with_named_components(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="banana-smasher update runtime is incomplete") as caught:
        _validate_runtime_components(tmp_path)
    assert "banana_smasher_physical_surface.py" in str(caught.value)
    assert "kmajor_autograd.py" in str(caught.value)
    assert "install the update extra" in str(caught.value)
