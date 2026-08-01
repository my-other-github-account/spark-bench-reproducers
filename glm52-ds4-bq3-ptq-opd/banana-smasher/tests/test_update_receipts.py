from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from banana_smasher.update_checkpoint import load_checkpoint
from banana_smasher.update_engine import run_segmented_update


def _run(root: Path, *, verbose: bool = False) -> dict:
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    segments = [torch.tensor([1.0, 2.0]), torch.tensor([4.0])]
    return run_segmented_update(
        parameters=[parameter],
        optimizer=optimizer,
        segments=segments,
        item_count=lambda values: int(values.numel()),
        loss_sum=lambda values: ((parameter - values) ** 2).sum(),
        output=root / "update.pt",
        receipt=root / "receipt.json",
        identity={"fixture": "receipt-v1"},
        verbose_receipts=verbose,
    )


def test_normal_receipt_is_concise_and_has_required_product_fields(tmp_path: Path) -> None:
    receipt = _run(tmp_path)
    assert receipt["schema"] == "banana-smasher-update-receipt-v3"
    assert receipt["status"] == "PASS_UPDATE"
    assert receipt["command"] == "update"
    assert receipt["backend"] == {
        "requested": "accelerated",
        "used": "accelerated",
    }
    assert receipt["fallback"] == {"used": False, "reason": None}
    assert receipt["logical_items"] == 3
    assert receipt["segments"] == 2
    assert receipt["completed_segments"] == 2
    assert receipt["optimizer_steps"] == 1
    assert receipt["output_artifact"] == str((tmp_path / "update.pt").resolve())
    assert receipt["receipt"] == str((tmp_path / "receipt.json").resolve())
    assert receipt["durable_completion"] is True
    assert "phase_seconds" not in receipt
    assert "parameter" not in receipt
    assert json.loads((tmp_path / "receipt.json").read_text()) == receipt


def test_verbose_receipt_adds_phases_diffs_and_fallback_details(tmp_path: Path) -> None:
    receipt = _run(tmp_path, verbose=True)
    assert len(receipt["phase_seconds"]["segments"]) == 2
    assert receipt["phase_seconds"]["optimizer"] >= 0
    assert receipt["parameter"]["sha256_before"] != receipt["parameter"]["sha256_after"]
    assert receipt["parameter"]["max_abs_diff"] > 0
    assert receipt["diffs"]["parameter_max_abs"] > 0
    assert receipt["diffs"]["parameter_sha256_changed"] is True
    assert receipt["fallback"] == {"used": False, "reason": None}
    assert receipt["backend"]["used"] == "accelerated"


def test_accelerated_failure_is_loud_and_does_not_publish_or_fallback(
    tmp_path: Path,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    reference_calls = 0

    def fail(_segment: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("accelerator unavailable")

    with pytest.raises(RuntimeError, match="accelerator unavailable"):
        run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=[torch.tensor([1.0])],
            item_count=lambda values: int(values.numel()),
            loss_sum=fail,
            backend="accelerated",
            output=tmp_path / "failed.pt",
            identity={"fixture": "fail-loud-v1", "reference_calls": reference_calls},
        )
    assert reference_calls == 0
    assert not (tmp_path / "failed.pt").exists()


def test_corrupt_checkpoint_payload_is_rejected(tmp_path: Path) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    def stop(_index: int, manifest: dict) -> None:
        Path(manifest["payload_path"]).write_bytes(b"truncated")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_segmented_update(
            parameters=[parameter],
            optimizer=optimizer,
            segments=[torch.tensor([1.0]), torch.tensor([2.0])],
            item_count=lambda values: int(values.numel()),
            loss_sum=lambda values: ((parameter - values) ** 2).sum(),
            output=tmp_path / "corrupt.pt",
            identity={"fixture": "corrupt-v1"},
            on_segment_committed=stop,
        )
    with pytest.raises(RuntimeError, match="checkpoint payload (byte count|SHA-256) mismatch"):
        load_checkpoint(
            (tmp_path / "corrupt.pt.checkpoint"),
            expected_identity={"fixture": "corrupt-v1"},
            expected_backend="accelerated",
        )
