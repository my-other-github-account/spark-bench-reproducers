from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from typing import Any


def exact_accumulation_step(
    *,
    optimizer: Any,
    segments: Sequence[Any],
    item_count: Callable[[Any], int],
    loss_sum: Callable[[Any], Any],
) -> dict[str, Any]:
    """Accumulate one logical mean-loss update over physical segments.

    ``loss_sum`` must return the summed loss for a segment, not its mean. Scaling
    every segment by the total logical item count makes unequal final segments
    exact and performs exactly one optimizer mutation after all backward roots.
    """
    counts = [int(item_count(segment)) for segment in segments]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"accumulation segments must be non-empty, got {counts}")
    total = sum(counts)
    optimizer.zero_grad(set_to_none=True)
    detached_sum = 0.0
    for segment, count in zip(segments, counts, strict=True):
        current = loss_sum(segment)
        if getattr(current, "ndim", None) != 0:
            raise ValueError("loss_sum must return a scalar summed loss")
        detached_sum += float(current.detach())
        (current / total).backward()
    optimizer.step()
    return {
        "segments": len(segments),
        "segment_items": counts,
        "logical_items": total,
        "logical_mean_loss": detached_sum / total,
        "optimizer_steps": 1,
    }


def audit_split_vs_unsplit(
    *,
    segments: int = 8,
    items_per_segment: int = 7,
    seed: int = 1430,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Fit-shape audit of exact split accumulation against one unsplit update."""
    import torch
    from torch import nn

    if segments <= 1 or items_per_segment <= 0:
        raise ValueError("audit requires at least two non-empty segments")
    torch.manual_seed(seed)
    dtype = torch.float64
    reference = nn.Sequential(
        nn.Linear(9, 13, bias=True, dtype=dtype),
        nn.SiLU(),
        nn.Linear(13, 5, bias=True, dtype=dtype),
    )
    split = copy.deepcopy(reference)
    total = segments * items_per_segment
    features = torch.randn(total, 9, dtype=dtype)
    targets = torch.randn(total, 5, dtype=dtype)
    chunks = list(zip(features.chunk(segments), targets.chunk(segments), strict=True))

    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=3e-4)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_sum = (reference(features) - targets).square().sum()
    reference_mean = reference_sum / total
    reference_mean.backward()
    reference_gradients = [parameter.grad.detach().clone() for parameter in reference.parameters()]
    reference_optimizer.step()

    split_optimizer = torch.optim.Adam(split.parameters(), lr=3e-4)
    split_receipt = exact_accumulation_step(
        optimizer=split_optimizer,
        segments=chunks,
        item_count=lambda chunk: int(chunk[0].shape[0]),
        loss_sum=lambda chunk: (split(chunk[0]) - chunk[1]).square().sum(),
    )
    split_gradients = [
        parameter.grad.detach().clone() for parameter in split.parameters()
    ]
    gradient_max_abs = max(
        float((left - right).abs().max())
        for left, right in zip(reference_gradients, split_gradients, strict=True)
    )
    parameter_max_abs = max(
        float((left.detach() - right.detach()).abs().max())
        for left, right in zip(reference.parameters(), split.parameters(), strict=True)
    )
    loss_abs = abs(float(reference_mean.detach()) - split_receipt["logical_mean_loss"])
    passed = bool(
        loss_abs <= tolerance
        and gradient_max_abs <= tolerance
        and parameter_max_abs <= tolerance
        and split_receipt["optimizer_steps"] == 1
    )
    return {
        "schema": "banana-smasher-split-vs-unsplit-audit-v1",
        "status": "PASS" if passed else "FAIL",
        "dtype": str(dtype),
        "segments": segments,
        "items_per_segment": items_per_segment,
        "logical_items": total,
        "tolerance": tolerance,
        "loss_abs_error": loss_abs,
        "gradient_max_abs_error": gradient_max_abs,
        "post_step_parameter_max_abs_error": parameter_max_abs,
        "optimizer_steps": split_receipt["optimizer_steps"],
        "normalization": "sum_each_segment/divide_once_by_total_logical_items",
    }


def seal_audit_receipt(path: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    """Atomically create an immutable split-vs-unsplit audit receipt."""
    from pathlib import Path

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if target.exists():
        temporary.unlink()
        raise FileExistsError(f"immutable audit receipt already exists: {target}")
    os.replace(temporary, target)
    return {
        **receipt,
        "receipt": str(target),
        "receipt_sha256": hashlib.sha256(data).hexdigest(),
    }
