from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .accumulation import backward_logical_mean
from .update_checkpoint import (
    atomic_json,
    atomic_torch_save,
    commit_segment_checkpoint,
    finalize_checkpoint,
    load_checkpoint,
)


def _tensor_bytes(value: Any) -> bytes:
    tensor = value.detach().cpu().contiguous()
    try:
        return tensor.numpy().tobytes()
    except TypeError:
        return tensor.view(-1).view(__import__("torch").uint8).numpy().tobytes()


def _parameter_sha256(parameters: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def _cpu_parameters(parameters: Sequence[Any]) -> list[Any]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _cpu_gradients(parameters: Sequence[Any]) -> list[Any | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for parameter in parameters
    ]


def _restore_parameters(parameters: Sequence[Any], values: Sequence[Any]) -> None:
    import torch

    if len(parameters) != len(values):
        raise RuntimeError("checkpoint trainable parameter count mismatch")
    with torch.no_grad():
        for parameter, value in zip(parameters, values, strict=True):
            if tuple(parameter.shape) != tuple(value.shape) or parameter.dtype != value.dtype:
                raise RuntimeError("checkpoint trainable parameter identity mismatch")
            parameter.copy_(value.to(device=parameter.device))


def _restore_gradients(parameters: Sequence[Any], gradients: Sequence[Any | None]) -> None:
    if len(parameters) != len(gradients):
        raise RuntimeError("checkpoint gradient count mismatch")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = (
            None
            if gradient is None
            else gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
        )


def _rng_state(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(torch: Any, state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    run_id: str,
    next_segment_index: int,
    detached_loss_sum: float,
    base_parameters: Sequence[Any],
    parameters: Sequence[Any],
    optimizer: Any,
    optimizer_steps: int,
    phase_rows: list[dict[str, Any]],
    torch: Any,
    state: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "state": state,
        "next_segment_index": int(next_segment_index),
        "completed_segments": list(range(int(next_segment_index))),
        "detached_loss_sum": float(detached_loss_sum),
        "base_parameters": list(base_parameters),
        "parameters": _cpu_parameters(parameters),
        "gradients": _cpu_gradients(parameters),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_steps": int(optimizer_steps),
        "rng_state": _rng_state(torch),
        "phase_rows": phase_rows,
    }


def run_segmented_update(
    *,
    parameters: Sequence[Any],
    optimizer: Any,
    segments: Sequence[Any],
    item_count: Callable[[Any], int],
    loss_sum: Callable[[Any], Any],
    output: str | Path,
    receipt: str | Path | None = None,
    identity: dict[str, Any] | None = None,
    backend: str = "accelerated",
    resume: bool = True,
    restart: bool = False,
    verbose_receipts: bool = False,
    synchronize: Callable[[], None] | None = None,
    on_segment_committed: Callable[[int, dict[str, Any]], None] | None = None,
    receipt_fields: dict[str, Any] | None = None,
    post_step_validate: Callable[[], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Execute one exact logical-mean optimizer update with durable segment state."""
    import torch

    if backend not in {"accelerated", "reference"}:
        raise ValueError(f"unsupported update backend {backend!r}")
    values = list(parameters)
    work = list(segments)
    counts = [int(item_count(segment)) for segment in work]
    if not values:
        raise ValueError("update requires at least one trainable parameter")
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"accumulation segments must be non-empty, got {counts}")
    logical_items = sum(counts)
    output_path = Path(output).resolve()
    receipt_path = (
        Path(receipt).resolve()
        if receipt is not None
        else output_path.with_name(f"{output_path.name}.receipt.json")
    )
    checkpoint_dir = Path(f"{output_path}.checkpoint")
    raw_identity: dict[str, Any] = {} if identity is None else identity
    try:
        canonical_identity = json.loads(
            json.dumps(raw_identity, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"update identity must be canonical JSON: {exc}") from exc
    if not isinstance(canonical_identity, dict):  # pragma: no cover - typed API guard
        raise ValueError("update identity must be a JSON object")
    identity = canonical_identity
    synchronize = (lambda: None) if synchronize is None else synchronize
    payload: dict[str, Any] | None = None

    if restart:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        output_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    elif checkpoint_dir.is_dir():
        payload, manifest = load_checkpoint(
            checkpoint_dir,
            expected_identity=identity,
            expected_backend=backend,
            expected_segment_plan=counts,
        )
        if manifest["status"] == "COMPLETE":
            if not output_path.is_file() or not receipt_path.is_file():
                raise RuntimeError("completed update checkpoint is missing its artifact or receipt")
            output_record = manifest["output"]
            if output_path.stat().st_size != int(output_record["bytes"]):
                raise RuntimeError("completed update artifact byte count mismatch")
            from .update_checkpoint import _sha256

            if _sha256(output_path) != output_record["sha256"]:
                raise RuntimeError("completed update artifact SHA-256 mismatch")
            return json.loads(receipt_path.read_text())
        if not resume:
            raise RuntimeError("incomplete update checkpoint exists; use resume or restart")
    elif output_path.exists() or receipt_path.exists():
        raise RuntimeError("update output exists without a completion checkpoint; use restart")

    started = time.perf_counter()
    if payload is None:
        run_id = uuid.uuid4().hex
        base_parameters = _cpu_parameters(values)
        start_index = 0
        detached_loss_sum = 0.0
        phase_rows: list[dict[str, Any]] = []
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
    else:
        run_id = str(payload["run_id"])
        base_parameters = payload["base_parameters"]
        _restore_parameters(values, payload["parameters"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _restore_gradients(values, payload["gradients"])
        _restore_rng(torch, payload["rng_state"])
        start_index = int(payload["next_segment_index"])
        detached_loss_sum = float(payload["detached_loss_sum"])
        phase_rows = list(payload["phase_rows"])
        optimizer_steps = int(payload["optimizer_steps"])
        if optimizer_steps not in {0, 1}:
            raise RuntimeError(f"checkpoint optimizer step count is invalid: {optimizer_steps}")

    for index in range(start_index, len(work)):
        forward_started = time.perf_counter()
        current = loss_sum(work[index])
        synchronize()
        forward_seconds = time.perf_counter() - forward_started
        if getattr(current, "ndim", None) != 0:
            raise ValueError("loss_sum must return a scalar summed loss")
        if not bool(torch.isfinite(current)):
            raise RuntimeError(f"non-finite update loss at segment {index}")
        detached_loss_sum += float(current.detach())
        backward_started = time.perf_counter()
        backward_logical_mean(current, logical_items)
        synchronize()
        backward_seconds = time.perf_counter() - backward_started
        phase_rows.append(
            {
                "segment_index": index,
                "items": counts[index],
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "loss_sum": float(current.detach()),
            }
        )
        checkpoint_payload = _checkpoint_payload(
            run_id=run_id,
            next_segment_index=index + 1,
            detached_loss_sum=detached_loss_sum,
            base_parameters=base_parameters,
            parameters=values,
            optimizer=optimizer,
            optimizer_steps=optimizer_steps,
            phase_rows=phase_rows,
            torch=torch,
            state="optimizer_pending" if index + 1 == len(work) else "accumulating",
        )
        manifest = commit_segment_checkpoint(
            checkpoint_dir,
            checkpoint_payload,
            identity=identity,
            backend=backend,
            segment_plan=counts,
        )
        if on_segment_committed is not None:
            on_segment_committed(index, manifest)

    gradients = [parameter.grad for parameter in values if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("update produced missing or non-finite gradients")
    optimizer_seconds = 0.0
    if optimizer_steps == 0:
        optimizer_started = time.perf_counter()
        optimizer.step()
        synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        optimizer_steps = 1
        commit_segment_checkpoint(
            checkpoint_dir,
            _checkpoint_payload(
                run_id=run_id,
                next_segment_index=len(work),
                detached_loss_sum=detached_loss_sum,
                base_parameters=base_parameters,
                parameters=values,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                phase_rows=phase_rows,
                torch=torch,
                state="optimizer_done",
            ),
            identity=identity,
            backend=backend,
            segment_plan=counts,
        )
    if optimizer_steps != 1:
        raise RuntimeError(f"update requires exactly one optimizer step, got {optimizer_steps}")

    before_sha = _parameter_sha256(base_parameters)
    after_sha = _parameter_sha256(values)
    max_abs_diff = max(
        float((after.detach().cpu() - before).abs().max())
        for before, after in zip(base_parameters, values, strict=True)
    )
    if before_sha == after_sha or not math.isfinite(max_abs_diff) or max_abs_diff <= 0:
        raise RuntimeError("optimizer step did not produce a finite parameter mutation")
    validated_fields = post_step_validate() if post_step_validate is not None else None

    artifact = {
        "schema": "banana-smasher-update-artifact-v1",
        "backend": backend,
        "identity": identity,
        "logical_items": logical_items,
        "segments": len(work),
        "optimizer_steps": optimizer_steps,
        "parameters": _cpu_parameters(values),
        "optimizer_state": optimizer.state_dict(),
    }
    output_record = atomic_torch_save(output_path, artifact)
    result: dict[str, Any] = {
        "schema": "banana-smasher-update-receipt-v3",
        "status": "PASS_UPDATE",
        "command": "update",
        "backend": {"requested": backend, "used": backend},
        "fallback": {"used": False, "reason": None},
        "logical_items": logical_items,
        "segments": len(work),
        "segment_items": counts,
        "completed_segments": len(work),
        "resumed_segments": start_index,
        "optimizer_steps": optimizer_steps,
        "logical_mean_loss": detached_loss_sum / logical_items,
        "total_wall_seconds": time.perf_counter() - started,
        "output_artifact": str(output_path),
        "receipt": str(receipt_path),
        "durable_completion": True,
    }
    if receipt_fields:
        result.update(receipt_fields)
    if validated_fields:
        result.update(validated_fields)
    if verbose_receipts:
        result.update(
            {
                "phase_seconds": {
                    "segments": phase_rows,
                    "forward": sum(row["forward_seconds"] for row in phase_rows),
                    "backward": sum(row["backward_seconds"] for row in phase_rows),
                    "optimizer": optimizer_seconds,
                },
                "loss": {
                    "sum": detached_loss_sum,
                    "mean": detached_loss_sum / logical_items,
                    "finite": math.isfinite(detached_loss_sum),
                },
                "gradients": {"tensors": len(gradients), "finite": True},
                "parameter": {
                    "sha256_before": before_sha,
                    "sha256_after": after_sha,
                    "max_abs_diff": max_abs_diff,
                },
                "diffs": {
                    "parameter_max_abs": max_abs_diff,
                    "parameter_sha256_changed": before_sha != after_sha,
                },
                "output": output_record,
            }
        )
    atomic_json(receipt_path, result)
    finalize_checkpoint(
        checkpoint_dir,
        receipt=receipt_path,
        output_record=output_record,
    )
    return result
