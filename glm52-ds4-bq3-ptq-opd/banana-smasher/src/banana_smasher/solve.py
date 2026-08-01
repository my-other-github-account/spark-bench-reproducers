from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class SolveInputError(ValueError):
    """The public solve-input bundle is incomplete or inconsistent."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bundle_path(root: Path, raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SolveInputError(f"{field} must be a non-empty relative path")
    candidate = (root / raw).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise SolveInputError(f"{field} escapes the solve-input root")
    if not candidate.is_file():
        raise SolveInputError(f"{field} is missing: {candidate}")
    return candidate


def _load_matrix(path: Path, *, field: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise SolveInputError(f"{field} must be one rank-2 NPY array")
    if value.shape[1] != 4:
        raise SolveInputError(f"{field} must use the exact D=4 geometry")
    if value.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise SolveInputError(f"{field} must have a floating dtype")
    if not np.isfinite(value).all():
        raise SolveInputError(f"{field} contains NaN or infinity")
    return np.ascontiguousarray(value, dtype=np.float32)


def _load_manifest(root: Path) -> tuple[dict[str, Any], str]:
    manifest_path = root / "solve.json"
    if not manifest_path.is_file():
        raise SolveInputError(f"solve-input manifest is missing: {manifest_path}")
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if not isinstance(manifest, dict):
        raise SolveInputError("solve.json must contain one JSON object")
    if manifest.get("schema") != "banana-smasher-solve-input-v1":
        raise SolveInputError(f"unsupported solve-input schema: {manifest.get('schema')!r}")
    if not isinstance(manifest.get("layer"), int) or int(manifest["layer"]) < 0:
        raise SolveInputError("solve-input layer must be a non-negative integer")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SolveInputError("solve-input cells must be a non-empty array")
    return manifest, hashlib.sha256(payload).hexdigest()


def run_solve(
    source_root: str | Path,
    output: str | Path,
    *,
    device: str = "cuda",
    reference_search: bool = False,
    verbose_receipts: bool = False,
) -> dict[str, Any]:
    """Run exact full-codebook search for every declared solve cell.

    The ordinary path requires CUDA plus Triton and never silently falls back.
    The exhaustive implementation is an explicit hidden developer/CI mode.
    """

    import torch

    from . import exact_codebook

    source_root = Path(source_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source_root.is_dir():
        raise SolveInputError(f"solve-input root is missing: {source_root}")
    if output.exists():
        raise FileExistsError(output)

    manifest, manifest_sha256 = _load_manifest(source_root)
    backend = "reference-search" if reference_search else "exact-gemm"
    if not reference_search:
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("exact-gemm requires a CUDA device; no reference fallback is performed")
        if exact_codebook.triton is None:
            raise RuntimeError(
                "exact-gemm requires Triton; install banana-smasher[solve] on a supported CUDA host"
            )
    output.mkdir(parents=True)
    _fsync_directory(output.parent)

    started = time.perf_counter()
    winners_by_cell: dict[str, np.ndarray] = {}
    bucket_scores_by_cell: dict[str, np.ndarray] = {}
    bucket_rows: list[dict[str, Any]] = []
    verbose_cells: list[dict[str, Any]] = []
    candidate_count: int | None = None
    total_rows = 0
    seen_cells: set[str] = set()

    for index, raw_cell in enumerate(manifest["cells"]):
        if not isinstance(raw_cell, dict):
            raise SolveInputError(f"cells[{index}] must be an object")
        cell = raw_cell.get("cell")
        if not isinstance(cell, str) or not cell or cell in seen_cells:
            raise SolveInputError(f"cells[{index}].cell must be a unique non-empty string")
        seen_cells.add(cell)
        vectors_array = _load_matrix(
            _bundle_path(source_root, raw_cell.get("vectors"), field=f"cells[{index}].vectors"),
            field=f"cells[{index}].vectors",
        )
        codebook_array = _load_matrix(
            _bundle_path(source_root, raw_cell.get("codebook"), field=f"cells[{index}].codebook"),
            field=f"cells[{index}].codebook",
        )
        if codebook_array.shape[0] < 2:
            raise SolveInputError(f"cells[{index}].codebook needs at least two candidates")
        if candidate_count is None:
            candidate_count = int(codebook_array.shape[0])
        elif candidate_count != int(codebook_array.shape[0]):
            raise SolveInputError("all solve cells must use one common candidate count")
        if not reference_search and candidate_count % 64:
            raise SolveInputError("exact-gemm candidate count must be divisible by 64")

        vectors = torch.from_numpy(vectors_array).to(device)
        codebook = torch.from_numpy(codebook_array).to(device)
        if reference_search:
            winners = exact_codebook.exhaustive_reference_winners(vectors, codebook)
            details: dict[str, Any] = {"rows": int(vectors.shape[0])}
        else:
            winners, details = exact_codebook.exact_codebook_winners(vectors, codebook)
        winners_by_cell[cell] = winners.detach().cpu().numpy().astype(np.int64, copy=False)

        frozen_bucket = raw_cell.get("frozen_bucket")
        if frozen_bucket is not None:
            from . import frozen_score

            if not isinstance(frozen_bucket, dict):
                raise SolveInputError(f"cells[{index}].frozen_bucket must be an object")

            def load_bucket_array(field: str) -> np.ndarray:
                path = _bundle_path(
                    source_root,
                    frozen_bucket.get(field),
                    field=f"cells[{index}].frozen_bucket.{field}",
                )
                value = np.load(path, allow_pickle=False)
                if not isinstance(value, np.ndarray):
                    raise SolveInputError(
                        f"cells[{index}].frozen_bucket.{field} must be one NPY array"
                    )
                return np.ascontiguousarray(value)

            options = frozen_bucket.get("options")
            vector_width = frozen_bucket.get("vector_width")
            if not isinstance(options, list) or not options or not all(
                isinstance(option, str) and option for option in options
            ):
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket.options must be non-empty strings"
                )
            if len(set(options)) != len(options):
                raise SolveInputError(f"cells[{index}].frozen_bucket.options must be unique")
            if not isinstance(vector_width, int) or vector_width < 1:
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket.vector_width must be positive"
                )

            weights = torch.from_numpy(load_bucket_array("weights")).to(
                device=device, dtype=torch.bfloat16
            )
            h = torch.from_numpy(load_bucket_array("h")).to(
                device=device, dtype=torch.float32
            )
            codes = torch.from_numpy(load_bucket_array("codes")).to(device)
            scales = torch.from_numpy(load_bucket_array("scales")).to(device)
            bucket_codebooks = torch.from_numpy(load_bucket_array("codebooks")).to(
                device=device, dtype=torch.float32
            )
            offsets = torch.from_numpy(load_bucket_array("codebook_offsets")).to(device)
            if int(codes.shape[0]) != len(options):
                raise SolveInputError(
                    f"cells[{index}].frozen_bucket options/codes count mismatch"
                )
            scorer = (
                frozen_score.reference_frozen_weighted_errors
                if reference_search
                else frozen_score.fused_frozen_weighted_errors
            )
            bucket_scores = scorer(
                weights,
                h,
                codes,
                scales,
                bucket_codebooks,
                offsets,
                vector_width=vector_width,
            )
            bucket_score_array = (
                bucket_scores.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            winner_index = int(bucket_scores.argmin())
            bucket_scores_by_cell[cell] = bucket_score_array
            bucket_rows.append(
                {
                    "cell": cell,
                    "options": options,
                    "winner_index": winner_index,
                    "winner": options[winner_index],
                }
            )

        total_rows += int(vectors.shape[0])
        verbose_cells.append({"cell": cell, **details})

    artifact = output / "winners.npz"
    _atomic_npz(artifact, winners_by_cell)
    bucket_artifact: Path | None = None
    if bucket_scores_by_cell:
        bucket_artifact = output / "bucket_scores.npz"
        _atomic_npz(bucket_artifact, bucket_scores_by_cell)
    elapsed = time.perf_counter() - started
    receipt_path = output / "SOLVE_RECEIPT.json"
    receipt: dict[str, Any] = {
        "schema": "banana-smasher-solve-receipt-v1",
        "status": "PASS",
        "command": "solve",
        "backend": backend,
        "layer": int(manifest["layer"]),
        "shape": {
            "cells": len(winners_by_cell),
            "rows": total_rows,
            "candidates": int(candidate_count or 0),
        },
        "elapsed_seconds": elapsed,
        "artifact": str(artifact),
        "receipt": str(receipt_path),
    }
    if bucket_artifact is not None:
        receipt["bucket_artifact"] = str(bucket_artifact)
        receipt["buckets"] = bucket_rows
    if verbose_receipts:
        receipt["verbose"] = {
            "input_manifest_sha256": manifest_sha256,
            "cells": verbose_cells,
        }
    _atomic_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "command": receipt["command"],
        "backend": receipt["backend"],
        "elapsed_seconds": receipt["elapsed_seconds"],
        "artifact": receipt["artifact"],
        "receipt": receipt["receipt"],
    }
