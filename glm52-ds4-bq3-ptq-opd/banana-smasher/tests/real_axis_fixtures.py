from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from banana_smasher.durability import sha256_file


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_array(root: Path, relative: str, value: np.ndarray) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }


def write_runtime(
    root: Path,
    *,
    instance_id: str,
    layer_count: int,
    layer_scale: float = 1.0,
    head_shift: float = 0.0,
    divergent_layer: int | None = None,
    pack: bool,
) -> Path:
    hidden = 4
    layers: list[dict[str, Any]] = []
    for layer in range(layer_count):
        weight = np.eye(hidden, dtype=np.float32) * np.float32(
            layer_scale + layer * 0.001
        )
        bias = np.full(hidden, layer * 0.0001, dtype=np.float32)
        layers.append(
            {
                "layer": layer,
                "activation": "tanh",
                "descriptor": {
                    "source_shard": (
                        f"divergent-shard-{layer}"
                        if layer == divergent_layer
                        else f"layer-shard-{layer}"
                    ),
                    "overlay_experts": layer + 1,
                    "native_fallback_experts": layer % 3,
                },
                "weight": _write_array(root, f"layers/{layer:03d}/weight.npy", weight),
                "bias": _write_array(root, f"layers/{layer:03d}/bias.npy", bias),
            }
        )
    head_weight = (
        np.arange(hidden * 8, dtype=np.float32).reshape(hidden, 8) / 31.0
    )
    head_weight = head_weight + np.float32(head_shift) * np.linspace(
        -1.0, 1.0, 8, dtype=np.float32
    )[None, :]
    head_bias = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
    _write_json(
        root / "real_axis.json",
        {
            "schema": "bs-real-axis-runtime-v1",
            "schema_version": 1,
            "model_id": "fixture-real-axis-model",
            "layers": layers,
            "head": {
                "weight": _write_array(root, "head/weight.npy", head_weight),
                "bias": _write_array(root, "head/bias.npy", head_bias),
            },
        },
    )
    if pack:
        _write_json(
            root / "BANANA_PACK_MANIFEST.json",
            {
                "schema": "bs-pack",
                "schema_version": 1,
                "instance_id": instance_id,
                "model_id": "fixture-real-axis-model",
            },
        )
    return root


def write_population(root: Path) -> tuple[Path, Path]:
    corpus = root / "corpus"
    rows: list[dict[str, Any]] = []
    values = (
        np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [-0.2, 0.1, 0.5, -0.3],
                [0.7, -0.4, 0.2, 0.1],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [-0.1, 0.5, 0.2, 0.0],
                [0.3, -0.3, 0.6, 0.2],
                [0.8, 0.1, -0.4, 0.3],
                [0.2, 0.4, -0.1, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    for ordinal, value in enumerate(values):
        descriptor = _write_array(corpus, f"window_{ordinal}.npy", value)
        rows.append(
            {
                "window_id": 100 + ordinal * 7,
                "class": "reasoning" if ordinal == 0 else "code",
                **descriptor,
            }
        )
    manifest = root / "windows.json"
    _write_json(
        manifest,
        {
            "schema": "bs-real-axis-windows-v1",
            "schema_version": 1,
            "corpus_id": "fixture-corpus",
            "windows": rows,
        },
    )
    return corpus, manifest


def write_instrument(root: Path) -> Path:
    return _write_json(
        root / "instrument.json",
        {
            "schema": "bs-real-axis-instrument-v1",
            "schema_version": 1,
            "profile": "fixture-real-axis-v1",
            "teacher_storage": "top_support_logprob",
            "support": 4,
            "cutoff": 4,
            "direction": "kl_teacher_candidate",
            "attention": "eager",
            "estimator": "position_weighted_mean",
        },
    )


def real_axis_fixture(tmp_path: Path, *, layer_count: int = 3) -> dict[str, Path]:
    model = write_runtime(
        tmp_path / "model",
        instance_id="native",
        layer_count=layer_count,
        pack=False,
    )
    candidate = write_runtime(
        tmp_path / "candidate",
        instance_id="candidate-pack",
        layer_count=layer_count,
        layer_scale=0.999,
        head_shift=0.01,
        divergent_layer=12 if layer_count > 12 else None,
        pack=True,
    )
    reference = write_runtime(
        tmp_path / "reference",
        instance_id="reference-pack",
        layer_count=layer_count,
        layer_scale=0.97,
        head_shift=0.08,
        divergent_layer=None,
        pack=True,
    )
    corpus, windows = write_population(tmp_path)
    return {
        "model": model,
        "candidate": candidate,
        "reference": reference,
        "corpus": corpus,
        "windows": windows,
        "instrument": write_instrument(tmp_path),
        "bank": tmp_path / "bank",
        "evaluation": tmp_path / "evaluation",
    }
