from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .durability import (
    canonical_sha256,
    load_json_object,
    safe_relative_path,
    sha256_file,
)
from .metrics import MetricsError, require_finite_float

REAL_AXIS_SCHEMA = "bs-real-axis-runtime-v1"
WINDOWS_SCHEMA = "bs-real-axis-windows-v1"
INSTRUMENT_SCHEMA = "bs-real-axis-instrument-v1"


class RealAxisError(ValueError):
    """Raised when a manifest-driven physical walk cannot be admitted."""


def default_instrument_profile_path() -> Path:
    return Path(__file__).with_name("profiles") / "real-axis-v1.json"


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RealAxisError(f"{label}_INVALID")
    return value


def _declared_array(
    root: Path,
    row: Any,
    *,
    label: str,
    rank: int | None = None,
) -> np.ndarray[Any, Any]:
    if not isinstance(row, dict):
        raise RealAxisError(f"{label}_DESCRIPTOR_INVALID")
    path = safe_relative_path(root, _require_string(row.get("path"), label=label), label=label)
    if path.stat().st_size != row.get("bytes"):
        raise RealAxisError(f"{label}_BYTES_MISMATCH: {path}")
    if sha256_file(path) != row.get("sha256"):
        raise RealAxisError(f"{label}_SHA256_MISMATCH: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RealAxisError(f"{label}_ARRAY_INVALID: {path}: {exc}") from exc
    if list(array.shape) != row.get("shape") or array.dtype.str != row.get("dtype"):
        raise RealAxisError(f"{label}_SCHEMA_MISMATCH: {path}")
    try:
        return require_finite_float(array, label=label, rank=rank)
    except MetricsError as exc:
        raise RealAxisError(str(exc)) from exc


def load_instrument_profile(path: str | Path | None = None) -> dict[str, Any]:
    profile_path = default_instrument_profile_path() if path is None else Path(path)
    if profile_path.is_symlink() or not profile_path.is_file():
        raise RealAxisError(f"INSTRUMENT_PROFILE_INVALID: {profile_path}")
    value = load_json_object(profile_path, label="INSTRUMENT_PROFILE")
    if value.get("schema") != INSTRUMENT_SCHEMA or value.get("schema_version") != 1:
        raise RealAxisError("INSTRUMENT_PROFILE_SCHEMA_MISMATCH")
    for field in ("profile", "teacher_storage", "direction", "attention", "estimator"):
        _require_string(value.get(field), label=f"INSTRUMENT_{field.upper()}")
    if value["direction"] != "kl_teacher_candidate":
        raise RealAxisError("INSTRUMENT_DIRECTION_UNSUPPORTED")
    for field in ("support", "cutoff"):
        if not isinstance(value.get(field), int) or value[field] <= 0:
            raise RealAxisError(f"INSTRUMENT_{field.upper()}_INVALID")
    return {
        **value,
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
    }


@dataclass(frozen=True)
class Window:
    ordinal: int
    window_id: str | int
    class_name: str
    path: Path
    descriptor: dict[str, Any]


class WindowPopulation:
    def __init__(self, *, corpus: str | Path, manifest: str | Path) -> None:
        self.corpus = Path(corpus).resolve()
        supplied_manifest = Path(manifest)
        if supplied_manifest.is_symlink() or not supplied_manifest.is_file():
            raise RealAxisError(f"WINDOWS_MANIFEST_INVALID: {supplied_manifest}")
        self.manifest_path = supplied_manifest.resolve()
        if not self.manifest_path.is_file():
            raise RealAxisError(f"WINDOWS_MANIFEST_INVALID: {self.manifest_path}")
        value = load_json_object(self.manifest_path, label="WINDOWS_MANIFEST")
        if value.get("schema") != WINDOWS_SCHEMA or value.get("schema_version") != 1:
            raise RealAxisError("WINDOWS_MANIFEST_SCHEMA_MISMATCH")
        self.corpus_id = _require_string(value.get("corpus_id"), label="CORPUS_ID")
        rows = value.get("windows")
        if not isinstance(rows, list) or not rows:
            raise RealAxisError("WINDOWS_EMPTY")
        windows: list[Window] = []
        identities: list[str | int] = []
        for ordinal, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RealAxisError("WINDOW_ROW_INVALID")
            window_id = row.get("window_id")
            if not isinstance(window_id, (str, int)) or isinstance(window_id, bool):
                raise RealAxisError("WINDOW_ID_INVALID")
            class_name = _require_string(row.get("class"), label="WINDOW_CLASS")
            path = safe_relative_path(
                self.corpus,
                _require_string(row.get("path"), label="WINDOW_PATH"),
                label="WINDOW",
            )
            if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get(
                "sha256"
            ):
                raise RealAxisError(f"WINDOW_IDENTITY_MISMATCH: {window_id!r}")
            try:
                array = np.load(path, allow_pickle=False, mmap_mode="r")
            except Exception as exc:
                raise RealAxisError(f"WINDOW_ARRAY_INVALID: {path}: {exc}") from exc
            if list(array.shape) != row.get("shape") or array.dtype.str != row.get("dtype"):
                raise RealAxisError(f"WINDOW_SCHEMA_MISMATCH: {window_id!r}")
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
                raise RealAxisError(f"WINDOW_SHAPE_OR_DTYPE_INVALID: {window_id!r}")
            windows.append(
                Window(
                    ordinal=ordinal,
                    window_id=window_id,
                    class_name=class_name,
                    path=path,
                    descriptor=dict(row),
                )
            )
            identities.append(window_id)
        if len(set(map(str, identities))) != len(identities):
            raise RealAxisError("WINDOW_IDS_DUPLICATE")
        self.windows = tuple(windows)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.ordered_window_ids_sha256 = canonical_sha256(identities)

    def load(self, window: Window) -> np.ndarray[Any, Any]:
        value = np.load(window.path, allow_pickle=False)
        try:
            return require_finite_float(value, label="WINDOW_INPUT", rank=2)
        except MetricsError as exc:
            raise RealAxisError(str(exc)) from exc

    def verify_identity_unchanged(self) -> None:
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise RealAxisError("WINDOWS_MANIFEST_CHANGED_DURING_RUN")
        for window in self.windows:
            if (
                window.path.stat().st_size != window.descriptor.get("bytes")
                or sha256_file(window.path) != window.descriptor.get("sha256")
            ):
                raise RealAxisError(
                    f"WINDOW_CHANGED_DURING_RUN: {window.window_id!r}"
                )


class RealAxisRunner:
    """A manifest-driven layer walk with no process-global monkeypatching."""

    def __init__(self, root: str | Path, *, require_pack: bool = False) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "real_axis.json"
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise RealAxisError(f"REAL_AXIS_MANIFEST_INVALID: {self.manifest_path}")
        value = load_json_object(self.manifest_path, label="REAL_AXIS_MANIFEST")
        if value.get("schema") != REAL_AXIS_SCHEMA or value.get("schema_version") != 1:
            raise RealAxisError("REAL_AXIS_MANIFEST_SCHEMA_MISMATCH")
        self.model_id = _require_string(value.get("model_id"), label="MODEL_ID")
        rows = value.get("layers")
        if not isinstance(rows, list) or not rows:
            raise RealAxisError("REAL_AXIS_LAYERS_EMPTY")
        for expected, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("layer") != expected:
                raise RealAxisError(
                    f"REAL_AXIS_LAYER_ORDER_INVALID: expected layer {expected}"
                )
            if row.get("activation") not in ("identity", "relu", "tanh"):
                raise RealAxisError(f"REAL_AXIS_ACTIVATION_INVALID: layer {expected}")
            if not isinstance(row.get("descriptor"), dict):
                raise RealAxisError(f"REAL_AXIS_DESCRIPTOR_INVALID: layer {expected}")
        self.layers = tuple(dict(row) for row in rows)
        if not isinstance(value.get("head"), dict):
            raise RealAxisError("REAL_AXIS_HEAD_INVALID")
        self.head = dict(value["head"])
        self.manifest_sha256 = sha256_file(self.manifest_path)
        pack_path = self.root / "BANANA_PACK_MANIFEST.json"
        if require_pack and (pack_path.is_symlink() or not pack_path.is_file()):
            raise RealAxisError(f"PACK_MANIFEST_MISSING: {pack_path}")
        self.pack_manifest_sha256 = sha256_file(pack_path) if pack_path.is_file() else None
        self.pack_manifest = (
            load_json_object(pack_path, label="PACK_MANIFEST") if pack_path.is_file() else None
        )
        if self.pack_manifest is not None:
            if (
                self.pack_manifest.get("schema") != "bs-pack"
                or not isinstance(self.pack_manifest.get("instance_id"), str)
                or not self.pack_manifest["instance_id"]
                or self.pack_manifest.get("model_id", self.model_id) != self.model_id
            ):
                raise RealAxisError("PACK_MANIFEST_REAL_AXIS_IDENTITY_MISMATCH")

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def identity(self) -> dict[str, Any]:
        result = {
            "model_id": self.model_id,
            "real_axis_manifest_sha256": self.manifest_sha256,
            "layer_count": self.layer_count,
        }
        if self.pack_manifest_sha256 is not None:
            result["pack_manifest_sha256"] = self.pack_manifest_sha256
            result["pack_instance_id"] = self.pack_manifest.get("instance_id")
        return result

    def verify_identity_unchanged(self) -> None:
        if sha256_file(self.manifest_path) != self.manifest_sha256:
            raise RealAxisError("REAL_AXIS_MANIFEST_CHANGED_DURING_RUN")
        if self.pack_manifest_sha256 is not None:
            pack_path = self.root / "BANANA_PACK_MANIFEST.json"
            if sha256_file(pack_path) != self.pack_manifest_sha256:
                raise RealAxisError("PACK_MANIFEST_CHANGED_DURING_RUN")

    def layer_descriptor(self, layer: int) -> dict[str, Any]:
        try:
            row = self.layers[layer]
        except IndexError as exc:
            raise RealAxisError(f"LAYER_OUT_OF_RANGE: {layer}") from exc
        descriptor = {
            "layer": layer,
            "activation": row["activation"],
            "descriptor": row["descriptor"],
            "weight": row["weight"],
            "bias": row.get("bias"),
        }
        return {**descriptor, "sha256": canonical_sha256(descriptor)}

    def apply_layer(
        self, layer: int, hidden: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        row = self.layers[layer]
        weight = _declared_array(
            self.root, row.get("weight"), label=f"LAYER_{layer}_WEIGHT", rank=2
        )
        bias = None
        if row.get("bias") is not None:
            bias = _declared_array(
                self.root, row["bias"], label=f"LAYER_{layer}_BIAS", rank=1
            )
            if bias.shape != (weight.shape[1],):
                raise RealAxisError(f"LAYER_BIAS_SHAPE_MISMATCH: layer={layer}")
        return self._apply_layer_arrays(layer, hidden, weight=weight, bias=bias)

    def _apply_layer_arrays(
        self,
        layer: int,
        hidden: np.ndarray[Any, Any],
        *,
        weight: np.ndarray[Any, Any],
        bias: np.ndarray[Any, Any] | None,
    ) -> np.ndarray[Any, Any]:
        state = require_finite_float(hidden, label="HIDDEN", rank=2)
        row = self.layers[layer]
        if state.shape[1] != weight.shape[0]:
            raise RealAxisError(
                f"LAYER_INPUT_SHAPE_MISMATCH: layer={layer} "
                f"hidden={state.shape} weight={weight.shape}"
            )
        output = state.astype(np.float64) @ weight.astype(np.float64)
        if bias is not None:
            output += bias.astype(np.float64)
        activation = row["activation"]
        if activation == "relu":
            output = np.maximum(output, 0.0)
        elif activation == "tanh":
            output = np.tanh(output)
        if not np.isfinite(output).all():
            raise RealAxisError(f"LAYER_OUTPUT_NONFINITE: layer={layer}")
        return output.astype(np.float32)

    def apply_layer_batch(
        self, layer: int, states: list[np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        if not states:
            raise RealAxisError("LAYER_STATE_BATCH_EMPTY")
        row = self.layers[layer]
        weight = _declared_array(
            self.root, row.get("weight"), label=f"LAYER_{layer}_WEIGHT", rank=2
        )
        bias = None
        if row.get("bias") is not None:
            bias = _declared_array(
                self.root, row["bias"], label=f"LAYER_{layer}_BIAS", rank=1
            )
            if bias.shape != (weight.shape[1],):
                raise RealAxisError(f"LAYER_BIAS_SHAPE_MISMATCH: layer={layer}")
        return [
            self._apply_layer_arrays(layer, state, weight=weight, bias=bias)
            for state in states
        ]

    def project_logits(self, hidden: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        weight = _declared_array(
            self.root, self.head.get("weight"), label="HEAD_WEIGHT", rank=2
        )
        bias = None
        if self.head.get("bias") is not None:
            bias = _declared_array(
                self.root, self.head["bias"], label="HEAD_BIAS", rank=1
            )
            if bias.shape != (weight.shape[1],):
                raise RealAxisError("HEAD_BIAS_SHAPE_MISMATCH")
        return self._project_logits_arrays(hidden, weight=weight, bias=bias)

    def _project_logits_arrays(
        self,
        hidden: np.ndarray[Any, Any],
        *,
        weight: np.ndarray[Any, Any],
        bias: np.ndarray[Any, Any] | None,
    ) -> np.ndarray[Any, Any]:
        state = require_finite_float(hidden, label="FINAL_HIDDEN", rank=2)
        if state.shape[1] != weight.shape[0]:
            raise RealAxisError("HEAD_INPUT_SHAPE_MISMATCH")
        logits = state.astype(np.float64) @ weight.astype(np.float64)
        if bias is not None:
            logits += bias.astype(np.float64)
        if not np.isfinite(logits).all():
            raise RealAxisError("HEAD_OUTPUT_NONFINITE")
        return logits.astype(np.float32)

    def project_logits_batch(
        self, states: list[np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        if not states:
            raise RealAxisError("HEAD_STATE_BATCH_EMPTY")
        weight = _declared_array(
            self.root, self.head.get("weight"), label="HEAD_WEIGHT", rank=2
        )
        bias = None
        if self.head.get("bias") is not None:
            bias = _declared_array(
                self.root, self.head["bias"], label="HEAD_BIAS", rank=1
            )
            if bias.shape != (weight.shape[1],):
                raise RealAxisError("HEAD_BIAS_SHAPE_MISMATCH")
        return [
            self._project_logits_arrays(state, weight=weight, bias=bias)
            for state in states
        ]

    def walk(self, hidden: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        state = hidden
        for layer in range(self.layer_count):
            state = self.apply_layer(layer, state)
        return self.project_logits(state)
