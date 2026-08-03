from __future__ import annotations

import importlib
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

FAST_PATH_ERROR = "BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING"
EXPECTED_LAYOUT_SHA256 = "0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8"
EXPECTED_FAMILY_CODES = {"qtip2": 0, "qtip3": 1, "d4": 2, "native": 3}
_LOGGER = logging.getLogger(__name__)
_posix_fadvise = getattr(os, "posix_fadvise", None)
_POSIX_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)
_PLANE_LOAD_PROGRESS: dict[Path, int] = {}
_PLANE_LOAD_TOTALS: dict[Path, int] = {}
OS_FLOOR_BYTES = 4 << 30
SELECTION_SCHEMA = "bs-pack-selected-payloads-v1"


class NativePlanePrerequisiteError(RuntimeError):
    pass


def _fail(message: str) -> NativePlanePrerequisiteError:
    return NativePlanePrerequisiteError(f"{FAST_PATH_ERROR}: {message}")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise _fail(f"invalid or missing JSON prerequisite {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"JSON prerequisite is not an object: {path}")
    return value


def _mem_available_kib() -> int:
    try:
        return next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    except Exception:
        return -1


def _local_capacity_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError) as exc:
        raise _fail(f"cannot determine local memory capacity before allocation: {exc}") from exc
    capacity = pages * page_size
    if capacity <= OS_FLOOR_BYTES:
        raise _fail(
            f"local capacity {capacity} does not exceed required {OS_FLOOR_BYTES} byte OS floor"
        )
    return capacity


def _selected_residency_preflight(
    root: Path, selection: dict[str, Any]
) -> dict[str, Any]:
    producer = selection.get("producer_stage")
    layers = selection.get("layers")
    runtime_floor = selection.get("runtime_floor_bytes")
    dense_base = selection.get("dense_base_bytes")
    if not isinstance(producer, str) or not producer:
        raise _fail("selected-payload manifest is missing producer_stage")
    if not isinstance(layers, dict) or not layers:
        raise _fail("selected-payload manifest has no layers")
    if not isinstance(runtime_floor, int) or runtime_floor < 0:
        raise _fail("selected-payload manifest runtime_floor_bytes is invalid")
    if not isinstance(dense_base, int) or dense_base < 0:
        raise _fail("selected-payload manifest dense_base_bytes is invalid")
    additional = selection.get("additional_resident_role_bytes", {})
    if not isinstance(additional, dict) or any(
        not isinstance(role, str) or not isinstance(size, int) or size < 0
        for role, size in additional.items()
    ):
        raise _fail("selected-payload manifest additional resident role bytes are invalid")

    role_bytes: dict[str, int] = {}
    seen_files: set[str] = set()
    file_metadata: dict[str, tuple[list[int], str, int]] = {}
    planes_root = (root / "planes").resolve()
    for layer, projections in sorted(layers.items()):
        if not isinstance(projections, dict):
            raise _fail(f"selected-payload layer {layer} is not an object")
        for projection, route in sorted(projections.items()):
            payloads = route.get("payloads") if isinstance(route, dict) else None
            if not isinstance(payloads, dict) or not payloads:
                raise _fail(f"selected-payload route {layer}/{projection} has no payloads")
            for tier, payload in sorted(payloads.items()):
                tensors = payload.get("tensors") if isinstance(payload, dict) else None
                if not isinstance(tensors, dict) or not tensors:
                    raise _fail(f"selected payload {layer}/{projection}/{tier} has no tensors")
                for role, spec in sorted(tensors.items()):
                    relative = spec.get("file") if isinstance(spec, dict) else None
                    data_bytes = spec.get("data_bytes") if isinstance(spec, dict) else None
                    if not isinstance(relative, str) or not relative:
                        raise _fail(
                            f"selected payload {layer}/{projection}/{tier}/{role} has no explicit file"
                        )
                    relative_path = Path(relative)
                    if (
                        relative_path.is_absolute()
                        or len(relative_path.parts) != 1
                        or ".." in relative_path.parts
                    ):
                        raise _fail(f"selected payload has unsafe explicit file: {relative!r}")
                    if not isinstance(data_bytes, int) or data_bytes < 0:
                        raise _fail(
                            f"selected payload {layer}/{projection}/{tier}/{role} has invalid data_bytes"
                        )
                    actual = file_metadata.get(relative)
                    if actual is None:
                        unresolved = planes_root / relative_path
                        path = unresolved.resolve()
                        if (
                            planes_root not in path.parents
                            or not unresolved.is_file()
                            or unresolved.is_symlink()
                        ):
                            raise _fail(f"selected payload file is missing or unsafe: {relative}")
                        try:
                            array = np.load(path, mmap_mode="r", allow_pickle=False)
                        except Exception as exc:
                            raise _fail(
                                f"cannot inspect selected payload before allocation {relative}: {exc}"
                            ) from exc
                        actual = (list(array.shape), str(array.dtype), int(array.nbytes))
                        del array
                        file_metadata[relative] = actual
                    expected_shape = spec.get("shape") if isinstance(spec, dict) else None
                    expected_dtype = spec.get("dtype") if isinstance(spec, dict) else None
                    if (
                        expected_shape != actual[0]
                        or expected_dtype != actual[1]
                        or data_bytes != actual[2]
                    ):
                        raise _fail(
                            f"selected payload metadata drift {relative}: "
                            f"manifest={expected_shape}/{expected_dtype}/{data_bytes} "
                            f"actual={actual[0]}/{actual[1]}/{actual[2]}; "
                            f"producer_stage={producer}; remediation=re-export the selected pack"
                        )
                    if relative in seen_files:
                        continue
                    seen_files.add(relative)
                    key = f"selected:{role}"
                    role_bytes[key] = role_bytes.get(key, 0) + actual[2]
    role_bytes["dense_base"] = dense_base
    role_bytes["runtime_floor"] = runtime_floor
    for role, size in additional.items():
        role_bytes[f"additional:{role}"] = size
    resident = sum(role_bytes.values())
    capacity = _local_capacity_bytes()
    budget = capacity - OS_FLOOR_BYTES
    report = {
        "producer_stage": producer,
        "role_bytes": dict(sorted(role_bytes.items())),
        "selected_file_count": len(seen_files),
        "resident_bytes": resident,
        "capacity_bytes": capacity,
        "os_floor_bytes": OS_FLOOR_BYTES,
        "budget_bytes": budget,
    }
    if resident > budget:
        math_text = " + ".join(
            f"{role}={value}" for role, value in sorted(role_bytes.items())
        )
        raise _fail(
            "STRUCTURAL_MEMORY_PREFLIGHT_OVER_BUDGET at t=0 before tensor allocation: "
            f"{math_text}; resident={resident}; capacity={capacity}; "
            f"os_floor={OS_FLOOR_BYTES}; budget={budget}; over={resident - budget}; "
            f"producer_stage={producer}; remediation=re-export/re-tier the selected manifest payloads "
            "or reduce the documented runtime floor"
        )
    return report


def _release_mmap_pages(path: Path, array: np.ndarray) -> None:
    if getattr(array, "_mmap", None) is None:
        return
    if _posix_fadvise is None:
        _LOGGER.warning("POSIX_FADV_DONTNEED unavailable for streamed plane %s", path)
        return
    try:
        with path.open("rb") as plane:
            _posix_fadvise(plane.fileno(), 0, 0, _POSIX_FADV_DONTNEED)
    except Exception as exc:
        _LOGGER.warning("POSIX_FADV_DONTNEED failed for streamed plane %s: %s", path, exc)


def _record_plane_load(root: Path) -> None:
    root = root.resolve()
    loaded = _PLANE_LOAD_PROGRESS.get(root, 0) + 1
    _PLANE_LOAD_PROGRESS[root] = loaded
    total = _PLANE_LOAD_TOTALS.setdefault(root, sum(1 for _ in (root / "planes").glob("*.npy")))
    if loaded % 50 == 0:
        _LOGGER.info(
            "BANANA_SMASHER_PLANE_LOAD_WATERMARK loaded=%d total=%d MemAvailable_kB=%d",
            loaded,
            total,
            _mem_available_kib(),
        )


@dataclass(frozen=True)
class NativePlanePack:
    root: Path
    layers: tuple[int, ...]
    layout_sha256: str
    architecture: str
    selected_payloads: dict[str, Any]
    residency: dict[str, Any]

    @classmethod
    def from_model_root(cls, model_root: str | Path) -> "NativePlanePack":
        model_root = Path(model_root).expanduser().resolve()
        config = _json(model_root / "config.json")
        quant = config.get("quantization_config")
        if not isinstance(quant, dict) or quant.get("quant_method") != "banana_smasher":
            raise _fail("model quantization_config.quant_method must be banana_smasher")
        if quant.get("format") != "bs-pack" or quant.get("format_version") != 1:
            raise _fail("model quantization_config must select bs-pack format_version=1")
        relative_root = quant.get("pack_root", ".")
        if not isinstance(relative_root, str):
            raise _fail("quantization_config.pack_root must be a path string")
        root = (model_root / relative_root).resolve()
        if root != model_root and model_root not in root.parents:
            raise _fail(f"pack_root escapes model directory: {relative_root}")
        manifest_name = quant.get("pack_manifest", "BANANA_PACK_MANIFEST.json")
        if not isinstance(manifest_name, str) or Path(manifest_name).is_absolute() or ".." in Path(manifest_name).parts:
            raise _fail(f"unsafe pack_manifest path: {manifest_name!r}")
        manifest = _json(root / manifest_name)
        expected = {
            "schema": "bs-pack",
            "schema_version": 1,
            "quant_method": "banana_smasher",
            "source_format": "p1016-true-c-native-planes-v1",
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise _fail(f"pack {key} mismatch: expected {value!r}, got {manifest.get(key)!r}")
        layout = manifest.get("tensor_layout_sha256")
        configured_layout = quant.get("tensor_layout_sha256", layout)
        if not isinstance(layout, str) or len(layout) != 64 or configured_layout != layout:
            raise _fail(
                "native-plane tensor layout mismatch: "
                f"config={configured_layout!r} manifest={layout!r}"
            )
        if layout != EXPECTED_LAYOUT_SHA256:
            raise _fail(
                "native-plane tensor layout prerequisite mismatch: "
                f"expected={EXPECTED_LAYOUT_SHA256} actual={layout}"
            )
        layers_value = manifest.get("layers")
        if not isinstance(layers_value, list) or not layers_value:
            raise _fail("pack manifest has no native-plane layers")
        try:
            layers = tuple(int(value) for value in layers_value)
        except Exception as exc:
            raise _fail("pack layer list is malformed") from exc
        if len(layers) != len(set(layers)) or layers != tuple(sorted(layers)):
            raise _fail(f"pack layer list is duplicated or unordered: {layers}")
        selection = manifest.get("selected_payloads")
        if not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA:
            raise _fail(
                f"pack manifest must own exactly one {SELECTION_SCHEMA} selection; no fallback is allowed"
            )
        selection_layers = selection.get("layers")
        if not isinstance(selection_layers, dict) or set(selection_layers) != {
            str(layer) for layer in layers
        }:
            raise _fail("selected-payload manifest layer set does not match pack layers")
        residency = _selected_residency_preflight(root, selection)
        pack = cls(
            root,
            layers,
            layout,
            str(quant.get("architecture", "")),
            selection,
            residency,
        )
        for layer in layers:
            meta = _json(pack.meta_path(layer))
            if (
                meta.get("format") != "p1016-true-c-native-planes-v1"
                or meta.get("layer") != layer
                or meta.get("family_codes") != EXPECTED_FAMILY_CODES
            ):
                raise _fail(f"native-plane metadata binding drift for layer {layer}")
        return pack

    def meta_path(self, layer: int) -> Path:
        return self.root / "planes" / f"layer_{int(layer):03d}.meta.json"

    def selected_projection(self, layer: int, projection: str) -> dict[str, Any]:
        try:
            value = self.selected_payloads["layers"][str(int(layer))][projection]
        except (KeyError, TypeError) as exc:
            raise _fail(f"manifest selection missing layer {layer} projection {projection}") from exc
        if not isinstance(value, dict):
            raise _fail(f"manifest selection route {layer}/{projection} is not an object")
        return value


@dataclass
class ProjectionState:
    name: str
    input_width: int
    output_width: int
    families: torch.Tensor
    tiers: tuple[str, ...]
    slots: torch.Tensor
    payloads: dict[str, dict[str, torch.Tensor]]
    pointer_tables: dict[str, torch.Tensor]
    offsets2: torch.Tensor
    offsets3: torch.Tensor
    lut: torch.Tensor


Dispatch = Callable[..., torch.Tensor]


def _fwht(value: torch.Tensor) -> torch.Tensor:
    """Exact normalized transform used by the sealed P1016 QTIP path."""
    width = value.shape[-1]
    if width <= 0 or width & (width - 1):
        raise _fail(f"QTIP FWHT width must be a positive power of two, got {width}")
    output = value.contiguous()
    block = 1
    while block < width:
        output = output.reshape(*output.shape[:-1], -1, 2, block)
        left = output[..., 0, :].clone()
        right = output[..., 1, :].clone()
        output = torch.cat((left + right, left - right), dim=-1).reshape(
            *output.shape[:-3], -1
        )
        block *= 2
    return output / math.sqrt(width)


def _expanded_qtip_lut(device: torch.device) -> torch.Tensor:
    path = Path(__file__).with_name("qtip_tlut.npy")
    try:
        tlut_array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise _fail(f"QTIP TLUT prerequisite unavailable: {exc}") from exc
    if tuple(tlut_array.shape) != (512, 2) or str(tlut_array.dtype) != "float32":
        raise _fail(
            f"QTIP TLUT shape/dtype drift: {tlut_array.shape}/{tlut_array.dtype}"
        )
    digest = hashlib.sha256(tlut_array.tobytes()).hexdigest()
    if digest != "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19":
        raise _fail(f"QTIP TLUT tensor drift: {digest}")
    tlut = torch.from_numpy(tlut_array.copy()).to(device=device)
    index = torch.arange(1 << 16, device=device, dtype=torch.int64)
    quadratic = (index + 1) * index
    sign_flip = 1 - ((quadratic >> 15) & 1) * 2
    lookup = (quadratic >> 6) & 511
    expanded = tlut[lookup]
    expanded[:, 0] *= sign_flip
    return expanded.contiguous()


def _load_accelerated_dispatch() -> Dispatch:
    try:
        kernels = importlib.import_module("banana_smasher_plugin.p1016_kernels")
        dispatch = cast(Dispatch, getattr(kernels, "mixed_exact_gemv"))
    except Exception as exc:
        raise _fail(f"accelerated dispatch unavailable: {exc}") from exc
    if not callable(dispatch):
        raise _fail("accelerated dispatch symbol mixed_exact_gemv is not callable")

    def run(*, projection: str, x: torch.Tensor, expert_ids: torch.Tensor, state: ProjectionState) -> torch.Tensor:
        del projection
        selected_family = state.families.index_select(0, expert_ids)
        qtip_mask = selected_family.lt(2).reshape(-1, 1)
        su = state.pointer_tables["su"].index_select(0, expert_ids)
        transformed = _fwht(x.float() * su)
        kernel_input = torch.where(qtip_mask, transformed, x.float())
        raw = dispatch(
            kernel_input,
            expert_ids,
            state.families,
            state.pointer_tables,
            state.offsets2,
            state.offsets3,
            state.lut,
        )
        qtip_result = _fwht(
            raw * state.pointer_tables["wscale"].index_select(0, expert_ids).reshape(-1, 1)
        ) * state.pointer_tables["sv"].index_select(0, expert_ids)
        return torch.where(qtip_mask, qtip_result, raw).to(torch.bfloat16)

    return run


class NativePlaneLayer:
    """One stock-vLLM DeepSeek-V4 routed-MoE layer bound to P1016 planes."""

    def __init__(
        self,
        pack: NativePlanePack,
        layer_index: int,
        *,
        device: str | torch.device,
        dispatch: Dispatch | None = None,
    ) -> None:
        self.pack = pack
        self.layer_index = int(layer_index)
        if self.layer_index not in pack.layers:
            raise _fail(f"pack does not contain required layer {self.layer_index}")
        self.device = torch.device(device)
        self.meta = _json(pack.meta_path(self.layer_index))
        self._dispatch = dispatch if dispatch is not None else _load_accelerated_dispatch()
        self._states = {
            projection: self._load_projection(projection)
            for projection in ("fused13", "down")
        }

    def _tensor(self, spec: dict[str, Any]) -> torch.Tensor:
        relative = spec.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise _fail(f"layer {self.layer_index} has unsafe plane path {relative!r}")
        path = (self.pack.root / "planes" / relative).resolve()
        if self.pack.root / "planes" not in path.parents:
            raise _fail(f"layer {self.layer_index} plane escapes root: {relative}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise _fail(f"missing native plane {path}: {exc}") from exc
        if list(array.shape) != spec.get("shape") or str(array.dtype) != spec.get("dtype"):
            raise _fail(
                f"native-plane shape/dtype mismatch {path}: "
                f"actual={list(array.shape)}/{array.dtype} "
                f"expected={spec.get('shape')}/{spec.get('dtype')}"
            )
        tensor = torch.from_numpy(array.copy() if self.device.type == "cpu" else array)
        if tensor.device != self.device:
            tensor = tensor.to(self.device, non_blocking=False)
        tensor = tensor.contiguous()
        _release_mmap_pages(path, array)
        del array
        _record_plane_load(self.pack.root)
        return tensor

    def _load_projection(self, projection: str) -> ProjectionState:
        input_width = int(self.meta["K13" if projection == "fused13" else "K2"])
        output_width = int(self.meta["N13" if projection == "fused13" else "N2"])
        experts = int(self.meta["E"])
        selected = self.pack.selected_projection(self.layer_index, projection)
        tiers_value = selected.get("tiers")
        slots_value = selected.get("slots")
        families_value = selected.get("families")
        if not all(isinstance(value, list) and len(value) == experts for value in (tiers_value, slots_value, families_value)):
            raise _fail(f"layer {self.layer_index} {projection} route shape drift")
        tier_values = cast(list[Any], tiers_value)
        slot_values = cast(list[Any], slots_value)
        family_values = cast(list[Any], families_value)
        tiers = tuple(str(value) for value in tier_values)
        if not all(isinstance(value, int) and value >= 0 for value in slot_values):
            raise _fail(f"layer {self.layer_index} {projection} selected slots are malformed")
        if len(set(zip(tiers, slot_values, strict=True))) != experts:
            raise _fail(
                f"layer {self.layer_index} {projection} selected cell binds more than once"
            )
        if not all(isinstance(value, int) for value in family_values) or set(
            family_values
        ) - {0, 1, 2, 3}:
            raise _fail(f"layer {self.layer_index} {projection} has unsupported family code")
        specs = selected.get("payloads")
        if not isinstance(specs, dict) or not specs:
            raise _fail(f"layer {self.layer_index} {projection} selected payload map missing")
        routed_tiers = set(tiers)
        if set(specs) != routed_tiers:
            raise _fail(
                f"layer {self.layer_index} {projection} selected payload set drift: "
                f"routes={sorted(routed_tiers)} payloads={sorted(specs)}"
            )
        family_codes = self.meta["family_codes"]
        d4_bits_by_tier: dict[str, int] = {}
        for tier, payload_spec in specs.items():
            if not isinstance(payload_spec, dict):
                raise _fail(f"layer {self.layer_index} {projection}/{tier} payload is malformed")
            if payload_spec.get("family") != "d4":
                continue
            code_spec = (payload_spec.get("tensors") or {}).get("codes")
            if not isinstance(code_spec, dict):
                raise _fail(f"layer {self.layer_index} {projection}/{tier} D4 codes missing")
            bits = code_spec.get("index_bits")
            codebook_size = payload_spec.get("k")
            if (
                code_spec.get("encoding") != "little-endian-packed-index-rows-v1"
                or not isinstance(bits, int)
                or not 1 <= bits <= 16
                or not isinstance(codebook_size, int)
                or codebook_size != 1 << bits
                or code_spec.get("dtype") != "uint8"
            ):
                raise _fail(
                    f"layer {self.layer_index} {projection}/{tier} D4 codes are not "
                    "the required V4 little-endian row-packed payload"
                )
            d4_bits_by_tier[tier] = bits
        for expert, tier in enumerate(tiers):
            payload_spec = specs[tier]
            payload_family = (
                payload_spec.get("family") if isinstance(payload_spec, dict) else None
            )
            if family_codes.get(payload_family) != family_values[expert]:
                raise _fail(
                    f"layer {self.layer_index} {projection}/{tier} family binding drift "
                    f"at expert {expert}"
                )
        slots = torch.tensor(slot_values, dtype=torch.int64, device=self.device)
        families = torch.tensor(family_values, dtype=torch.int8, device=self.device)
        payloads: dict[str, dict[str, torch.Tensor]] = {}
        for tier, payload_spec in specs.items():
            tensor_specs = payload_spec.get("tensors") if isinstance(payload_spec, dict) else None
            if not isinstance(tensor_specs, dict) or not tensor_specs:
                raise _fail(f"layer {self.layer_index} {projection}/{tier} tensor map missing")
            payloads[tier] = {name: self._tensor(spec) for name, spec in tensor_specs.items()}
        states: list[dict[str, torch.Tensor]] = []
        for expert, tier in enumerate(tiers):
            if tier not in payloads:
                raise _fail(f"layer {self.layer_index} {projection} missing tier {tier}")
            payload = payloads[tier]
            slot = int(slot_values[expert])
            expert_ids = payload.get("expert_ids")
            if expert_ids is None or slot < 0 or slot >= expert_ids.numel() or int(expert_ids[slot]) != expert:
                raise _fail(
                    f"layer {self.layer_index} {projection}/{tier} slot binding drift at expert {expert}"
                )
            state = {name: value[slot] if name not in {"codebooks"} else value for name, value in payload.items()}
            if "codebook_index" in payload and "codebooks" in payload:
                state["codebook"] = payload["codebooks"][int(payload["codebook_index"][slot])]
            states.append(state)
        placeholder = next(iter(next(iter(payloads.values())).values()))

        def pointers(name: str) -> torch.Tensor:
            return torch.tensor(
                [int(state.get(name, placeholder).data_ptr()) for state in states],
                dtype=torch.int64,
                device=self.device,
            )

        pointer_tables = {
            "qtip_sources": pointers("trellis"),
            "d4_codes": pointers("codes"),
            "d4_scales": pointers("scales"),
            "d4_codebooks": pointers("codebook"),
            "native_packed": pointers("packed"),
            "native_scales": pointers("scales"),
            "d4_index_bits": torch.tensor(
                [d4_bits_by_tier.get(tier, 0) for tier in tiers],
                dtype=torch.int32,
                device=self.device,
            ),
        }
        input_ones = torch.ones(input_width, dtype=torch.float32, device=self.device)
        output_ones = torch.ones(output_width, dtype=torch.float32, device=self.device)
        pointer_tables["su"] = torch.stack(
            [state.get("SU", input_ones).float() for state in states]
        ).contiguous()
        pointer_tables["sv"] = torch.stack(
            [state.get("SV", output_ones).float() for state in states]
        ).contiguous()
        pointer_tables["wscale"] = torch.stack(
            [state.get("Wscale", input_ones.new_ones(())).float().reshape(()) for state in states]
        ).contiguous()
        try:
            kernels = importlib.import_module("banana_smasher_plugin.p1016_kernels")
            offsets2 = kernels.qtip_offset_map(2).to(self.device)
            offsets3 = kernels.qtip_offset_map(3).to(self.device)
        except Exception:
            offsets2 = torch.zeros(256, dtype=torch.int64, device=self.device)
            offsets3 = torch.zeros(384, dtype=torch.int64, device=self.device)
        lut = _expanded_qtip_lut(self.device)
        return ProjectionState(
            projection,
            input_width,
            output_width,
            families,
            tiers,
            slots,
            payloads,
            pointer_tables,
            offsets2,
            offsets3,
            lut,
        )

    def state(self, projection: str) -> ProjectionState:
        try:
            return self._states[projection]
        except KeyError as exc:
            raise ValueError(f"unknown projection: {projection}") from exc

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        projection: str,
    ) -> torch.Tensor:
        state = self.state(projection)
        x = x.reshape(-1, x.shape[-1])
        expert_ids = expert_ids.to(device=x.device, dtype=torch.int64).reshape(-1)
        if x.shape[0] != expert_ids.numel() or x.shape[1] != state.input_width:
            raise _fail(
                f"layer {self.layer_index} {projection} routed shape mismatch: "
                f"x={tuple(x.shape)} ids={tuple(expert_ids.shape)} expected_k={state.input_width}"
            )
        range_error = (
            f"layer {self.layer_index} {projection} expert id out of range"
        )
        # Keep the fail-closed range guards on the accelerator. Converting either
        # predicate to a Python bool synchronizes the stream and is illegal while
        # vLLM captures its startup CUDA graphs on SM12x.
        torch.ops.aten._assert_async.msg(torch.all(expert_ids >= 0), range_error)
        torch.ops.aten._assert_async.msg(
            torch.all(expert_ids < len(state.tiers)), range_error
        )
        result = self._dispatch(
            projection=projection,
            x=x,
            expert_ids=expert_ids,
            state=state,
        )
        if tuple(result.shape) != (x.shape[0], state.output_width):
            raise _fail(
                f"layer {self.layer_index} {projection} accelerated output shape drift: "
                f"{tuple(result.shape)}"
            )
        return result
