from __future__ import annotations

import importlib
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import torch

FAST_PATH_ERROR = "BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING"
EXPECTED_LAYOUT_SHA256 = "0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8"
EXPECTED_FAMILY_CODES = {"qtip2": 0, "qtip3": 1, "d4": 2, "native": 3}


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


@dataclass(frozen=True)
class NativePlanePack:
    root: Path
    layers: tuple[int, ...]
    layout_sha256: str
    architecture: str

    @classmethod
    def from_model_root(cls, model_root: str | Path) -> "NativePlanePack":
        model_root = Path(model_root).expanduser().resolve()
        config = _json(model_root / "config.json")
        quant = config.get("quantization_config")
        if not isinstance(quant, dict) or quant.get("quant_method") != "bs-mixed-tier":
            raise _fail("model quantization_config.quant_method must be bs-mixed-tier")
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
            "quant_method": "bs-mixed-tier",
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
        pack = cls(root, layers, layout, str(quant.get("architecture", "sm_120")))
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
        return tensor.contiguous()

    def _load_projection(self, projection: str) -> ProjectionState:
        suffix = "13" if projection == "fused13" else "2"
        input_width = int(self.meta["K13" if projection == "fused13" else "K2"])
        output_width = int(self.meta["N13" if projection == "fused13" else "N2"])
        experts = int(self.meta["E"])
        tiers_value = self.meta.get(f"tier{suffix}")
        slots_value = self.meta.get(f"slot{suffix}")
        families_value = self.meta.get(f"family{suffix}")
        if not all(isinstance(value, list) and len(value) == experts for value in (tiers_value, slots_value, families_value)):
            raise _fail(f"layer {self.layer_index} {projection} route shape drift")
        tiers = tuple(str(value) for value in tiers_value)
        slots = torch.tensor(slots_value, dtype=torch.int64, device=self.device)
        families = torch.tensor(families_value, dtype=torch.int8, device=self.device)
        if set(int(value) for value in families_value) - {0, 1, 2, 3}:
            raise _fail(f"layer {self.layer_index} {projection} has unsupported family code")
        specs = (self.meta.get("payloads") or {}).get(projection)
        if not isinstance(specs, dict) or not specs:
            raise _fail(f"layer {self.layer_index} {projection} payload map missing")
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
            slot = int(slots_value[expert])
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
        if bool(torch.any(expert_ids < 0)) or bool(torch.any(expert_ids >= len(state.tiers))):
            raise _fail(f"layer {self.layer_index} {projection} expert id out of range")
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
