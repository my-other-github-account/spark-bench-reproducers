from __future__ import annotations

import json
import logging
import re
from functools import wraps
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .native_planes import (
    NativePlaneLayer,
    NativePlanePack,
    NativePlanePrerequisiteError,
    _fail,
)
from .contract import load_runtime_contract
from .repair import apply_runtime_repairs

try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
except ImportError as exc:  # pragma: no cover - wheel dependency is mandatory
    raise RuntimeError("banana-smasher-plugin requires stock vLLM") from exc


_DEEPSEEK_V4_EXPERT_PREFIX = re.compile(
    r"(?:^|\.)model\.layers\.(?P<layer>[0-9]+)\.ffn\.experts$"
)
QUANT_METHOD = "banana_smasher"
_LOG = logging.getLogger("banana_smasher_plugin.quantization")
_DENSE_STACKED_MAPPING = (
    ("gate_up_proj", "w1"),
    ("gate_up_proj", "w3"),
    ("attn.fused_wqa_wkv", "attn.wq_a"),
    ("attn.fused_wqa_wkv", "attn.wkv"),
    ("compressor.fused_wkv_wgate", "compressor.wkv"),
    ("compressor.fused_wkv_wgate", "compressor.wgate"),
)


def _stacked_dense_parameter_name(name: str) -> str:
    for parameter_name, checkpoint_name in _DENSE_STACKED_MAPPING:
        if checkpoint_name in name:
            return name.replace(checkpoint_name, parameter_name)
    return name


def _is_mtp_parameter_name(name: str) -> bool:
    return name.startswith(("mtp.", "model.mtp."))


def _runtime_mtp_enabled(model: Any) -> bool:
    """Return False only for an explicitly disabled runtime MTP graph."""
    if hasattr(model, "_banana_smasher_mtp_enabled"):
        return bool(model._banana_smasher_mtp_enabled)
    for candidate in (model, getattr(model, "model", None)):
        vllm_config = getattr(candidate, "vllm_config", None)
        if vllm_config is not None and hasattr(vllm_config, "speculative_config"):
            return getattr(vllm_config.speculative_config, "method", None) == "mtp"
    return True


def preflight_dense_weight_map(
    config: "BananaSmasherQuantizationConfig",
    named_parameter_names: set[str],
    *,
    map_checkpoint_names,
    mtp_enabled: bool = True,
) -> dict[str, int]:
    """Fail closed if any dense checkpoint key cannot reach a named parameter."""
    config.dense_preflight_passed = False
    root = config.model_root
    if root is None:
        raise _fail("DENSE_WEIGHT_MAP_PREFLIGHT model root is not bound")
    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
    except Exception as exc:
        raise _fail(
            f"DENSE_WEIGHT_MAP_PREFLIGHT invalid or missing weight map {index_path}: {exc}"
        ) from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise _fail("DENSE_WEIGHT_MAP_PREFLIGHT weight map is empty")
    mapped = map_checkpoint_names(list(weight_map))
    if not isinstance(mapped, list):
        mapped = list(mapped)
    dense_names = [
        name
        for name in mapped
        if ".experts." not in name
        and (mtp_enabled or not _is_mtp_parameter_name(name))
    ]
    resolved = {_stacked_dense_parameter_name(name) for name in dense_names}
    missing = sorted(resolved - named_parameter_names)
    if missing:
        preview = ", ".join(missing[:8])
        raise _fail(
            "DENSE_WEIGHT_MAP_PREFLIGHT checkpoint names do not resolve to registered "
            f"named parameters: missing={len(missing)} first=[{preview}]"
        )
    config.dense_preflight_passed = True
    return {
        "dense_checkpoint_names": len(dense_names),
        "mapped_named_parameters": len(resolved),
        "excluded_disabled_mtp_names": sum(
            not mtp_enabled and _is_mtp_parameter_name(name) for name in mapped
        ),
    }


def install_deepseek_v4_dense_preflight() -> None:
    """Install the stock model's pre-load named-parameter gate once per process."""
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    original_init = DeepseekV4ForCausalLM.__init__
    if not getattr(original_init, "_banana_smasher_mtp_state", False):

        @wraps(original_init)
        def init_with_mtp_state(model, *args, **kwargs):
            vllm_config = kwargs.get("vllm_config")
            original_init(model, *args, **kwargs)
            if vllm_config is not None and hasattr(vllm_config, "speculative_config"):
                model._banana_smasher_mtp_enabled = (
                    getattr(vllm_config.speculative_config, "method", None) == "mtp"
                )

        setattr(init_with_mtp_state, "_banana_smasher_mtp_state", True)
        DeepseekV4ForCausalLM.__init__ = init_with_mtp_state

    original = DeepseekV4ForCausalLM.load_weights
    if getattr(original, "_banana_smasher_dense_preflight", False):
        return

    def load_weights(model, weights):
        config = getattr(getattr(model, "model", None), "quant_config", None)
        if isinstance(config, BananaSmasherQuantizationConfig):
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None or not callable(getattr(mapper, "apply_list", None)):
                raise _fail(
                    "DENSE_WEIGHT_MAP_PREFLIGHT stock DeepSeek-V4 weights mapper is unavailable"
                )
            preflight_dense_weight_map(
                config,
                {name for name, _ in model.named_parameters()},
                map_checkpoint_names=mapper.apply_list,
                mtp_enabled=_runtime_mtp_enabled(model),
            )
        result = original(model, weights)
        if (
            isinstance(config, BananaSmasherQuantizationConfig)
            and config.raw.get("repair_format") == "bs-basic-repair-v1"
        ):
            if config.model_root is None:
                raise _fail("DENSE_REPAIR_RUNTIME model root is not bound")
            contract = load_runtime_contract(config.model_root)
            repair_report = apply_runtime_repairs(model, contract)
            _LOG.warning(
                "BANANA_SMASHER_DENSE_REPAIR_APPLIED norms=%d output_log_gains=%d",
                len(repair_report["norms"]),
                len(repair_report["output_log_gains"]),
            )
        return result

    load_weights._banana_smasher_dense_preflight = True
    DeepseekV4ForCausalLM.load_weights = load_weights


class BananaSmasherMoEMethod(FusedMoEMethodBase):
    """Stock-vLLM routed-expert method backed only by P1016 native planes."""

    def __init__(
        self,
        quant_config: "BananaSmasherQuantizationConfig",
        moe: Any,
        layer_index: int,
        prefix: str,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.layer_index = int(layer_index)
        self.prefix = prefix
        self.native_layer: NativePlaneLayer | None = None

    @property
    def supports_internal_mk(self) -> bool:
        # vLLM 0.24 uses this migration flag to leave a quant method's native
        # apply() hook intact instead of wrapping it in a stock GEMM method.
        return True

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return False

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del extra_weight_attrs
        expected = (256, 4096, 2048, torch.bfloat16)
        actual = (
            int(num_experts),
            int(hidden_size),
            int(intermediate_size_per_partition),
            params_dtype,
        )
        if actual != expected:
            raise _fail(
                f"stock DeepSeek-V4 MoE shape prerequisite mismatch for {self.prefix}: "
                f"actual={actual} expected={expected}"
            )
        pack = self.quant_config.pack
        if pack is None:
            raise _fail("quantization config was not bound to the stock model root")
        device = next(layer.buffers(), torch.empty(0)).device
        if device.type != "cuda":
            raise _fail(
                f"accelerated native-plane layer requires CUDA, got device={device}"
            )
        self.native_device = device
        # No dense routed-expert parameters are registered. Plane allocation is
        # deliberately deferred until after the full dense weight-map preflight.
        layer.bs_native_plane_layer = None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if not self.quant_config.dense_preflight_passed:
            raise _fail(
                "DENSE_WEIGHT_MAP_PREFLIGHT did not pass before expert plane allocation"
            )
        if self.native_layer is None:
            pack = self.quant_config.pack
            if pack is None:
                raise _fail("quantization config was not bound to the stock model root")
            self.native_layer = NativePlaneLayer(
                pack,
                self.layer_index,
                device=self.native_device,
            )
        layer.bs_native_plane_layer = self.native_layer

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        del layer
        return FusedMoEQuantConfig.make()

    def maybe_make_prepare_finalize(self, routing_tables=None):
        del routing_tables
        raise _fail("stock modular-GEMM wrapping is forbidden for native planes")

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del layer, shared_experts, shared_experts_input
        native_layer = self.native_layer
        if native_layer is None:
            raise _fail(f"native planes were not loaded for {self.prefix}")
        original_shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        weights = topk_weights.reshape(flat.shape[0], -1)
        ids = topk_ids.to(device=flat.device, dtype=torch.int64).reshape(
            flat.shape[0], -1
        )
        if tuple(weights.shape) != tuple(ids.shape) or ids.shape[1] != 6:
            raise _fail(
                f"stock DeepSeek-V4 top-k route shape mismatch for {self.prefix}: "
                f"weights={tuple(weights.shape)} ids={tuple(ids.shape)}"
            )
        routed_ids = ids.reshape(-1)
        expanded = flat[:, None, :].expand(
            flat.shape[0], ids.shape[1], flat.shape[1]
        ).reshape(-1, flat.shape[1])
        fused = native_layer.forward(expanded, routed_ids, "fused13")
        gate, up = fused.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        down = native_layer.forward(activated, routed_ids, "down")
        result = (
            down.reshape(flat.shape[0], ids.shape[1], flat.shape[1])
            * weights[..., None].to(down.dtype)
        ).sum(dim=1)
        return result.reshape(original_shape)


class BananaSmasherQuantizationConfig(QuantizationConfig):
    def __init__(self, raw: dict[str, Any]):
        super().__init__()
        self.raw = dict(raw)
        self.model_root: Path | None = None
        self.pack: NativePlanePack | None = None
        activation_scheme = self.raw.get("activation_scheme")
        fmt = self.raw.get("fmt")
        weight_block_size = self.raw.get("weight_block_size")
        if activation_scheme != "dynamic" or fmt != "e4m3":
            raise ValueError(
                "banana-smasher-plugin requires dense FP8 descriptors "
                "activation_scheme='dynamic' and fmt='e4m3'"
            )
        if (
            not isinstance(weight_block_size, list)
            or len(weight_block_size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in weight_block_size)
        ):
            raise ValueError(
                "banana-smasher-plugin requires a two-dimensional dense FP8 weight_block_size"
            )
        self.dense_fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            weight_block_size=list(weight_block_size),
        )
        scale_fmt = self.raw.get("scale_fmt")
        if scale_fmt not in {"ue8m0", "float32"}:
            raise ValueError(
                "banana-smasher-plugin requires dense FP8 scale_fmt to be "
                "'ue8m0' or 'float32'"
            )
        # Expose blocked weights to VllmConfig before model construction.  Stock
        # vLLM then enables quant_fp8 plus norm/activation quant fusion instead
        # of silently selecting custom_ops=['none'] for this plugin format.
        self.weight_block_size = list(weight_block_size)
        # Preserve the checkpoint's UE8M0 contract.  On SM12x this is the input
        # that selects DeepGemmFp8BlockScaledMMKernel rather than the generic
        # Triton scaled-MM fallback.
        self.dense_fp8_config.is_scale_e8m0 = scale_fmt == "ue8m0"
        self.dense_checkpoint_scale_fmt = scale_fmt
        self.dense_preflight_passed = False

    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BananaSmasherQuantizationConfig":
        if config.get("quant_method") != QUANT_METHOD:
            raise ValueError(
                f"banana-smasher-plugin requires quant_method={QUANT_METHOD}"
            )
        if config.get("format") != "bs-pack" or config.get("format_version") != 1:
            raise ValueError("banana-smasher-plugin refuses unsupported pack format")
        return cls(config)

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: Any | None = None,
        revision: str | None = None,
    ) -> None:
        del hf_config, revision
        root = Path(model_name).expanduser()
        if not root.is_dir():
            raise _fail(
                "native-plane stock adapter requires a local complete model directory: "
                f"{model_name}"
            )
        self.model_root = root.resolve()
        self.pack = NativePlanePack.from_model_root(self.model_root)
        if self.pack.architecture not in {"sm_120", "sm_121", "sm_121a"}:
            raise _fail(
                f"native-plane architecture prerequisite mismatch: {self.pack.architecture}"
            )

    def _select_stock_dense_backend(self) -> None:
        if self.dense_checkpoint_scale_fmt != "ue8m0":
            return
        try:
            from vllm.config import get_current_vllm_config_or_none
        except ImportError:
            # Lightweight unit-test stubs do not provide vLLM's process config.
            return
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            raise _fail(
                "stock dense FP8 backend selection requires the active vLLM config"
            )
        backend = vllm_config.kernel_config.linear_backend
        if backend != "auto":
            raise _fail(
                "UE8M0 dense FP8 requires stock automatic DeepGEMM dispatch; "
                f"linear_backend must remain 'auto', got {backend!r}"
            )
        compilation = vllm_config.compilation_config
        pass_config = compilation.pass_config
        if "+quant_fp8" not in compilation.custom_ops:
            raise _fail(
                "blocked UE8M0 weights did not enable the stock quant_fp8 custom op"
            )
        if not pass_config.fuse_norm_quant or not pass_config.fuse_act_quant:
            raise _fail(
                "stock UE8M0 compile fast paths are disabled: "
                f"fuse_norm_quant={pass_config.fuse_norm_quant!r}, "
                f"fuse_act_quant={pass_config.fuse_act_quant!r}"
            )
        if not getattr(self, "_fast_path_logged", False):
            _LOG.warning(
                "BANANA_SMASHER_VLLM_COMPILE_FAST_PATHS "
                "fuse_norm_quant=true fuse_act_quant=true custom_op=quant_fp8 "
                "linear_backend=auto"
            )
            self._fast_path_logged = True

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Any | None:
        if isinstance(layer, LinearBase):
            self._select_stock_dense_backend()
            return Fp8LinearMethod(self.dense_fp8_config)
        if not isinstance(layer, RoutedExperts):
            return None
        match = _DEEPSEEK_V4_EXPERT_PREFIX.search(prefix)
        if match is None:
            raise NativePlanePrerequisiteError(
                "BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING: "
                f"stock DeepSeek-V4 routed-expert prefix mismatch: {prefix}"
            )
        layer_index = int(match.group("layer"))
        if self.pack is None or layer_index not in self.pack.layers:
            raise _fail(
                f"stock DeepSeek-V4 layer {layer_index} is not bound by the native-plane pack"
            )
        return BananaSmasherMoEMethod(
            self,
            layer.moe_config,
            layer_index,
            prefix,
        )
