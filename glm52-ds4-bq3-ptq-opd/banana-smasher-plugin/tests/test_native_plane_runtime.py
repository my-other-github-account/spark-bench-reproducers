from __future__ import annotations

import inspect
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import banana_smasher_plugin.native_planes as native_planes

from banana_smasher_plugin.native_planes import (
    EXPECTED_LAYOUT_SHA256,
    FAST_PATH_ERROR,
    NativePlaneLayer,
    NativePlanePack,
    NativePlanePrerequisiteError,
    ProjectionState,
    _expanded_qtip_lut,
    _fwht,
    _load_accelerated_dispatch,
)


def _write_array(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / name
    np.save(path, value)
    return {
        "file": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "data_bytes": int(value.nbytes),
    }


def _write_packed_codes(
    root: Path, name: str, *, rows: int, outputs: int
) -> dict[str, object]:
    value = np.zeros((rows, outputs, 1), dtype=np.uint8)
    spec = _write_array(root, name, value)
    spec.update(
        {
            "encoding": "little-endian-packed-index-rows-v1",
            "index_bits": 4,
            "values_per_row": 1,
            "packed_row_bytes": 1,
            "decoded_dtype": "int16",
            "decoded_shape": [rows, outputs, 1],
            "decoded_data_bytes": rows * outputs * 2,
            "decoded_data_sha256": "0" * 64,
        }
    )
    return spec


def _tiny_pack(root: Path, *, layout: str = EXPECTED_LAYOUT_SHA256) -> Path:
    planes = root / "planes"
    planes.mkdir(parents=True)
    payloads: dict[str, dict[str, object]] = {}
    for projection, suffix, width in (("fused13", "13", 4), ("down", "2", 2)):
        payloads[projection] = {
            "d4_k16": {
                "family": "d4",
                "d": 4,
                "k": 16,
                "tensors": {
                    "expert_ids": _write_array(
                        planes, f"layer_000.d4_k16.{suffix}.expert_ids.npy", np.array([0, 1], dtype=np.int16)
                    ),
                    "codes": _write_packed_codes(
                        planes,
                        f"layer_000.d4_k16.{suffix}.codes.npy",
                        rows=2,
                        outputs=width,
                    ),
                    "scales": _write_array(
                        planes,
                        f"layer_000.d4_k16.{suffix}.scales.npy",
                        np.full((2, width, 1), 127, dtype=np.uint8),
                    ),
                    "codebooks": _write_array(
                        planes,
                        f"layer_000.d4_k16.{suffix}.codebooks.npy",
                        np.ones((1, 16, 4), dtype=np.float16),
                    ),
                    "codebook_index": _write_array(
                        planes,
                        f"layer_000.d4_k16.{suffix}.codebook_index.npy",
                        np.zeros(2, dtype=np.int16),
                    ),
                },
            }
        }
    meta = {
        "format": "p1016-true-c-native-planes-v1",
        "layer": 0,
        "E": 2,
        "K13": 4,
        "N13": 4,
        "K2": 2,
        "N2": 4,
        "family_codes": {"qtip2": 0, "qtip3": 1, "d4": 2, "native": 3},
        "tier13": ["d4_k16", "d4_k16"],
        "slot13": [0, 1],
        "family13": [2, 2],
        "tier2": ["d4_k16", "d4_k16"],
        "slot2": [0, 1],
        "family2": [2, 2],
        "payloads": payloads,
    }
    (planes / "layer_000.meta.json").write_text(json.dumps(meta))
    manifest = {
        "schema": "bs-pack",
        "schema_version": 1,
        "quant_method": "banana_smasher",
        "source_format": "p1016-true-c-native-planes-v1",
        "layers": [0],
        "tensor_layout_sha256": layout,
        "selected_payloads": {
            "schema": "bs-pack-selected-payloads-v1",
            "producer_stage": "smash export:selected-payload-wire-v1",
            "runtime_floor_bytes": 0,
            "dense_base_bytes": 0,
            "layers": {
                "0": {
                    projection: {
                        "tiers": list(meta[f"tier{suffix}"]),
                        "slots": list(meta[f"slot{suffix}"]),
                        "families": list(meta[f"family{suffix}"]),
                        "payloads": payloads[projection],
                    }
                    for projection, suffix in (("fused13", "13"), ("down", "2"))
                }
            },
        },
    }
    (root / "BANANA_PACK_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "banana_smasher",
                    "format": "bs-pack",
                    "format_version": 1,
                    "pack_root": ".",
                    "pack_manifest": "BANANA_PACK_MANIFEST.json",
                    "tensor_layout_sha256": layout,
                    "architecture": "sm_120",
                }
            }
        )
    )
    return root


def test_pack_binds_quant_config_root_and_every_declared_layer(tmp_path: Path) -> None:
    root = _tiny_pack(tmp_path / "model")
    pack = NativePlanePack.from_model_root(root)
    assert pack.root == root.resolve()
    assert pack.layers == (0,)
    assert pack.layout_sha256 == EXPECTED_LAYOUT_SHA256
    assert pack.meta_path(0) == root.resolve() / "planes/layer_000.meta.json"


def test_plane_loader_moves_named_planes_and_dispatches_projection(tmp_path: Path) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def dispatch(*, projection, x, expert_ids, state):
        calls.append((projection, tuple(x.shape), tuple(expert_ids.tolist())))
        width = state.output_width
        return torch.full((x.shape[0], width), 3.0, dtype=torch.float32)

    layer = NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    result = layer.forward(
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.tensor([1, 0]),
        "fused13",
    )
    assert result.shape == (2, 4)
    assert calls == [("fused13", (2, 4), (1, 0))]
    assert layer.state("fused13").families.tolist() == [2, 2]
    assert set(layer.state("fused13").payloads) == {"d4_k16"}
    assert layer.state("fused13").payloads["d4_k16"]["codes"].dtype == torch.uint8
    assert layer.state("fused13").pointer_tables["d4_index_bits"].tolist() == [4, 4]


def test_plane_forward_uses_capture_safe_async_expert_range_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    calls: list[tuple[bool, str]] = []

    def assert_async(condition: torch.Tensor, message: str) -> None:
        calls.append((bool(condition.item()), message))
        if not condition.item():
            raise RuntimeError(message)

    monkeypatch.setattr(torch.ops.aten._assert_async, "msg", assert_async)
    monkeypatch.setattr(
        torch,
        "any",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("host-synchronizing torch.any guard is forbidden")
        ),
    )
    layer = NativePlaneLayer(
        pack,
        0,
        device="cpu",
        dispatch=lambda **kwargs: torch.zeros(
            (kwargs["x"].shape[0], kwargs["state"].output_width)
        ),
    )

    result = layer.forward(torch.ones((2, 4)), torch.tensor([0, 1]), "fused13")

    assert result.shape == (2, 4)
    assert calls == [
        (True, "layer 0 fused13 expert id out of range"),
        (True, "layer 0 fused13 expert id out of range"),
    ]


def test_plane_forward_async_guard_rejects_out_of_range_expert(tmp_path: Path) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    layer = NativePlaneLayer(
        pack,
        0,
        device="cpu",
        dispatch=lambda **kwargs: kwargs["x"],
    )

    with pytest.raises(RuntimeError, match="expert id out of range"):
        layer.forward(torch.ones((2, 4)), torch.tensor([0, 2]), "fused13")


def test_plane_forward_safely_zeroes_batched_padding_sentinel(tmp_path: Path) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    observed_ids: list[torch.Tensor] = []

    def dispatch(**kwargs):
        safe_ids = kwargs["expert_ids"]
        observed_ids.append(safe_ids.clone())
        assert bool(torch.all((safe_ids >= 0) & (safe_ids < 2)))
        return torch.full(
            (safe_ids.numel(), kwargs["state"].output_width),
            3.0,
            dtype=torch.float32,
        )

    layer = NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    expert_ids = torch.tensor(([0, 1, 0, 1, 0, -1] * 16), dtype=torch.long)

    result = layer.forward(torch.ones((96, 4)), expert_ids, "fused13")

    assert observed_ids and observed_ids[0].shape == (96,)
    assert torch.equal(observed_ids[0][5::6], torch.zeros(16, dtype=torch.long))
    assert torch.count_nonzero(result[5::6]) == 0
    assert torch.all(result[:5] == 3)


def test_manifest_selection_is_the_only_payload_allocation_source(tmp_path: Path) -> None:
    root = _tiny_pack(tmp_path / "model")
    meta_path = root / "planes/layer_000.meta.json"
    meta = json.loads(meta_path.read_text())
    candidate = {
        "family": "d4",
        "d": 4,
        "k": 32,
        "tensors": {
            "expert_ids": _write_array(
                root / "planes", "layer_000.candidate.13.expert_ids.npy", np.array([0], dtype=np.int16)
            ),
            "codes": _write_array(
                root / "planes", "layer_000.candidate.13.codes.npy", np.zeros((1, 4, 1), dtype=np.int16)
            ),
        },
    }
    meta["payloads"]["fused13"]["candidate_nonfixed_tier"] = candidate
    meta_path.write_text(json.dumps(meta))

    pack = NativePlanePack.from_model_root(root)
    layer = NativePlaneLayer(pack, 0, device="cpu", dispatch=lambda **kwargs: kwargs["x"])

    assert set(layer.state("fused13").payloads) == {"d4_k16"}


def test_manifest_selection_rejects_duplicate_cell_before_plane_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    manifest_path = root / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_payloads"]["layers"]["0"]["fused13"]["slots"] = [0, 0]
    manifest_path.write_text(json.dumps(manifest))
    pack = NativePlanePack.from_model_root(root)
    allocations: list[object] = []

    def forbidden_tensor(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("plane allocation happened before exact-once binding gate")

    monkeypatch.setattr(NativePlaneLayer, "_tensor", forbidden_tensor)
    with pytest.raises(NativePlanePrerequisiteError, match="binds more than once"):
        NativePlaneLayer(pack, 0, device="cpu", dispatch=lambda **kwargs: kwargs["x"])

    assert allocations == []


def test_structural_memory_preflight_fails_before_tensor_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    manifest_path = root / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selected_payloads"]["dense_base_bytes"] = 101
    manifest["selected_payloads"]["runtime_floor_bytes"] = 17
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        native_planes,
        "_local_capacity_bytes",
        lambda: native_planes.OS_FLOOR_BYTES + 100,
    )
    allocations: list[object] = []

    def forbidden_tensor(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("tensor allocation happened before memory preflight")

    monkeypatch.setattr(torch, "tensor", forbidden_tensor)
    with pytest.raises(NativePlanePrerequisiteError) as caught:
        NativePlanePack.from_model_root(root)

    message = str(caught.value)
    assert allocations == []
    assert "STRUCTURAL_MEMORY_PREFLIGHT_OVER_BUDGET at t=0 before tensor allocation" in message
    assert "dense_base=101" in message
    assert "runtime_floor=17" in message
    assert "selected:codes=" in message
    assert "os_floor=4294967296" in message
    assert "producer_stage=smash export:selected-payload-wire-v1" in message
    assert "remediation=" in message


def test_streamed_plane_load_releases_mmap_pages_and_logs_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    plane = next((pack.root / "planes").glob("*.npy"))
    array = np.load(plane, allow_pickle=False)
    spec = {"file": plane.name, "shape": list(array.shape), "dtype": str(array.dtype)}
    calls: list[tuple[int, int, int]] = []

    def fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((offset, length, advice))

    monkeypatch.setattr(native_planes, "_posix_fadvise", fadvise)
    monkeypatch.setattr(native_planes, "_POSIX_FADV_DONTNEED", 4)
    native_planes._PLANE_LOAD_PROGRESS.clear()
    layer = object.__new__(NativePlaneLayer)
    layer.pack = pack
    layer.layer_index = 0
    layer.device = torch.device("cpu")

    with caplog.at_level("INFO", logger=native_planes.__name__):
        for _ in range(50):
            tensor = layer._tensor(spec)
            assert tensor.shape == array.shape

    assert calls == [(0, 0, 4)] * 50
    assert "BANANA_SMASHER_PLANE_LOAD_WATERMARK loaded=50" in caplog.text
    assert "total=10" in caplog.text
    assert "MemAvailable_kB=" in caplog.text


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    class QuantizationConfig:
        pass

    class FusedMoEMethodBase:
        def __init__(self, moe):
            self.moe = moe
            self.moe_kernel = None
            self.moe_quant_config = None

    class RoutedExperts:
        pass

    class LinearBase:
        pass

    class UnquantizedLinearMethod:
        pass

    class Fp8Config:
        def __init__(
            self,
            is_checkpoint_fp8_serialized=False,
            activation_scheme="dynamic",
            ignored_layers=None,
            weight_block_size=None,
            store_dtype=None,
        ):
            self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
            self.activation_scheme = activation_scheme
            self.ignored_layers = ignored_layers or []
            self.weight_block_size = weight_block_size
            self.store_dtype = store_dtype
            self.is_scale_e8m0 = False

    class Fp8LinearMethod:
        def __init__(self, quant_config):
            self.quant_config = quant_config

    class FusedMoEQuantConfig:
        @classmethod
        def make(cls, **kwargs):
            return cls()

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.quantization": ModuleType(
            "vllm.model_executor.layers.quantization"
        ),
        "vllm.model_executor.layers.quantization.base_config": ModuleType(
            "vllm.model_executor.layers.quantization.base_config"
        ),
        "vllm.model_executor.layers.quantization.fp8": ModuleType(
            "vllm.model_executor.layers.quantization.fp8"
        ),
        "vllm.model_executor.layers.fused_moe": ModuleType(
            "vllm.model_executor.layers.fused_moe"
        ),
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base": ModuleType(
            "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
        ),
        "vllm.model_executor.layers.fused_moe.config": ModuleType(
            "vllm.model_executor.layers.fused_moe.config"
        ),
        "vllm.model_executor.layers.linear": ModuleType(
            "vllm.model_executor.layers.linear"
        ),
    }
    modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizationConfig = QuantizationConfig
    modules["vllm.model_executor.layers.quantization.fp8"].Fp8Config = Fp8Config
    modules["vllm.model_executor.layers.quantization.fp8"].Fp8LinearMethod = (
        Fp8LinearMethod
    )
    modules["vllm.model_executor.layers.fused_moe"].RoutedExperts = RoutedExperts
    modules[
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base"
    ].FusedMoEMethodBase = FusedMoEMethodBase
    modules[
        "vllm.model_executor.layers.fused_moe.config"
    ].FusedMoEQuantConfig = FusedMoEQuantConfig
    modules["vllm.model_executor.layers.linear"].LinearBase = LinearBase
    modules[
        "vllm.model_executor.layers.linear"
    ].UnquantizedLinearMethod = UnquantizedLinearMethod
    # A real SM121 route test may have imported vllm.config earlier in this process.
    # Keep this helper fully fake and order-independent instead of leaking that module.
    monkeypatch.delitem(sys.modules, "vllm.config", raising=False)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("banana_smasher_plugin.quantization", None)


def test_native_moe_apply_uses_two_accelerated_projections_and_original_route_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = NativePlanePack.from_model_root(_tiny_pack(tmp_path / "model"))
    calls: list[tuple[str, tuple[int, ...]]] = []

    def dispatch(*, projection, x, expert_ids, state):
        calls.append((projection, tuple(expert_ids.tolist())))
        family = expert_ids.to(torch.float32).reshape(-1, 1) + 1
        if projection == "fused13":
            return torch.cat((family.expand(-1, 2), (family + 0.5).expand(-1, 2)), dim=1)
        return x.repeat(1, 2) * (family + 1)

    plane_layer = NativePlaneLayer(pack, 0, device="cpu", dispatch=dispatch)
    method = SimpleNamespace(native_layer=plane_layer, prefix="model.layers.0.ffn.experts")
    _install_fake_vllm(monkeypatch)
    from banana_smasher_plugin.quantization import BananaSmasherMoEMethod

    x = torch.tensor([[0.25, -0.75, 0.5, -0.5]], dtype=torch.float32)
    ids = torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=torch.long)
    weights = torch.tensor([[0.10, 0.20, 0.15, 0.25, 0.05, 0.25]], dtype=torch.float32)
    out = BananaSmasherMoEMethod.apply(method, object(), x, weights, ids, None, None)
    assert out.shape == x.shape
    assert calls == [
        ("fused13", (1, 0, 1, 0, 1, 0)),
        ("down", (1, 0, 1, 0, 1, 0)),
    ]


def test_qtip_transform_and_lut_are_exact_and_family_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    assert torch.allclose(_fwht(_fwht(x)), x)
    lut = _expanded_qtip_lut(torch.device("cpu"))
    assert lut.shape == (65536, 2)

    kernels = ModuleType("banana_smasher_plugin.p1016_kernels")

    def mixed_exact_gemv(kernel_input, *args):
        del args
        return kernel_input

    kernels.mixed_exact_gemv = mixed_exact_gemv
    monkeypatch.setitem(sys.modules, "banana_smasher_plugin.p1016_kernels", kernels)
    state = ProjectionState(
        "fused13",
        4,
        4,
        torch.tensor([0, 2], dtype=torch.int8),
        ("qtip2_2.0117", "d4_k16"),
        torch.tensor([0, 0]),
        {},
        {
            "su": torch.tensor([[2.0] * 4, [9.0] * 4]),
            "sv": torch.tensor([[3.0] * 4, [9.0] * 4]),
            "wscale": torch.tensor([0.5, 9.0]),
        },
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        lut,
    )
    run = _load_accelerated_dispatch()
    result = run(
        projection="fused13",
        x=x,
        expert_ids=torch.tensor([0, 1]),
        state=state,
    )
    assert torch.allclose(result[0].float(), x[0] * 3.0, atol=2e-2)
    assert torch.allclose(result[1].float(), x[1], atol=2e-2)


def test_quant_config_selects_only_stock_deepseek_routed_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    _install_fake_vllm(monkeypatch)
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
    from banana_smasher_plugin.quantization import (
        BananaSmasherMoEMethod,
        BananaSmasherQuantizationConfig,
    )

    raw = json.loads((root / "config.json").read_text())["quantization_config"]
    raw.update(
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    config = BananaSmasherQuantizationConfig.from_config(raw)
    config.maybe_update_config(str(root))
    assert config.get_quant_method(object(), "model.embed_tokens") is None
    linear_method = config.get_quant_method(LinearBase(), "model.layers.0.attn.q_proj")
    assert isinstance(linear_method, Fp8LinearMethod)
    assert linear_method.quant_config.is_checkpoint_fp8_serialized is True
    assert linear_method.quant_config.activation_scheme == "dynamic"
    assert linear_method.quant_config.weight_block_size == [128, 128]
    assert linear_method.quant_config.is_scale_e8m0 is True
    assert config.weight_block_size == [128, 128]
    assert config.dense_checkpoint_scale_fmt == "ue8m0"
    layer = RoutedExperts()
    layer.moe_config = SimpleNamespace()
    method = config.get_quant_method(layer, "model.layers.0.ffn.experts")
    assert isinstance(method, BananaSmasherMoEMethod)
    assert method.layer_index == 0
    with pytest.raises(NativePlanePrerequisiteError, match="DeepSeek-V4"):
        config.get_quant_method(layer, "model.layers.bad.ffn.experts")


def test_dense_weight_map_preflight_maps_stacked_attention_scales_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    index = {
        "weight_map": {
            "layers.0.attn.wq_a.scale": "model-00001-of-00001.safetensors",
            "layers.0.attn.wkv.scale": "model-00001-of-00001.safetensors",
            "layers.0.attn.wq_a.weight": "model-00001-of-00001.safetensors",
            "layers.0.attn.wkv.weight": "model-00001-of-00001.safetensors",
            "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00001.safetensors",
            "mtp.0.attn.q_norm.weight": "model-00001-of-00001.safetensors",
        }
    }
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    _install_fake_vllm(monkeypatch)
    from banana_smasher_plugin.quantization import (
        BananaSmasherQuantizationConfig,
        preflight_dense_weight_map,
    )

    raw = json.loads((root / "config.json").read_text())["quantization_config"]
    raw.update(
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    config = BananaSmasherQuantizationConfig.from_config(raw)
    config.model_root = root
    mapped_names = {
        "model.layers.0.attn.fused_wqa_wkv.weight_scale_inv",
        "model.layers.0.attn.fused_wqa_wkv.weight",
    }

    report = preflight_dense_weight_map(
        config,
        mapped_names,
        map_checkpoint_names=lambda names: [
            (
                f"model.{name}" if name.startswith("mtp.") else name.replace(
                    "layers.", "model.layers.", 1
                )
            ).replace(".scale", ".weight_scale_inv")
            for name in names
        ],
        mtp_enabled=False,
    )
    assert report["dense_checkpoint_names"] == 4
    assert report["mapped_named_parameters"] == 2
    assert report["excluded_disabled_mtp_names"] == 1
    assert config.dense_preflight_passed is True

    with pytest.raises(NativePlanePrerequisiteError, match="DENSE_WEIGHT_MAP_PREFLIGHT"):
        preflight_dense_weight_map(
            config,
            {"model.layers.0.attn.fused_wqa_wkv.weight"},
            map_checkpoint_names=lambda names: [
                name.replace("layers.", "model.layers.", 1).replace(
                    ".scale", ".weight_scale_inv"
                )
                for name in names
            ],
            mtp_enabled=False,
        )


def test_native_expert_allocation_is_deferred_until_dense_preflight_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    _install_fake_vllm(monkeypatch)
    import banana_smasher_plugin.quantization as quantization

    raw = json.loads((root / "config.json").read_text())["quantization_config"]
    raw.update(
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    config = quantization.BananaSmasherQuantizationConfig.from_config(raw)
    config.pack = NativePlanePack.from_model_root(root)
    method = object.__new__(quantization.BananaSmasherMoEMethod)
    method.quant_config = config
    method.layer_index = 0
    method.prefix = "model.layers.0.ffn.experts"
    method.native_layer = None
    class Layer:
        def buffers(self):
            return iter([SimpleNamespace(device=torch.device("cuda"))])

    layer = Layer()
    calls: list[tuple[object, ...]] = []

    class FakeNativePlaneLayer:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(quantization, "NativePlaneLayer", FakeNativePlaneLayer)
    quantization.BananaSmasherMoEMethod.create_weights(
        method,
        layer,
        num_experts=256,
        hidden_size=4096,
        intermediate_size_per_partition=2048,
        params_dtype=torch.bfloat16,
    )
    assert calls == []
    with pytest.raises(NativePlanePrerequisiteError, match="DENSE_WEIGHT_MAP_PREFLIGHT"):
        quantization.BananaSmasherMoEMethod.process_weights_after_loading(method, layer)
    assert calls == []
    config.dense_preflight_passed = True
    quantization.BananaSmasherMoEMethod.process_weights_after_loading(method, layer)
    assert len(calls) == 1
    assert layer.bs_native_plane_layer is method.native_layer


def test_registered_deepseek_loader_runs_full_preflight_before_weight_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.attn.wq_a.scale": "one.safetensors",
                    "layers.0.attn.wkv.scale": "one.safetensors",
                    "mtp.0.attn.q_norm.weight": "one.safetensors",
                }
            }
        )
    )
    _install_fake_vllm(monkeypatch)
    import banana_smasher_plugin.quantization as quantization

    raw = json.loads((root / "config.json").read_text())["quantization_config"]
    raw.update(
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    config = quantization.BananaSmasherQuantizationConfig.from_config(raw)
    config.model_root = root
    events: list[str] = []

    class Mapper:
        def apply_list(self, names):
            return [
                (
                    f"model.{name}" if name.startswith("mtp.") else name.replace(
                        "layers.", "model.layers.", 1
                    )
                ).replace(".scale", ".weight_scale_inv")
                for name in names
            ]

    class DeepseekV4ForCausalLM(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.received_vllm_config = vllm_config
            self.received_prefix = prefix
            self.model = SimpleNamespace(quant_config=config)
            self.hf_to_vllm_mapper = Mapper()
            self.register_parameter(
                "dense_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
            )

        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            return iter(
                [
                    (
                        "model.layers.0.attn.fused_wqa_wkv.weight_scale_inv",
                        self.dense_scale,
                    )
                ]
            )

        def load_weights(self, weights):
            del weights
            assert config.dense_preflight_passed is True
            events.append("load")
            return {"loaded"}

    model_module = ModuleType("vllm.models.deepseek_v4.nvidia.model")
    model_module.DeepseekV4ForCausalLM = DeepseekV4ForCausalLM
    monkeypatch.setitem(sys.modules, "vllm.models", ModuleType("vllm.models"))
    monkeypatch.setitem(
        sys.modules, "vllm.models.deepseek_v4", ModuleType("vllm.models.deepseek_v4")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.deepseek_v4.nvidia",
        ModuleType("vllm.models.deepseek_v4.nvidia"),
    )
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

    original_init_signature = inspect.signature(DeepseekV4ForCausalLM.__init__)
    quantization.install_deepseek_v4_dense_preflight()
    assert inspect.signature(DeepseekV4ForCausalLM.__init__) == original_init_signature
    model = DeepseekV4ForCausalLM(
        vllm_config=SimpleNamespace(speculative_config=None),
        prefix="model",
    )
    assert model.received_prefix == "model"
    assert model.load_weights([]) == {"loaded"}
    assert events == ["load"]

    enabled_model = DeepseekV4ForCausalLM(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method="mtp")
        )
    )
    with pytest.raises(NativePlanePrerequisiteError, match=r"model\.mtp\.0"):
        enabled_model.load_weights([])
    assert events == ["load"]


def test_real_vllm_initialize_model_forwards_vllm_config_after_plugin_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_utils = pytest.importorskip("vllm.model_executor.model_loader.utils")
    model_module = pytest.importorskip("vllm.models.deepseek_v4.nvidia.model")

    received: dict[str, object] = {}

    class DeepseekV4ForCausalLM(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            received["vllm_config"] = vllm_config
            received["prefix"] = prefix

        def load_weights(self, weights):
            return weights

    monkeypatch.setattr(model_module, "DeepseekV4ForCausalLM", DeepseekV4ForCausalLM)
    monkeypatch.setattr(
        loader_utils,
        "set_current_vllm_config",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(loader_utils, "record_metadata_for_reloading", lambda model: None)
    sys.modules.pop("banana_smasher_plugin.quantization", None)
    import banana_smasher_plugin.quantization as quantization

    quantization.install_deepseek_v4_dense_preflight()
    vllm_config = SimpleNamespace(quant_config=None)
    model = loader_utils.initialize_model(
        vllm_config,
        model_class=DeepseekV4ForCausalLM,
        model_config=SimpleNamespace(),
        prefix="model",
    )

    assert isinstance(model, DeepseekV4ForCausalLM)
    assert received == {"vllm_config": vllm_config, "prefix": "model"}


def test_real_vllm_stock_fp8_route_uses_triton_for_normalized_ue8m0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm.model_executor.kernels.linear")
    if not torch.cuda.is_available():
        pytest.skip("real stock-vLLM FP8 route regression requires CUDA")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.kernel import KernelConfig
    import vllm.distributed.parallel_state as parallel_state
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.layers.linear import LinearBase

    import banana_smasher_plugin.quantization as quantization

    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(rank_in_group=0, world_size=1),
    )
    vllm_config = VllmConfig(
        kernel_config=KernelConfig(linear_backend="auto"),
    )
    vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
    )
    config = quantization.BananaSmasherQuantizationConfig.from_config(
        {
            "quant_method": "banana_smasher",
            "format": "bs-pack",
            "format_version": 1,
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)

    with set_current_vllm_config(vllm_config):
        method = config.get_quant_method(layer, "model.layers.0.attn.q_proj")
        assert vllm_config.kernel_config.linear_backend == "triton"
        method.out_dtype = torch.bfloat16
        with torch.device("cuda"):
            method.create_weights(
                layer,
                input_size_per_partition=4096,
                output_partition_sizes=[1536],
                input_size=4096,
                output_size=1536,
                params_dtype=torch.bfloat16,
            )
        layer.weight.data.fill_(0.25)
        checkpoint_scale = torch.full(
            layer.weight_scale_inv.shape,
            127,
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.float8_e8m0fnu)
        layer.weight_scale_inv.data.copy_(checkpoint_scale)
        method.process_weights_after_loading(layer)
        output = method.apply(
            layer,
            torch.full((1, 4096), 0.25, dtype=torch.bfloat16, device="cuda"),
        )

    assert isinstance(method.fp8_linear, TritonFp8BlockScaledMMKernel)
    assert method.is_scale_e8m0 is False
    assert layer.weight_scale_inv.dtype == torch.float32
    assert output.shape == (1, 1536)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        torch.full_like(output, 256.0),
        rtol=0.0,
        atol=0.0,
    )


def test_real_vllm_stock_cutlass_scaled_mm_accepts_ue8m0_checkpoint_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm._custom_ops")
    if not torch.cuda.is_available():
        pytest.skip("real stock-vLLM scaled-mm regression requires CUDA")

    from vllm import _custom_ops as ops
    import vllm.distributed.parallel_state as parallel_state
    from vllm.model_executor.layers.quantization import fp8 as fp8_module
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        create_fp8_scale_parameter,
    )
    from vllm.model_executor.parameter import BlockQuantScaleParameter

    import banana_smasher_plugin.quantization as quantization

    monkeypatch.setattr(
        fp8_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )
    monkeypatch.setattr(
        parallel_state,
        "_TP",
        SimpleNamespace(rank_in_group=0, world_size=1),
    )
    config = quantization.BananaSmasherQuantizationConfig.from_config(
        {
            "quant_method": "banana_smasher",
            "format": "bs-pack",
            "format_version": 1,
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        }
    )
    method = fp8_module.Fp8LinearMethod(config.dense_fp8_config)
    runtime_scale = create_fp8_scale_parameter(
        BlockQuantScaleParameter,
        [128],
        128,
        [128, 128],
        None,
        scale_dtype=(
            torch.float8_e8m0fnu if method.is_scale_e8m0 else None
        ),
    ).cuda()
    checkpoint_scale = torch.full(
        runtime_scale.shape,
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(torch.float8_e8m0fnu)
    runtime_scale.data.copy_(checkpoint_scale)

    a = torch.full((1, 128), 0.25, dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.full(
        (128, 128), 0.25, dtype=torch.float8_e4m3fn, device="cuda"
    ).T
    output = ops.cutlass_scaled_mm(
        a,
        b,
        scale_a=torch.ones((1, 1), dtype=torch.float32, device="cuda"),
        scale_b=runtime_scale,
        out_dtype=torch.bfloat16,
    )
    assert runtime_scale.dtype == torch.float32
    assert output.shape == (1, 128)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        torch.full_like(output, 8.0),
        rtol=0.0,
        atol=0.0,
    )


def test_missing_plane_and_missing_kernel_fail_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tiny_pack(tmp_path / "model")
    (root / "planes/layer_000.meta.json").unlink()
    with pytest.raises(NativePlanePrerequisiteError, match=FAST_PATH_ERROR):
        NativePlanePack.from_model_root(root)

    root = _tiny_pack(tmp_path / "model2")
    pack = NativePlanePack.from_model_root(root)

    def missing_dispatch() -> None:
        raise NativePlanePrerequisiteError(
            f"{FAST_PATH_ERROR}: accelerated dispatch unavailable"
        )

    monkeypatch.setattr(native_planes, "_load_accelerated_dispatch", missing_dispatch)
    with pytest.raises(NativePlanePrerequisiteError, match="accelerated dispatch"):
        NativePlaneLayer(pack, 0, device="cpu", dispatch=None)


def test_layout_mismatch_fails_before_plane_load(tmp_path: Path) -> None:
    root = _tiny_pack(tmp_path / "model")
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"]["tensor_layout_sha256"] = "1" * 64
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(NativePlanePrerequisiteError, match="layout"):
        NativePlanePack.from_model_root(root)