from __future__ import annotations

import json
import sys
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
    return {"file": name, "shape": list(value.shape), "dtype": str(value.dtype)}


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
                    "codes": _write_array(
                        planes,
                        f"layer_000.d4_k16.{suffix}.codes.npy",
                        np.zeros((2, width, 1), dtype=np.int16),
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
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from banana_smasher_plugin.quantization import (
        BananaSmasherMoEMethod,
        BananaSmasherQuantizationConfig,
    )

    raw = json.loads((root / "config.json").read_text())["quantization_config"]
    config = BananaSmasherQuantizationConfig.from_config(raw)
    config.maybe_update_config(str(root))
    assert config.get_quant_method(object(), "model.embed_tokens") is None
    linear_method = config.get_quant_method(LinearBase(), "model.layers.0.attn.q_proj")
    assert isinstance(linear_method, UnquantizedLinearMethod)
    layer = RoutedExperts()
    layer.moe_config = SimpleNamespace()
    method = config.get_quant_method(layer, "model.layers.0.ffn.experts")
    assert isinstance(method, BananaSmasherMoEMethod)
    assert method.layer_index == 0
    with pytest.raises(NativePlanePrerequisiteError, match="DeepSeek-V4"):
        config.get_quant_method(layer, "model.layers.bad.ffn.experts")


def test_missing_plane_and_missing_kernel_fail_loudly(tmp_path: Path) -> None:
    root = _tiny_pack(tmp_path / "model")
    (root / "planes/layer_000.meta.json").unlink()
    with pytest.raises(NativePlanePrerequisiteError, match=FAST_PATH_ERROR):
        NativePlanePack.from_model_root(root)

    root = _tiny_pack(tmp_path / "model2")
    pack = NativePlanePack.from_model_root(root)
    with pytest.raises(NativePlanePrerequisiteError, match="accelerated dispatch"):
        NativePlaneLayer(pack, 0, device="cpu", dispatch=None)


def test_layout_mismatch_fails_before_plane_load(tmp_path: Path) -> None:
    root = _tiny_pack(tmp_path / "model")
    config = json.loads((root / "config.json").read_text())
    config["quantization_config"]["tensor_layout_sha256"] = "1" * 64
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(NativePlanePrerequisiteError, match="layout"):
        NativePlanePack.from_model_root(root)
