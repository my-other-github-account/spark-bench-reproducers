from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import banana_smasher_plugin


def _install_sm121_layout_modules(monkeypatch: pytest.MonkeyPatch):
    platforms = ModuleType("vllm.platforms")
    setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            get_device_capability=lambda: SimpleNamespace(major=12, minor=1)
        ),
    )
    o_proj = ModuleType("vllm.models.deepseek_v4.nvidia.ops.o_proj")
    fp8_utils = ModuleType(
        "vllm.model_executor.layers.quantization.utils.fp8_utils"
    )

    def stock_recipe():
        return (1, 1, 128), True

    def stock_o_proj(*args, **kwargs):
        return args, kwargs

    delegated: list[tuple[object, ...]] = []

    def stock_postprocess(
        wq,
        ws,
        quant_block_shape,
        use_e8m0,
        is_bmm=False,
        bmm_batch_size=0,
    ):
        delegated.append(
            (wq, ws, quant_block_shape, use_e8m0, is_bmm, bmm_batch_size)
        )
        return "stock"

    setattr(o_proj, "compute_fp8_einsum_recipe", stock_recipe)
    setattr(o_proj, "deep_gemm_fp8_o_proj", stock_o_proj)
    setattr(fp8_utils, "deepgemm_post_process_fp8_weight_block", stock_postprocess)
    setattr(fp8_utils, "_upcast_e8m0_to_fp32", lambda value: value.to(torch.float32))
    setattr(
        fp8_utils,
        "requant_weight_ue8m0_inplace",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.deepseek_v4.nvidia.ops.o_proj",
        o_proj,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        fp8_utils,
    )
    return o_proj, fp8_utils, stock_o_proj, delegated


def test_sm121_o_proj_preserves_deepgemm_and_uses_raw_e8m0_group_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    o_proj, fp8_utils, stock_o_proj, delegated = _install_sm121_layout_modules(
        monkeypatch
    )

    assert banana_smasher_plugin.configure_stock_deepseek_v4_o_proj() is True
    assert o_proj.deep_gemm_fp8_o_proj is stock_o_proj
    assert o_proj.compute_fp8_einsum_recipe() == ((1, 128, 128), False)

    wq = torch.zeros((256, 512), dtype=torch.float8_e4m3fn)
    ws = torch.ones((2, 4), dtype=torch.float32)
    grouped_weight, raw_scales = fp8_utils.deepgemm_post_process_fp8_weight_block(
        wq,
        ws,
        (128, 128),
        False,
        is_bmm=True,
        bmm_batch_size=2,
    )
    assert grouped_weight.shape == (2, 128, 512)
    assert raw_scales.shape == (2, 1, 4)
    assert raw_scales.is_contiguous()
    assert delegated == []

    assert (
        fp8_utils.deepgemm_post_process_fp8_weight_block(
            wq, ws, (128, 128), False, is_bmm=False
        )
        == "stock"
    )
    assert len(delegated) == 1


def test_sm121_mhc_preserves_stock_public_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep_gemm = ModuleType("vllm.utils.deep_gemm")

    def stock_prenorm(*args, **kwargs):
        return args, kwargs

    setattr(deep_gemm, "tf32_hc_prenorm_gemm", stock_prenorm)
    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)

    assert banana_smasher_plugin.configure_stock_mhc_backend() is False
    assert deep_gemm.tf32_hc_prenorm_gemm is stock_prenorm
