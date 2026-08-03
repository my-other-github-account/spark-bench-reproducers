from __future__ import annotations

from pathlib import Path

import pytest


REQUIRED_RUNTIME_SENTINELS = (
    "DeepGemmFp8BlockScaledMMKernel",
    "BANANA_SMASHER_DSV4_O_PROJ_LAYOUT compute_capability=12.1 backend=deep_gemm_e8m0",
    "BANANA_SMASHER_VLLM_COMPILE_FAST_PATHS fuse_norm_quant=true fuse_act_quant=true",
    "BANANA_SMASHER_NATIVE_DISPATCH backend=cubit_iq3_vq",
)
FORBIDDEN_RUNTIME_SENTINELS = (
    "dense_fp8_recipe=disabled_sparse_only",
    "backend=stock_triton_fp8",
    "BANANA_SMASHER_MHC_BACKEND_OVERRIDE",
    "TritonFp8BlockScaledMMKernel",
    "_mixed_exact_gemv",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assert_v4_runtime_log_contract(log: str) -> None:
    missing = [needle for needle in REQUIRED_RUNTIME_SENTINELS if needle not in log]
    forbidden = [needle for needle in FORBIDDEN_RUNTIME_SENTINELS if needle in log]
    if missing or forbidden:
        raise AssertionError(
            f"V4 fast-path dispatch contract failed: missing={missing}, forbidden={forbidden}"
        )


def test_runtime_log_contract_accepts_only_v4_fast_paths() -> None:
    with pytest.raises(AssertionError, match="disabled_sparse_only"):
        assert_v4_runtime_log_contract(
            "TritonFp8BlockScaledMMKernel\n"
            "dense_fp8_recipe=disabled_sparse_only\n"
            "BANANA_SMASHER_MHC_BACKEND_OVERRIDE\n"
            "_mixed_exact_gemv\n"
        )
    assert_v4_runtime_log_contract("\n".join(REQUIRED_RUNTIME_SENTINELS))


def test_public_image_pins_dense_sm121_deepgemm_source() -> None:
    dockerfile = (_repo_root() / "docker/Dockerfile").read_text()
    assert "https://github.com/deepseek-ai/DeepGEMM.git" in dockerfile
    assert "a6b593d32eabfea81a699693a3e2ae1061cd835c" in dockerfile
    assert "https://github.com/jasl/DeepGEMM.git" not in dockerfile
    assert "7a7a41a1bac7dacabe74057e7600e59f98f85bce" not in dockerfile


def test_plugin_source_requires_dense_e8m0_and_compile_fusions() -> None:
    source = (
        _repo_root()
        / "banana-smasher-plugin/src/banana_smasher_plugin/quantization.py"
    ).read_text()
    assert "self.weight_block_size = list(weight_block_size)" in source
    assert 'self.dense_fp8_config.is_scale_e8m0 = scale_fmt == "ue8m0"' in source
    assert 'linear_backend = "triton"' not in source

    plugin = (
        _repo_root()
        / "banana-smasher-plugin/src/banana_smasher_plugin/__init__.py"
    ).read_text()
    assert "dense_fp8_recipe=enabled_e8m0" in plugin
    assert "dense_fp8_recipe=disabled_sparse_only" not in plugin
    assert "backend=stock_triton_fp8" not in plugin
    assert "BANANA_SMASHER_MHC_BACKEND_OVERRIDE" not in plugin
    assert "fuse_norm_quant=true fuse_act_quant=true" in source
