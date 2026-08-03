from __future__ import annotations

import importlib.metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version
import pytest


def test_plugin_requires_first_quack_release_compatible_with_stock_cutlass_45() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = [Requirement(item) for item in pyproject["project"]["dependencies"]]
    quack = next((item for item in requirements if item.name == "quack-kernels"), None)

    assert quack is not None, (
        "plugin metadata must install the public quack/CUTLASS compatibility contract"
    )
    assert Version("0.4.0") not in quack.specifier
    assert Version("0.4.1") in quack.specifier
    assert Version("0.5.0") in quack.specifier
    assert Version("0.5.1") not in quack.specifier


def test_real_stock_deepseek_v4_fused_indexer_imports_supported_quack() -> None:
    module = pytest.importorskip(
        "vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl"
    )
    cutlass_version = Version(importlib.metadata.version("nvidia-cutlass-dsl"))
    quack_version = Version(importlib.metadata.version("quack-kernels"))
    assert cutlass_version == Version("4.5.2"), (
        f"stock vLLM 0.24.0 contract changed: nvidia-cutlass-dsl={cutlass_version}"
    )
    assert Version("0.4.1") <= quack_version < Version("0.5.1"), (
        "stock CuTeDSL fused indexer requires quack-kernels>=0.4.1,<0.5.1 when "
        f"nvidia-cutlass-dsl={cutlass_version}; found quack-kernels={quack_version}"
    )

    assert callable(module.fused_indexer_q_rope_quant_fp8_cutedsl)
