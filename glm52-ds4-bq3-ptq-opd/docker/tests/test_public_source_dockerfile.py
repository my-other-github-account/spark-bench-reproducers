from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/Dockerfile"
DEPLOY = ROOT / "README-DEPLOY.md"


def test_public_source_dockerfile_contract() -> None:
    text = DOCKERFILE.read_text()
    lower = text.lower()
    assert "vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b" in text
    assert "ee0da84a" in text
    assert "COPY banana-smasher /src/banana-smasher" in text
    assert "COPY banana-smasher-plugin /src/banana-smasher-plugin" in text
    assert "python3 -m build --wheel" in text
    assert "mkdir -p /wheel" in text
    assert "https://github.com/jasl/DeepGEMM.git" in text
    assert "7a7a41a1bac7dacabe74057e7600e59f98f85bce" in text
    assert "DG_FORCE_BUILD=1" in text
    assert "cuda-nvrtc-dev-13-0=13.0.88-1" in text
    assert "deep_gemm-2.5.0" in text
    assert "banana_smasher_plugin:register" not in text  # verified by the image script
    assert "libcudart_stub.so" in text
    assert "libcudart.so.13" in text
    assert "runtime_defaults.json" in text
    assert "cubins-sm120" in text and "cubins-e43" in text
    assert 'CMD ["vllm", "serve", "/model"' in text
    forbidden = (
        "vllm_runtime",
        "pyoverlay",
        "pythonpath",
        "ld_preload",
        "lic" + "ense",
        "sp" + "dx",
        "gene" + "sis",
        "hf_token",
    )
    for token in forbidden:
        assert token not in lower


def test_runtime_defaults_are_baked_and_parseable() -> None:
    defaults = json.loads((ROOT / "docker/runtime_defaults.json").read_text())
    assert defaults["model"] == "/model"
    assert defaults["serve"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
    assert defaults["serve"]["max_num_seqs"] == 16
    assert defaults["serve"]["kv_cache_dtype"] == "fp8"
    assert defaults["environment"]["VLLM_USE_DEEP_GEMM"] == "0"


def test_readme_has_literal_stranger_path_and_no_runtime_environment_flags() -> None:
    text = DEPLOY.read_text()
    assert "git clone --branch t_63769bff-public-source" in text
    assert "docker build --no-cache" in text
    run_lines = [line for line in text.splitlines() if line.startswith("docker run --rm --gpus all -v pack:/model")]
    assert len(run_lines) == 1
    assert "-p8000:8000" in run_lines[0]
    assert " -e " not in run_lines[0]
    assert run_lines[0].count(" -v ") == 1
    assert "smash export" in text and "smash verify" in text
