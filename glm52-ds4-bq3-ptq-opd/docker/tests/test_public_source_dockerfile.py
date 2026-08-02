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
    assert "python3 -m pytest -q" in text
    assert "/src/banana-smasher/tests" in text
    assert "/src/banana-smasher-plugin/tests" in text
    assert "mkdir -p /wheel" in text
    assert "https://github.com/flashinfer-ai/flashinfer.git" in text
    assert "d020372b068f335e2fe427372e134977a2235c49" in text
    assert "b34f49255f1640542da91665f58558a3e5e308f1" in text
    assert "76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9" in text
    assert "0c5fda59bb6fa71eae875693a024bb0fb37ba7d6" in text
    assert "BUILD_NVEP=0" in text
    uninstall = "pip uninstall -y flashinfer-cubin flashinfer-jit-cache"
    install = "/tmp/wheels/flashinfer_python-0.6.17-py3-none-any.whl"
    assert uninstall in text
    assert text.index(uninstall) < text.index(install, text.index("FROM ${VLLM_IMAGE} AS runtime"))
    assert 'for name in ("flashinfer_cubin","flashinfer_jit_cache")' in text
    assert 'find_spec("flashinfer_cubin") is None' in text
    assert 'find_spec("flashinfer_jit_cache") is None' in text
    assert '"flashinfer-cubin" not in names' in text
    assert '"flashinfer-jit-cache" not in names' in text
    assert "FLASHINFER_DISABLE_VERSION_CHECK" not in text
    assert "flashinfer-python==0.6.12" not in text
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


def test_runtime_removes_stale_flashinfer_binary_provider_namespaces() -> None:
    text = DOCKERFILE.read_text()
    uninstall = "pip uninstall -y flashinfer-cubin flashinfer-jit-cache"
    remove_cubin = "rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin"
    remove_jit_cache = "rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache"
    install_source = "/tmp/wheels/flashinfer_python-0.6.17-py3-none-any.whl"

    assert uninstall in text
    assert remove_cubin in text
    assert remove_jit_cache in text
    assert text.index(uninstall) < text.index(remove_cubin) < text.index(install_source, text.index(remove_cubin))
    assert text.index(uninstall) < text.index(remove_jit_cache) < text.index(install_source, text.index(remove_jit_cache))


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
