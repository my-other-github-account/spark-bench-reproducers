from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT_HAND_SEAL = "ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5"
P1321_SOURCE = {
    "vendor/p1321/runtime/moe_vq_triton.py":
        "7b25bf83a0e6a36a50af2eab2c4a5305b7f146824e2acf183dcbd82ca760f13a",
    "vendor/p1321/runtime/moe_w2_cubit.py":
        "b5532974081e73c7abeda61d9b6e2460ea2a2a50958bf0f1e89a678a204be1d7",
    "vendor/p1321/vq_warp_m4/csrc/vq_warp_gemv.cu":
        "4ff296f7e42e2a906543aa4563fc50e1e04d58e6d41de46cda9bce4ae686fef5",
}


def test_readme_has_exactly_three_copy_paste_commands() -> None:
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```(?:sh|bash)?\n(.*?)```", text, flags=re.S)
    commands = [
        line.strip()
        for block in blocks
        for line in block.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert commands == [
        "git clone https://github.com/my-other-github-account/spark-bench-reproducers.git && cd spark-bench-reproducers/glm52-ds4-bq3-ptq-opd/banana-smasher",
        "./build.sh",
        "docker run --gpus all -v <pack>:/model:ro -p 8000:8000 genesis-serve:golden vllm serve /model",
    ]


def test_root_dockerfile_is_multistage_stock_vllm_and_seal_labeled() -> None:
    text = (ROOT / "Dockerfile").read_text()
    assert len(re.findall(r"^FROM ", text, flags=re.M)) >= 2
    assert "vllm/vllm-openai:v0.24.0" in text
    assert f'io.genesis.parent-hand.ladder-seal.sha256="{PARENT_HAND_SEAL}"' in text
    assert 'ENTRYPOINT []' in text
    assert 'CMD ["vllm", "serve", "/model"]' in text
    assert "HEALTHCHECK" in text
    assert "FLASHINFER_DISABLE_JIT=1" in text
    assert "/root/.cache/flashinfer/0.6.14/cached_ops" in text


def test_build_script_is_zero_argument_and_writes_digest_receipt() -> None:
    text = (ROOT / "build.sh").read_text()
    assert "genesis-serve:golden" in text
    assert "docker build" in text
    assert "BUILD_RECEIPT.json" in text
    assert "image inspect" in text
    assert PARENT_HAND_SEAL in text
    subprocess.run(["bash", "-n", str(ROOT / "build.sh")], check=True)


def test_p1321_vendor_hashes_are_exact() -> None:
    import hashlib

    for relative, expected in P1321_SOURCE.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_pack_format_and_validate_pack_surface_exist() -> None:
    spec = (ROOT / "PACK_FORMAT.md").read_text()
    assert "bs-pack-v1" in spec
    assert "meta.json" in spec
    assert "SHA-256" in spec
    completed = subprocess.run(
        ["python3", "-m", "banana_smasher.cli", "validate-pack", "--help"],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_health_contract_uses_current_c1_and_c4_class_bars() -> None:
    perf = (ROOT / "golden-container/golden_perf_check.py").read_text()
    health = (ROOT / "golden-container/healthcheck.py").read_text()
    assert 'default=13.0' in perf
    assert 'default=27.0' in perf
    assert '"--c1-bar", "13.0"' in health
    assert '"--c4-bar", "27.0"' in health
    assert '"c1_minus_bar"' in perf
    assert '"c4_minus_bar"' in perf


def test_source_manifest_binds_p1321_and_parent_hand_seal() -> None:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    assert manifest["parent_hand_ladder_seal_sha256"] == PARENT_HAND_SEAL
    for relative, expected in P1321_SOURCE.items():
        assert manifest["files"][relative]["sha256"] == expected


def test_vendored_wheel_manifest_binds_p1336_build_lineage() -> None:
    manifest = json.loads((ROOT / "VENDORED_WHEELS.json").read_text())
    assert manifest["schema"] == "banana-smasher-wheel-inputs-v1"
    assert manifest["p1336"]["image_id"] == (
        "sha256:b8669a5984dee524e082d0cc0bdfcbb20e98305b32b3aaf254f7fa434e88d257"
    )
    assert manifest["p1336"]["receipt_sha256"] == (
        "a3af06014c073d04dee70978d98a32e302c3e77a647dd6dc6881e777f3d6d4d5"
    )
    names = {row["name"] for row in manifest["wheel_inputs"]}
    assert names == {"banana-smasher", "deep-gemm", "vq-warp-gemv"}
