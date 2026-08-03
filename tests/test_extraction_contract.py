from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_source_surfaces_exist() -> None:
    required = {
        "banana-smasher/pyproject.toml",
        "banana-smasher/src/banana_smasher/cli.py",
        "banana-smasher/src/banana_smasher/contract.py",
        "banana-smasher/src/banana_smasher/loader.py",
        "banana-smasher/src/banana_smasher/repack.py",
        "banana-smasher/src/banana_smasher/repair.py",
        "banana-smasher/src/banana_smasher/validation.py",
        "banana-smasher/schema/bs-pack-v1.schema.json",
        "banana-smasher-plugin/pyproject.toml",
        "banana-smasher-plugin/src/banana_smasher_plugin/contract.py",
        "banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py",
        "banana-smasher-plugin/src/banana_smasher_plugin/p1016_kernels.py",
        "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy",
        "banana-smasher-plugin/src/banana_smasher_plugin/quantization.py",
        "banana-smasher-plugin/src/banana_smasher_plugin/repair.py",
        "docker/Dockerfile",
        "docker/patches/flashinfer-real-libcudart.patch",
        "docker/runtime_defaults.json",
        "docker/scripts/verify_public_image.py",
        "docker/scripts/write_package_receipt.py",
    }
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing, f"missing retained source surfaces: {missing}"


def test_all_runtime_aot_assets_exist() -> None:
    sm120 = list((ROOT / "banana-smasher/kernels/cubins-sm120").glob("*.cubin"))
    e43 = list((ROOT / "banana-smasher/kernels/cubins-e43").glob("*.cubin"))
    autotune = list(
        (ROOT / "banana-smasher/vendor/flashinfer-autotune").rglob("autotune_configs.json")
    )
    assert len(sm120) == 26
    assert len(e43) == 6
    assert len(autotune) == 35
    assert (
        ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy"
    ).is_file()


def test_documented_product_path_and_examples_exist() -> None:
    required = (
        "README.md",
        "ACCELERATIONS.md",
        "PROVENANCE.md",
        "examples/export_model.sh",
        "examples/build_image.sh",
        "examples/serve.sh",
        "examples/smoke_api.py",
        "runtime/ACCELERATION_MANIFEST.json",
        "provenance/SOURCE_INVENTORY.json",
    )
    assert all((ROOT / path).is_file() for path in required)
    readme = (ROOT / "README.md").read_text()
    for command in ("smash export", "smash verify", "docker build", "docker run", "/v1/chat/completions"):
        assert command in readme
    export = (ROOT / "examples/export_model.sh").read_text()
    assert '--runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}"' in export
    assert "RUNTIME_FLOOR_BYTES:-" not in export
    assert "smash verify" in export
    serve = (ROOT / "examples/serve.sh").read_text()
    assert "docker run" in serve and "/model" in serve and "8000:8000" in serve


def test_acceleration_manifest_is_exact_and_test_mapped() -> None:
    manifest = json.loads((ROOT / "runtime/ACCELERATION_MANIFEST.json").read_text())
    assert manifest["schema"] == "banana-smasher-acceleration-manifest-v1"
    assert manifest["source_commit"] == "c00714c6803f7e2de7a95d103dbe172236b22adf"
    entries = manifest["accelerations"]
    by_id = {entry["id"]: entry for entry in entries}
    required_ids = {
        "bs-pack-export-verify",
        "stock-vllm-general-plugin",
        "native-plane-p1016",
        "p1016-cutedsl-tlut",
        "sm121-deepgemm-dense-e8m0",
        "sm121-deepgemm-sparse-indexer",
        "sm121-persistent-topk",
        "sm121-v4-attention-flashinfer",
        "stock-deepgemm-mhc",
        "flashinfer-sparse-decode-compat",
        "sm120-aot-cubins",
        "e43-aot-cubins",
        "flashinfer-autotune-cache",
        "real-libcudart-link",
    }
    assert set(by_id) == required_ids
    for entry in entries:
        assert entry["source"]
        assert entry["build_dependency_or_asset"]
        assert entry["runtime_activation"]
        assert entry["tests"]
        for path in entry["source"] + entry["tests"]:
            assert (ROOT / path).exists(), f"manifest path missing: {entry['id']}: {path}"


def test_runtime_pins_hooks_assets_and_exact_command() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    required = (
        "vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b",
        "ARG VLLM_UPSTREAM_REV=ee0da84a",
        "d020372b068f335e2fe427372e134977a2235c49",
        "b34f49255f1640542da91665f58558a3e5e308f1",
        "76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9",
        "0c5fda59bb6fa71eae875693a024bb0fb37ba7d6",
        "a6b593d32eabfea81a699693a3e2ae1061cd835c",
        "flashinfer-real-libcudart.patch",
        "libcudart.so.13",
        "COPY banana-smasher/kernels/cubins-sm120",
        "COPY banana-smasher/kernels/cubins-e43",
        "COPY banana-smasher/vendor/flashinfer-autotune/0.6.14/121a",
        "python3 /opt/banana-smasher/bin/verify_public_image.py",
        "/opt/banana-smasher/provenance/package-sbom.json",
        'CMD ["vllm", "serve", "/model"',
    )
    assert all(value in dockerfile for value in required)
    pyproject = (ROOT / "banana-smasher-plugin/pyproject.toml").read_text()
    assert '[project.entry-points."vllm.general_plugins"]' in pyproject
    assert 'banana_smasher_plugin = "banana_smasher_plugin:register"' in pyproject
    plugin = (ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/__init__.py").read_text()
    hooks = (
        "configure_flashinfer_sparse_mla_signature_compat()",
        "configure_stock_deepseek_v4_attention_backend()",
        "configure_sparse_indexer_deep_gemm_backend()",
        "configure_sparse_indexer_topk_backend()",
        "configure_stock_deepseek_v4_o_proj()",
        "install_deepseek_v4_dense_preflight()",
    )
    assert all(hook in plugin for hook in hooks)


def test_source_inventory_covers_and_hashes_every_retained_source_file() -> None:
    inventory = json.loads((ROOT / "provenance/SOURCE_INVENTORY.json").read_text())
    assert inventory["source_commit"] == "c00714c6803f7e2de7a95d103dbe172236b22adf"
    entries = inventory["files"]
    paths = {entry["path"] for entry in entries}
    retained_roots = (ROOT / "banana-smasher", ROOT / "banana-smasher-plugin", ROOT / "docker")
    actual = {
        path.relative_to(ROOT).as_posix()
        for base in retained_roots
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
    }
    assert paths == actual
    for entry in entries:
        path = ROOT / entry["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == entry["output_sha256"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
        assert not Path(entry["source_path"]).is_absolute()
        assert entry["source_path"].split("/", 1)[0] in {
            "banana-smasher",
            "banana-smasher-plugin",
            "docker",
        }
        assert entry["byte_preserved"] == (entry["source_sha256"] == digest)


def test_no_campaign_private_or_original_work_license_material_leaks() -> None:
    forbidden_names = {
        "NIGHTLY_" + "SEALED_RESULTS.md",
        "PROJECT_" + "REFERENCE.md",
        "PIPE" + "LINE.md",
        "GAP_" + "LEDGER.md",
        "PUBLICATION_" + "TRANSFORM.json",
        "Dockerfile.flashinfer-" + "cudart",
    }
    forbidden_parts = {"ref" + "erence", "rece" + "ipts", "whe" + "els", "__py" + "cache__"}
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".ruff_cache" not in path.parts
        and "__pycache__" not in path.parts
    ]
    assert not [path for path in files if path.name in forbidden_names]
    assert not [path for path in files if forbidden_parts.intersection(path.parts)]
    assert not [path for path in files if path.name.lower().startswith("license")]

    text_files = [
        path
        for path in files
        if path.suffix.lower() not in {".cubin", ".npy", ".pyc"}
    ]
    content = "\n".join(path.read_text(errors="ignore") for path in text_files)
    forbidden_content = (
        "NIGHTLY_" + "SEALED_RESULTS",
        "PROJECT_" + "REFERENCE",
        "GAP_" + "LEDGER",
        "PUBLICATION_" + "TRANSFORM",
        "SPDX-" + "License-Identifier",
        "/Use" + "rs/",
        "/ho" + "me/",
    )
    assert all(value not in content for value in forbidden_content)
    assert re.search(r"\bt_[0-9a-f]{8}\b", content) is None
    assert re.search(r"\b(?:10|192\.168)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", content) is None
    assert re.search(r"(?im)^\s*license\s*=", content) is None
    assert ("Lic" + "ense ::") not in content

    for script in (ROOT / "examples").glob("*"):
        script_text = script.read_text(errors="ignore")
        assert ("/Use" + "rs/") not in script_text
        assert ("/ho" + "me/") not in script_text
        assert re.search(r"\b(?:10|192\.168)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", script_text) is None
