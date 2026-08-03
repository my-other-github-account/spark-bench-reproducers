from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c00714c6803f7e2de7a95d103dbe172236b22adf"

EXPECTED_SM120 = {
    "mla_prefill_state.cubin": "e4ad2dc0ceb162c66ffaff33f20b454cfe34c65dc004e6bfe263e176e5b5d748",
    "moe_w2_mm_k1024.cubin": "3a75e609012b658b32bc284f8451f5392bb7600f6429194987a15499c248b437",
    "moe_w2_mm_k2048.cubin": "bcd715fb04bb7efb75a65594e8d96bba24452ba82da9bdec3b5db34be3366b1f",
    "moe_w2_mm_k4096.cubin": "19570b90424162e9f22b86ebdb0cd521ba9143013e7d284fef257d7724da3e11",
    "moe_w2_mm_k512.cubin": "6ce2965080048b8b82218ce6762150315e1dce5c1277b5683e193cd7849dc932",
    "moe_w2_mm_k6144.cubin": "1b0b39b9d89a56ecf0b08b871879e7d684b93594e12571e55548f1f95627b4aa",
    "moe_w2_mm_mc2_k1024.cubin": "727c80cad157b9716df8a94cf86fb1be169ef1d792804e355f5a8ffd8ad7f154",
    "moe_w2_mm_mc2_k2048.cubin": "f2e1093d106c79c122548c32d22f8ef95d5f844d0f83a483c9dee1ce58c65452",
    "moe_w2_mm_mc2_k4096.cubin": "100a3255fcfd5947ab9ad593bccb37433b5c70aeff65995334973a97dd11071c",
    "moe_w2_mm_mc2_k512.cubin": "ff99d3141a6503419cdec82efddde804ee24004035a79176d2057810b2a06998",
    "moe_w2_mm_mc2_k6144.cubin": "385f71df93b5b04d5634c84e7f9593bc340b375663ae59e600bf38fcf1a7ba18",
    "moe_w2_mm_mc4_k1024.cubin": "8744ab7a2c4184d36b8d5a09b65a0a434cea227505f6fb435a886322163e3e96",
    "moe_w2_mm_mc4_k2048.cubin": "f5c6db6eb4cfd2c9706a51f9303c68adaeabb123d8fb43cbb4d54f09fce6b829",
    "moe_w2_mm_mc4_k4096.cubin": "ac24c451d25f9e0cad4fc4f1cd05bba36eeaebc81a3552da7fff71187bc15e31",
    "moe_w2_mm_mc4_k512.cubin": "6576d059924a9824f6b8260ac662dc3eaf31a14c7a09d99708e1458615450b90",
    "moe_w2_mm_mc4_k6144.cubin": "8cbb0337a2c9180e3a412221cc57e57ee62d5a5e8797666542bbc778e6cfa724",
    "moe_w2_mm_mc4afrag_k1024.cubin": "4c7c1f513ff32411f0eba5c3b05a1b4d65bee74c1b0ac4b43f237717366d32e9",
    "moe_w2_mm_mc4afrag_k2048.cubin": "6e7eff0157ca2964d84b2d7ea30066babb81076cce0575272186d3e8412202e3",
    "moe_w2_mm_mc4afrag_k4096.cubin": "fcbe9b106139a10c9dbca07499adaaddc0fccdcff6ccfa87daaf8873b225e0b8",
    "moe_w2_mm_mc4afrag_k512.cubin": "47bb62018c969088ae3051b7b845c431577510d189ec0959e3d0704f3ffeacf5",
    "moe_w2_mm_mc4afrag_k6144.cubin": "bcce10f64998bc4f789b02ced4e907aa3ff2a3b8dab4ef9c9122e0b67925831a",
    "moe_w4_mm_k1024.cubin": "562f38acd649113b2b30e799f17d2a6d0a8ea5056f62c4ad359d6a1eba845371",
    "moe_w4_mm_k2048.cubin": "f8eac7c80f6abcd933f359f7ab5ebba7ddabbcc55ad7fc9d5ce4f5ab4069a7de",
    "moe_w4_mm_k4096.cubin": "f8668076b860030c628af8965e6465e65a43cc3b3764a6f54712b1ee99ad13be",
    "moe_w4_mm_k512.cubin": "44f98dfa84aaab7daca195b53027ec97aa76698336f18ecb840fd397a0972fc6",
    "moe_w4_mm_k6144.cubin": "1c2c6c8d157717c15a25999684fd552e4c074e623bad81f50e6d424f2a7de7f1",
}
EXPECTED_E43 = {
    "moe_w3_mm_k2048.cubin": "329ead7686b64a0b38a73a28ef2877e432da8ee8558d7a2c69b2ea563276d68f",
    "moe_w3_mm_k4096.cubin": "923812fc229524465af15869e5e16ad570dd0610bb8413feb43f3630e2245751",
    "moe_w3_mm_mc4_k2048.cubin": "7fb4f67af94d88d40ca7f42e13fc89b281335cf8b4db037dd943e219daac237b",
    "moe_w3_mm_mc4_k4096.cubin": "f9f8cd5f207da23440ea420a7e13068ca10291a181cb05435417f95085db0402",
    "moe_w3_mm_mc4afrag_k2048.cubin": "5feb524786cd38ffc7c91bd22b76cbf63472ba1e49f88cc70249c9f28b5db903",
    "moe_w3_mm_mc4afrag_k4096.cubin": "7d628b0656ce24fac21276aeab21ca26d3674570f07bce3d328548a249624a3a",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual_assets(relative: str) -> dict[str, str]:
    base = ROOT / relative
    return {path.name: _sha(path) for path in sorted(base.glob("*.cubin"))}


def test_active_binary_assets_are_exact_source_members() -> None:
    assert _actual_assets("banana-smasher/kernels/cubins-sm120") == EXPECTED_SM120
    assert _actual_assets("banana-smasher/kernels/cubins-e43") == EXPECTED_E43
    assert sum(path.stat().st_size for path in (ROOT / "banana-smasher/kernels/cubins-sm120").glob("*.cubin")) == 490800
    assert sum(path.stat().st_size for path in (ROOT / "banana-smasher/kernels/cubins-e43").glob("*.cubin")) == 154368
    tlut = ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy"
    assert tlut.stat().st_size == 4224
    assert _sha(tlut) == "ae499158dab839d08c29de0b4dde4a6b303e781dc24e62e4b33ac50309bc5147"


def test_asset_manifest_drives_exact_image_admission() -> None:
    manifest = json.loads((ROOT / "runtime/ASSET_MANIFEST.json").read_text())
    assert manifest["schema"] == "banana-smasher-active-assets-v1"
    assert manifest["source_commit"] == SOURCE_COMMIT
    groups = manifest["groups"]
    for group, expected, total in (
        ("sm120_cubins", EXPECTED_SM120, 490800),
        ("e43_cubins", EXPECTED_E43, 154368),
    ):
        records = groups[group]
        assert {record["name"]: record["sha256"] for record in records} == expected
        assert sum(record["bytes"] for record in records) == total
    assert manifest["qtip_tlut"] == {
        "bytes": 4224,
        "name": "qtip_tlut.npy",
        "sha256": "ae499158dab839d08c29de0b4dde4a6b303e781dc24e62e4b33ac50309bc5147",
    }
    verifier = (ROOT / "docker/scripts/verify_public_image.py").read_text()
    assert "ASSET_MANIFEST.json" in verifier
    assert "expected one exact asset set" in verifier
    assert "AOT cubin set is incomplete" not in verifier


def test_stale_flashinfer_cache_and_release_tool_are_excluded() -> None:
    assert not (ROOT / "banana-smasher/tools/seal_release.py").exists()
    assert not (ROOT / "banana-smasher/vendor/flashinfer-autotune/0.6.14").exists()
    assert not list((ROOT / "banana-smasher").rglob("autotune_configs.json"))
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    assert "flashinfer-autotune/0.6.14" not in dockerfile
    assert "flashinfer_autotune_cache/0.6.14" not in dockerfile
    excluded = json.loads((ROOT / "provenance/EXCLUDED_FLASHINFER_CACHE.json").read_text())
    assert excluded["source_commit"] == SOURCE_COMMIT
    assert excluded["reason"] == "flashinfer-version-mismatch"
    assert excluded["excluded_version"] == "0.6.14"
    assert excluded["required_version"] == "0.6.17"
    assert excluded["file_count"] == 35
    assert excluded["bytes"] == 164546
    assert excluded["canonical_records_sha256"] == (
        "e9bbe2d71dd3465e87300a52c037f6b9b68b05b8879af9394ff31fbd535e1c84"
    )
    assert len(excluded["members"]) == 35


def _cache_validator():
    path = ROOT / "docker/scripts/validate_flashinfer_cache.py"
    spec = importlib.util.spec_from_file_location("validate_flashinfer_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache_file(root: Path, version: str, arch: str, metadata_version: str) -> Path:
    path = root / version / arch / ("a" * 64) / "autotune_configs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"_metadata": {"flashinfer_version": metadata_version}, "op": ["runner", 1]}))
    return path


def test_cache_validator_accepts_only_matching_version_path_and_metadata(tmp_path: Path) -> None:
    validator = _cache_validator()
    valid = _cache_file(tmp_path / "valid", "0.6.17", "121a", "0.6.17")
    result = validator.validate_cache_path(valid.parents[1], expected_version="0.6.17", expected_arch="121a")
    assert result["status"] == "VALID"
    assert result["file_count"] == 1

    wrong_path = _cache_file(tmp_path / "wrong-path", "0.6.14", "121a", "0.6.14")
    with pytest.raises(ValueError, match="cache path version/architecture mismatch"):
        validator.validate_cache_path(wrong_path.parents[1], expected_version="0.6.17", expected_arch="121a")

    wrong_metadata = _cache_file(tmp_path / "wrong-metadata", "0.6.17", "121a", "0.6.14")
    with pytest.raises(ValueError, match="flashinfer_version mismatch"):
        validator.validate_cache_path(wrong_metadata.parents[1], expected_version="0.6.17", expected_arch="121a")

    wrong_arch = _cache_file(tmp_path / "wrong-arch", "0.6.17", "120a", "0.6.17")
    with pytest.raises(ValueError, match="cache path version/architecture mismatch"):
        validator.validate_cache_path(wrong_arch.parents[1], expected_version="0.6.17", expected_arch="121a")

    mixed = _cache_file(tmp_path / "mixed", "0.6.17", "121a", "0.6.17")
    second = mixed.parents[1] / ("b" * 64) / "autotune_configs.json"
    second.parent.mkdir()
    second.write_text(json.dumps({"_metadata": {"flashinfer_version": "0.6.16"}}))
    with pytest.raises(ValueError, match="flashinfer_version mismatch"):
        validator.validate_cache_path(mixed.parents[1], expected_version="0.6.17", expected_arch="121a")

    linked = _cache_file(tmp_path / "linked-source", "0.6.17", "121a", "0.6.17")
    linked_cache = tmp_path / "linked" / "0.6.17" / "121a"
    linked_cache.parent.mkdir(parents=True)
    linked_cache.symlink_to(linked.parents[1], target_is_directory=True)
    with pytest.raises(ValueError, match="not a local directory"):
        validator.validate_cache_path(linked_cache, expected_version="0.6.17", expected_arch="121a")


def test_cache_lifecycle_is_honest_and_persistent() -> None:
    manifest = json.loads((ROOT / "runtime/ACCELERATION_MANIFEST.json").read_text())
    cache = next(entry for entry in manifest["accelerations"] if entry["id"] == "flashinfer-autotune-cache")
    assert cache["status"] == "gpu-regeneration-outstanding"
    assert cache["active_cache_baked"] is False
    assert cache["required_version"] == "0.6.17"
    assert cache["architecture"] == "121a"
    serve = (ROOT / "examples/serve.sh").read_text()
    assert "docker volume create" in serve
    assert "/root/.cache/vllm/flashinfer_autotune_cache" in serve
    capture = (ROOT / "examples/capture_flashinfer_cache.sh").read_text()
    assert "validate_flashinfer_cache.py" in capture
    assert "0.6.17" in capture and "121a" in capture


def test_developer_and_release_interfaces_are_explicit() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "python -m build --wheel --outdir dist ./banana-smasher" in readme
    assert "python -m build --wheel --outdir dist ./banana-smasher-plugin" in readme
    assert "non-GPU static" in readme
    assert "Linux ARM64" in readme
    build = (ROOT / "examples/build_image.sh").read_text()
    assert "docker buildx build" in build
    assert "--platform linux/arm64" in build
    assert "--no-cache" in build
    smoke = (ROOT / "examples/smoke_api.py").read_text()
    assert smoke.index("/health") < smoke.index("/models") < smoke.index("/chat/completions")
    assert "expected served model" in smoke


def test_machine_manifest_contains_the_full_serve_profile() -> None:
    defaults = json.loads((ROOT / "docker/runtime_defaults.json").read_text())
    assert defaults["environment"] == {
        "BANANA_SMASHER_AOT_ROOT": "/opt/banana-smasher/aot",
        "CUDA_MODULE_LOADING": "LAZY",
        "MALLOC_MMAP_THRESHOLD_": "65536",
        "VLLM_MOE_W2_CUBIT_DIR": "/opt/banana-smasher/aot/cubins-sm120",
        "VLLM_MOE_W3_CUBIT_DIR": "/opt/banana-smasher/aot/cubins-e43",
        "VLLM_USE_DEEP_GEMM": "1",
        "VLLM_USE_DEEP_GEMM_E8M0": "1",
    }
    serve = defaults["serve"]
    expected = {
        "block_size": 256,
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        "enable_auto_tool_choice": True,
        "kv_cache_dtype": "fp8",
        "kv_cache_memory_bytes": 3221225472,
        "max_model_len": 8192,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 16,
        "no_scheduler_reserve_full_isl": True,
        "reasoning_parser": "deepseek_v4",
        "served_model_name": "banana-smasher-v5",
        "tokenizer_mode": "deepseek_v4",
        "tool_call_parser": "deepseek_v4",
    }
    for key, value in expected.items():
        assert serve[key] == value
    assert "BANANA_SMASHER_VLLM_COMPILE_FAST_PATHS" not in defaults["environment"]


def test_image_copies_and_checks_public_manifests() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    for source, target in (
        ("runtime/ASSET_MANIFEST.json", "/opt/banana-smasher/provenance/ASSET_MANIFEST.json"),
        ("runtime/ACCELERATION_MANIFEST.json", "/opt/banana-smasher/provenance/ACCELERATION_MANIFEST.json"),
        ("runtime/KERNEL_PRODUCERS.json", "/opt/banana-smasher/provenance/KERNEL_PRODUCERS.json"),
        ("provenance/SOURCE_INVENTORY.json", "/opt/banana-smasher/provenance/SOURCE_INVENTORY.json"),
    ):
        assert f"COPY {source} {target}" in dockerfile
    verifier = (ROOT / "docker/scripts/verify_public_image.py").read_text()
    assert "ACCELERATION_MANIFEST.json" in verifier
    assert "KERNEL_PRODUCERS.json" in verifier
    assert "SOURCE_INVENTORY.json" in verifier


def test_kernel_producer_manifest_covers_every_active_cubin_exactly() -> None:
    manifest = json.loads((ROOT / "runtime/KERNEL_PRODUCERS.json").read_text())
    assert manifest["schema"] == "banana-smasher-kernel-producers-v1"
    producers = manifest["producers"]
    sm120 = producers["sm120"]
    assert sm120["repository"] == "https://github.com/Sapid-Labs/vLLM-Moet.git"
    assert sm120["commit"] == "436d2a9100466198fc9cf23bd67a733d87fc9051"
    assert sm120["source_state"] == "source-available"
    assert sm120["exact_source_rebuild_state"] == "unsealed-assembler-identity-unresolved"
    assert sm120["assembler"]["unresolved_short_commit"] == "5912400"
    assert {item["name"]: item["sha256"] for item in sm120["assets"]} == EXPECTED_SM120

    e43 = producers["e43"]
    assert e43["repository"] == "external:spark-bench-reproducers"
    assert e43["commit"].startswith("f252699")
    assert e43["source_path"] == "deepseek-v4-flash-iq3-vq-warp-gb10/cubins/w3-source"
    assert e43["assembler_commit"] == "c139df8b34f1dcab607f8ccb685fdea948f3ae4d"
    assert e43["lut"] == {"hi": "0x4d463c21", "lo": "0xb6bfc6cd"}
    assert e43["exact_source_rebuild_state"] == "independently-rebuilt-byte-identical"
    assert e43["verification"]["exact_match_count"] == "6/6"
    assert e43["verification"]["receipt_manifest_sha256"] == (
        "a6effeb493e26e63c56bbec6266063ba8d1e822b39f9d390c8a1d8086c381bf9"
    )
    assert {item["name"]: item["sha256"] for item in e43["assets"]} == EXPECTED_E43

    all_assets = [item for producer in producers.values() for item in producer["assets"]]
    assert len(all_assets) == 32
    assert len({(item["name"], item["sha256"]) for item in all_assets}) == 32
    development = (ROOT / "KERNEL_DEVELOPMENT.md").read_text()
    assert "exact-source-rebuild seal" in development
    assert "5912400" in development
    assert "c139df8b34f1dcab607f8ccb685fdea948f3ae4d" in development


def test_register_activates_every_required_hook_and_dense_preflight() -> None:
    plugin = (ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/__init__.py").read_text()
    register_body = plugin.split("def register() -> None:", 1)[1]
    for hook in (
        "configure_flashinfer_sparse_mla_signature_compat()",
        "configure_stock_deepseek_v4_o_proj()",
        "configure_stock_deepseek_v4_attention_backend()",
        "configure_sparse_indexer_deep_gemm_backend()",
        "configure_sparse_indexer_topk_backend()",
        "install_deepseek_v4_dense_preflight()",
    ):
        assert hook in register_body
    assert "configure_stock_mhc_backend()" not in register_body
    assert "BANANA_SMASHER_MHC_BACKEND_OVERRIDE" not in plugin
    assert '"""Deprecated compatibility hook; stock public DeepGEMM owns MHC on SM12x."""' in plugin
    assert "BANANA_SMASHER_VLLM_COMPILE_FAST_PATHS" in (
        ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/quantization.py"
    ).read_text()
    quantization = (
        ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/quantization.py"
    ).read_text()
    assert "apply_runtime_repairs(model, contract)" in quantization


def test_d4_standalone_and_dynamic_kernels_decode_the_accepted_packed_wire() -> None:
    source = (
        ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/p1016_kernels.py"
    ).read_text()
    d4_scalar = source.split("def _d4_gemv(", 1)[1].split("@triton.jit", 1)[0]
    d4_dynamic = source.split("def _d4_gemv_dynamic(", 1)[1].split("@triton.jit", 1)[0]
    for body in (d4_scalar, d4_dynamic):
        assert "tl.pointer_type(tl.uint8)" in body
        assert "INDEX_BITS" in body
        assert "row_bytes" in body
        assert "tl.pointer_type(tl.int16)" not in body
