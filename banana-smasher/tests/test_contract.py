from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.contract import (
    KERNEL_MANIFEST_NAME,
    PackValidationError,
    export_pack,
    layout_sha256,
    verify_pack,
    verify_serve_compatibility,
)


def _write_qtip2_source(root: Path) -> Path:
    (root / "layers/layer_000/experts").mkdir(parents=True)
    (root / "layers/layer_000/qtip2").mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "hidden_size": 16}) + "\n"
    )
    np.save(
        root / "layers/layer_000/experts/tier_map.npy", np.zeros(256, dtype=np.uint8)
    )
    np.save(root / "layers/layer_000/qtip2/codes.npy", np.arange(32, dtype=np.uint8))
    np.save(root / "layers/layer_000/qtip2/scales.npy", np.arange(8, dtype=np.uint8))
    np.save(
        root / "layers/layer_000/qtip2/codebooks.npy",
        np.arange(16, dtype=np.float16).reshape(4, 4),
    )
    np.save(
        root / "layers/layer_000/qtip2/expert_ids.npy",
        np.arange(256, dtype=np.int16),
    )
    np.save(
        root / "layers/layer_000/qtip2/tensor_offsets.npy",
        np.array([0, 16, 32], dtype=np.int64),
    )
    return root


def test_export_hardlinks_and_verify_refuses_payload_drift(tmp_path: Path) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"

    manifest = export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-test-0001",
        link_mode="hardlink",
    )
    receipt = verify_pack(output)

    source_plane = source / "layers/layer_000/qtip2/codes.npy"
    exported_plane = output / "planes/layers/layer_000/qtip2/codes.npy"
    assert os.stat(source_plane).st_ino == os.stat(exported_plane).st_ino
    assert manifest["schema"] == "bs-pack"
    assert manifest["schema_version"] == 1
    assert receipt["status"] == "PASS"
    assert receipt["tensor_count"] == 6
    quant = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quant["quant_method"] == "banana_smasher"
    assert quant["pack_manifest"] == "BANANA_PACK_MANIFEST.json"
    assert quant["pack_root"] == "."
    assert quant["kernel_cache_root"] == "kernel-cache"
    assert quant["architecture"] == "sm_120"

    layer_meta_path = output / "planes/layers/layer_000/meta.json"
    layer_meta = json.loads(layer_meta_path.read_text())
    assert layer_meta == {
        "schema": "bs-pack-layer-meta",
        "schema_version": 1,
        "layer": 0,
        "experts_per_layer": 256,
        "expert_partitions": [64, 64, 64, 64],
        "tier_map": "layers.0.experts.tier_map",
        "dispatch_admission": {
            "scalar": {"predicate": "valid_m<4", "valid_m": [1, 2, 3]},
            "vector_m4": {"predicate": "valid_m==4", "valid_m": [4]},
        },
        "families": ["qtip2"],
        "tensors": sorted(manifest["tensor_index"]),
    }
    assert any(
        row["path"] == "planes/layers/layer_000/meta.json"
        and row["role"] == "layer_meta"
        for row in manifest["files"]
    )
    complete = json.loads((output / "PACK_COMPLETE").read_text())
    assert complete["status"] == "COMPLETE"
    assert any(
        row["path"] == "PACK_COMPLETE" and row["role"] == "pack_complete"
        for row in manifest["files"]
    )

    exported_plane.write_bytes(exported_plane.read_bytes() + b"drift")
    with pytest.raises(PackValidationError, match="byte count mismatch"):
        verify_pack(output)


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason="fsync ordering probe requires Linux /proc/self/fd",
)
def test_export_fsyncs_metadata_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"
    fsynced: list[str] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        try:
            fsynced.append(Path(os.readlink(f"/proc/self/fd/{fd}")).name)
        except OSError:
            fsynced.append("<directory>")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-durable-metadata",
        link_mode="copy",
    )

    config_event = next(index for index, name in enumerate(fsynced) if name == "config.json")
    manifest_event = next(
        index for index, name in enumerate(fsynced) if name == "BANANA_PACK_MANIFEST.json"
    )
    assert config_event < manifest_event
    assert verify_pack(output)["status"] == "PASS"


def test_verify_refuses_missing_pack_complete(tmp_path: Path) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-test-missing-complete",
        link_mode="copy",
    )
    (output / "PACK_COMPLETE").unlink()
    with pytest.raises(PackValidationError, match="missing PACK_COMPLETE marker"):
        verify_pack(output)


def test_verify_refuses_layer_meta_semantic_drift(tmp_path: Path) -> None:
    import hashlib

    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-test-meta-drift",
        link_mode="copy",
    )

    meta_path = output / "planes/layers/layer_000/meta.json"
    meta = json.loads(meta_path.read_text())
    meta["families"] = []
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    manifest_path = output / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(row for row in manifest["files"] if row["path"].endswith("meta.json"))
    row["bytes"] = meta_path.stat().st_size
    row["sha256"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(PackValidationError, match="layer meta mismatch"):
        verify_pack(output)


def test_verify_refuses_quant_detection_key_drift(tmp_path: Path) -> None:
    import hashlib

    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-test-config-drift",
        link_mode="copy",
    )

    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["pack_root"] = "../escape"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    manifest_path = output / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(row for row in manifest["files"] if row["path"] == "config.json")
    row["bytes"] = config_path.stat().st_size
    row["sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(PackValidationError, match="quantization_config.pack_root mismatch"):
        verify_pack(output)


def test_verify_refuses_unmanifested_files_and_bad_tier_map(tmp_path: Path) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    output = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="bs-pack-test-0002",
        link_mode="copy",
    )

    extra = output / "unlisted.bin"
    extra.write_bytes(b"not in the manifest")
    with pytest.raises(PackValidationError, match="file-set mismatch"):
        verify_pack(output)
    extra.unlink()

    tier_map = output / "planes/layers/layer_000/experts/tier_map.npy"
    np.save(tier_map, np.zeros(255, dtype=np.uint8))
    manifest_path = output / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(row for row in manifest["files"] if row["path"].endswith("tier_map.npy"))
    row["bytes"] = tier_map.stat().st_size
    import hashlib

    row["sha256"] = hashlib.sha256(tier_map.read_bytes()).hexdigest()
    tensor = manifest["tensor_index"]["layers.0.experts.tier_map"]
    tensor["shape"] = [255]
    tensor["data_bytes"] = 255
    tensor["data_sha256"] = hashlib.sha256(np.zeros(255, dtype=np.uint8)).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(PackValidationError, match=r"must be uint8\[256\]"):
        verify_pack(output)


def test_serve_check_binds_pack_layout_architecture_and_kernel_files(
    tmp_path: Path,
) -> None:
    import hashlib

    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="bs-pack-test-0003",
        link_mode="copy",
    )
    cache = tmp_path / "kernel-cache"
    kernel = cache / "kernels/qtip2-sm120.bin"
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"compiled-kernel-fixture")
    adapter = cache / "runtime_adapter.py"
    adapter.write_text(
        "class RuntimeAdapter:\n"
        "    API_VERSION = 1\n"
        "    def build_layer(self, **kwargs): return kwargs\n"
        "    def forward(self, state, **kwargs): return state\n"
    )
    cache_manifest = {
        "schema": "bs-kernel-cache",
        "schema_version": 1,
        "quant_method": "banana_smasher",
        "pack_schema": "bs-pack",
        "pack_schema_version": 1,
        "tensor_layout_sha256": layout_sha256(),
        "families": ["qtip2"],
        "architectures": ["sm_120"],
        "runtime_adapter": {
            "path": "runtime_adapter.py",
            "class": "RuntimeAdapter",
            "api_version": 1,
        },
        "files": [
            {
                "path": "kernels/qtip2-sm120.bin",
                "bytes": kernel.stat().st_size,
                "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            },
            {
                "path": "runtime_adapter.py",
                "bytes": adapter.stat().st_size,
                "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
            },
        ],
    }
    (cache / KERNEL_MANIFEST_NAME).write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )

    receipt = verify_serve_compatibility(pack, cache, architecture="sm_120")
    assert receipt["status"] == "PASS"
    assert receipt["required_families"] == ["qtip2"]
    assert receipt["runtime_adapter"]["class"] == "RuntimeAdapter"

    cache_manifest["tensor_layout_sha256"] = "0" * 64
    (cache / KERNEL_MANIFEST_NAME).write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(PackValidationError, match="tensor layout"):
        verify_serve_compatibility(pack, cache, architecture="sm_120")
