from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _verifier():
    path = ROOT / "docker/scripts/verify_public_image.py"
    spec = importlib.util.spec_from_file_location("verify_public_image", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_tree(tmp_path: Path) -> tuple[Path, Path]:
    aot = tmp_path / "aot"
    shutil.copytree(ROOT / "banana-smasher/kernels/cubins-sm120", aot / "cubins-sm120")
    shutil.copytree(ROOT / "banana-smasher/kernels/cubins-e43", aot / "cubins-e43")
    tlut = tmp_path / "qtip_tlut.npy"
    shutil.copy2(
        ROOT / "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy",
        tlut,
    )
    return aot, tlut


def test_exact_asset_admission_accepts_only_manifest_members(tmp_path: Path) -> None:
    verifier = _verifier()
    aot, tlut = _asset_tree(tmp_path)
    report = verifier.verify_asset_set(
        ROOT / "runtime/ASSET_MANIFEST.json",
        aot,
        tlut,
    )
    assert report["status"] == "PASS"
    assert report["counts"] == {"e43_cubins": 6, "sm120_cubins": 26}
    assert report["qtip_tlut"]["sha256"] == (
        "ae499158dab839d08c29de0b4dde4a6b303e781dc24e62e4b33ac50309bc5147"
    )

    (aot / "cubins-sm120/extra.cubin").write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="expected one exact asset set"):
        verifier.verify_asset_set(ROOT / "runtime/ASSET_MANIFEST.json", aot, tlut)

    (aot / "cubins-sm120/extra.cubin").unlink()
    target = aot / "cubins-e43/moe_w3_mm_k2048.cubin"
    target.write_bytes(target.read_bytes()[:-1] + b"x")
    with pytest.raises(RuntimeError, match="asset identity mismatch"):
        verifier.verify_asset_set(ROOT / "runtime/ASSET_MANIFEST.json", aot, tlut)

    target.unlink()
    target.symlink_to(aot / "cubins-e43/moe_w3_mm_k4096.cubin")
    with pytest.raises(RuntimeError, match="symlink"):
        verifier.verify_asset_set(ROOT / "runtime/ASSET_MANIFEST.json", aot, tlut)


def test_provenance_admission_binds_asset_and_producer_manifests(tmp_path: Path) -> None:
    verifier = _verifier()
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    for name, source in (
        ("ASSET_MANIFEST.json", ROOT / "runtime/ASSET_MANIFEST.json"),
        ("ACCELERATION_MANIFEST.json", ROOT / "runtime/ACCELERATION_MANIFEST.json"),
        ("KERNEL_PRODUCERS.json", ROOT / "runtime/KERNEL_PRODUCERS.json"),
        ("SOURCE_INVENTORY.json", ROOT / "provenance/SOURCE_INVENTORY.json"),
    ):
        shutil.copy2(source, provenance / name)

    report = verifier.verify_provenance_manifests(provenance)
    assert report["status"] == "PASS"
    assert report["producer_assets"] == 32

    producer_path = provenance / "KERNEL_PRODUCERS.json"
    producers = json.loads(producer_path.read_text())
    producers["producers"]["e43"]["assets"][0]["sha256"] = "0" * 64
    producer_path.write_text(json.dumps(producers))
    with pytest.raises(RuntimeError, match="producer asset mapping mismatch"):
        verifier.verify_provenance_manifests(provenance)
