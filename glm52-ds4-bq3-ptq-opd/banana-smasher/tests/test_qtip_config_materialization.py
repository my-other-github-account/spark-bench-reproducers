from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from banana_smasher.cli import _parser, main
from banana_smasher.qtip_materialize import materialize_qtip_configs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _manifest(tmp_path: Path, *, tier: str = "qtip2-next", layer: int = 101) -> tuple[Path, Path]:
    model_root = tmp_path / "local-model"
    index = _write(model_root / "model.safetensors.index.json", {"weight_map": {}})
    basis = _sha(index)

    qtip_root = tmp_path / "local-qtip"
    qtip_manifest = _write(qtip_root / "MANIFEST.json", {"status": "PASS", "basis": basis})
    runner = _write(tmp_path / "runtime" / "public_runner.py", {"public": True})
    reference = _write(tmp_path / "runtime" / "reference.pt", {"sealed": True})
    tlut = _write(tmp_path / "runtime" / "tlut.pt", {"sealed": True})
    capture_root = tmp_path / "captures"
    capture_manifest = _write(capture_root / "MANIFEST.json", {"status": "PASS", "basis": basis})
    hessian = _write(tmp_path / "hessians" / f"L{layer:03d}" / "MANIFEST.json", {"status": "PASS", "basis": basis})

    source_rows = []
    for expert, projection in ((7, "fused13"), (7, "down")):
        source = _write(
            tmp_path / "source-configs" / f"L{layer:03d}" / f"E{expert:03d}_{projection}.json",
            {
                "schema": "banana-smasher-qtip-profile-config-v1",
                "tier": "qtip3",
                "geometry": {"L": 16, "K": 3, "V": 2},
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "layer_census": {"qtip3": 2, "d4": 0},
                "pack_counts": {"qtip3": 2, "d4": 0},
                "input_identity": {"model_index": {"sha256": basis}},
                "model_root": "/forbidden/source/model",
                "fit_capture_root": "/forbidden/source/captures",
                "hessian_layer_manifest": "/forbidden/source/hessian.json",
                "qtip_root": "/forbidden/source/qtip",
                "qtip_runner": "/forbidden/source/runner.py",
                "reference_unit": "/forbidden/source/reference.pt",
                "tlut_source": "/forbidden/source/tlut.pt",
                "rht_seed": expert,
            },
        )
        source_rows.append(_artifact(source))

    output_root = tmp_path / "materialized-configs"
    manifest = _write(
        output_root / "QTIP_RUN_MANIFEST.json",
        {
            "schema": "banana-smasher-qtip-run-manifest-v1",
            "status": "PASS",
            "basis_sha256": basis,
            "tiers": [
                {
                    "name": tier,
                    "geometry": {"L": 32, "K": 4, "V": 1},
                    "bindings": {
                        "model_root": {"path": str(model_root), "index": _artifact(index)},
                        "qtip_root": {"path": str(qtip_root), "manifest": _artifact(qtip_manifest)},
                        "qtip_runner": _artifact(runner),
                        "reference_unit": _artifact(reference),
                        "tlut_source": _artifact(tlut),
                    },
                    "layers": [
                        {
                            "layer": layer,
                            "source_configs": source_rows,
                            "fit_capture_root": {"path": str(capture_root), "manifest": _artifact(capture_manifest)},
                            "hessian_layer_manifest": _artifact(hessian),
                        }
                    ],
                }
            ],
        },
    )
    return manifest, output_root


def test_materializer_uses_open_manifest_tier_layer_and_local_bindings(tmp_path: Path) -> None:
    manifest, output_root = _manifest(tmp_path)

    receipt = materialize_qtip_configs(manifest, tier="qtip2-next", layers=[101], output_root=output_root)

    assert receipt["status"] == "PASS"
    assert receipt["tier"] == "qtip2-next"
    assert receipt["layers"] == [101]
    assert receipt["members"] == 2
    output = json.loads((output_root / "L101" / "E007_fused13.json").read_text())
    assert output["tier"] == "qtip2-next"
    assert output["geometry"] == {"L": 32, "K": 4, "V": 1}
    assert output["model_root"] == str(tmp_path / "local-model")
    assert output["layer_census"] == {"d4": 0, "qtip2-next": 2, "qtip3": 0}
    assert output["input_identity"]["model_index"]["sha256"] == receipt["basis_sha256"]


def test_materializer_is_hash_validated_and_idempotent(tmp_path: Path) -> None:
    manifest, output_root = _manifest(tmp_path)
    materialize_qtip_configs(manifest, tier="qtip2-next", layers=[101], output_root=output_root)
    output = output_root / "L101" / "E007_down.json"
    before = (output.read_bytes(), output.stat().st_mtime_ns)

    receipt = materialize_qtip_configs(manifest, tier="qtip2-next", layers=[101], output_root=output_root)

    assert receipt["existing_valid_members"] == 2
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


def test_materializer_fails_before_output_on_source_hash_drift(tmp_path: Path) -> None:
    manifest, output_root = _manifest(tmp_path)
    value = json.loads(manifest.read_text())
    source = Path(value["tiers"][0]["layers"][0]["source_configs"][0]["path"])
    source.write_text("drift\n")

    with pytest.raises(ValueError, match="hash/size drift"):
        materialize_qtip_configs(manifest, tier="qtip2-next", layers=[101], output_root=output_root)
    assert not (output_root / "L101").exists()


def test_solve_auto_materializes_manifest_declared_tier_before_public_route(tmp_path: Path, monkeypatch) -> None:
    manifest, output_root = _manifest(tmp_path)
    calls: list[tuple[Path, int, str]] = []
    fake = types.ModuleType("banana_smasher.solver_qtip_profile")

    def fake_many(config_root, root, layer, *, tier, all_cells, profile_mode):
        assert all_cells is True and profile_mode is False
        assert (config_root / "L101" / "E007_down.json").is_file()
        calls.append((config_root, layer, tier))
        return {"status": "PASS", "layer": layer, "tier": tier}

    setattr(fake, "main_many", fake_many)
    monkeypatch.setitem(sys.modules, "banana_smasher.solver_qtip_profile", fake)

    assert main([
        "solve", "--root", str(tmp_path / "run"), "--source-root", str(output_root),
        "--tier", "qtip2-next", "--all-cells", "--layers", "101",
    ]) == 0
    assert calls == [(output_root, 101, "qtip2-next")]
    assert manifest.is_file()


def test_solve_missing_configs_names_public_producer(tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "empty"
    source_root.mkdir()
    assert main([
        "solve", "--root", str(tmp_path / "run"), "--source-root", str(source_root),
        "--tier", "future-tier", "--all-cells", "--layers", "9",
    ]) == 2
    assert "smash qtip-configs" in capsys.readouterr().err


def test_qtip_configs_public_parser_has_no_tier_or_layer_menu(tmp_path: Path) -> None:
    args = _parser().parse_args([
        "qtip-configs", "--manifest", str(tmp_path / "manifest.json"),
        "--tier", "future-tier", "--layers", "101,205", "--output", str(tmp_path / "out"),
    ])
    assert args.tier == "future-tier"
    assert args.layers == "101,205"


def test_workflow_tier_population_is_manifest_open_not_a_package_menu() -> None:
    from banana_smasher.workflow import validate_open_tiers

    assert validate_open_tiers(["future-tier", "vendor.qtip4"]) == [
        "future-tier",
        "vendor.qtip4",
    ]
    with pytest.raises(ValueError, match="open tier population"):
        validate_open_tiers(["../escape"])
