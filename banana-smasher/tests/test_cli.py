from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from banana_smasher.cli import _parser, main
from test_contract import _write_qtip2_source


def _write_serving_model(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "expert_dtype": "fp4",
                "rope_scaling": {"type": "yarn", "factor": 16},
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
        + "\n"
    )
    (root / "tokenizer.json").write_text('{"fixture":"tokenizer"}\n')
    (root / "tokenizer_config.json").write_text('{"fixture":"tokenizer-config"}\n')
    (root / "generation_config.json").write_text('{"fixture":"generation"}\n')
    return root


def _write_base_weights(root: Path) -> list[str]:
    shards = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    (root / shards[0]).write_bytes(b"dense-shard-one-fp8")
    (root / shards[1]).write_bytes(b"dense-shard-two-fp8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 38},
                "weight_map": {
                    "model.embed_tokens.weight": shards[0],
                    "lm_head.weight": shards[1],
                },
            }
        )
        + "\n"
    )
    return shards


def _write_symlinked_base_weights(root: Path, store: Path) -> list[str]:
    """Model root whose shards are symlinks into a separate store (NFS-mirror shape)."""
    store.mkdir(parents=True)
    shards = _write_base_weights(root)
    for shard in shards:
        real = store / shard
        (root / shard).rename(real)
        (root / shard).symlink_to(real)
    return shards


def test_smash_help_exposes_exactly_five_verbs() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert list(action.choices) == [
        "export",
        "verify",
        "serve-check",
        "validate",
    ]


def test_smash_validate_pack_compatibility_alias(tmp_path: Path, capsys) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "bs-pack-cli-validate-pack",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["validate-pack", str(pack)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["command"] == "validate-pack"
    assert receipt["status"] == "PASS"


def test_smash_export_verify_and_safetensors_repack(tmp_path: Path, capsys) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    pack = tmp_path / "pack"
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "bs-pack-cli-0001",
                "--link-mode",
                "copy",
                "--safetensors",
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["status"] == "PASS"
    assert exported["command"] == "export"
    assert exported["repack"]["byte_exact_tensors"] == 6

    assert main(["verify", str(pack)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "PASS"
    assert verified["tensor_count"] == 6


def test_smash_export_merges_full_serving_config_and_tokenizer_files(
    tmp_path: Path, capsys
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    pack = tmp_path / "pack"

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "serveable-export",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    config = json.loads((pack / "config.json").read_text())
    manifest = json.loads((pack / "BANANA_PACK_MANIFEST.json").read_text())

    assert receipt["status"] == "PASS"
    assert config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert config["hidden_size"] == 4096
    assert config["expert_dtype"] == "fp4"
    assert config["rope_scaling"] == {"type": "yarn", "factor": 16}
    assert config["quantization_config"]["quant_method"] == "banana_smasher"
    assert config["quantization_config"]["activation_scheme"] == "dynamic"
    assert config["quantization_config"]["fmt"] == "e4m3"
    assert config["quantization_config"]["scale_fmt"] == "ue8m0"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    assert {row["path"] for row in manifest["files"]} >= {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        assert (pack / name).read_bytes() == (serving_model / name).read_bytes()


def test_smash_export_canonicalizes_newline_lost_json_metadata(tmp_path: Path, capsys) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        path = serving_model / name
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
    pack = tmp_path / "pack"

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "canonical-json-newline",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    manifest = json.loads((pack / "BANANA_PACK_MANIFEST.json").read_text())
    rows = {row["path"]: row for row in manifest["files"]}
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        payload = (pack / name).read_bytes()
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")
        assert rows[name]["bytes"] == len(payload)
        assert rows[name]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert main(["verify", str(pack)]) == 0


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="atomic metadata refresh requires Linux renameat2(RENAME_EXCHANGE)",
)
def test_smash_export_refresh_metadata_preserves_tensor_files(
    tmp_path: Path, capsys
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    pack = tmp_path / "pack"
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-metadata",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    capsys.readouterr()
    plane = pack / "planes/layers/layer_000/qtip2/codes.npy"
    before = (os.stat(plane).st_ino, plane.read_bytes())
    old_quant = json.loads((pack / "config.json").read_text())["quantization_config"]

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-metadata",
                "--refresh-metadata",
            ]
        )
        == 0
    )
    refreshed = json.loads(capsys.readouterr().out)
    config = json.loads((pack / "config.json").read_text())

    assert refreshed["status"] == "PASS"
    assert refreshed["command"] == "export"
    assert refreshed["mode"] == "refresh-metadata"
    assert config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert {
        key: config["quantization_config"][key]
        for key in old_quant
    } == old_quant
    assert config["quantization_config"]["activation_scheme"] == "dynamic"
    assert config["quantization_config"]["fmt"] == "e4m3"
    assert config["quantization_config"]["scale_fmt"] == "ue8m0"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    assert (os.stat(plane).st_ino, plane.read_bytes()) == before
    assert main(["verify", str(pack)]) == 0


def test_smash_export_materializes_base_weight_shards(tmp_path: Path, capsys) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    shards = _write_base_weights(serving_model)
    pack = tmp_path / "pack"

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "serveable-export-weights",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    manifest = json.loads((pack / "BANANA_PACK_MANIFEST.json").read_text())

    assert receipt["status"] == "PASS"
    roles = {row["path"]: row["role"] for row in manifest["files"]}
    for shard in shards:
        assert roles[shard] == "base_weights_shard"
        assert (pack / shard).read_bytes() == (serving_model / shard).read_bytes()
        # hardlink default: same filesystem in tmp_path -> zero-copy links
        assert os.stat(pack / shard).st_ino == os.stat(serving_model / shard).st_ino
    assert roles["model.safetensors.index.json"] == "base_weights_index"
    assert (pack / "model.safetensors.index.json").read_bytes() == (
        serving_model / "model.safetensors.index.json"
    ).read_bytes()
    assert main(["verify", str(pack)]) == 0


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="atomic metadata refresh requires Linux renameat2(RENAME_EXCHANGE)",
)
def test_smash_refresh_metadata_adds_base_weights_without_tensor_rewrites(
    tmp_path: Path, capsys
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    shards = _write_base_weights(serving_model)
    pack = tmp_path / "pack"
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-weights",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    capsys.readouterr()
    plane = pack / "planes/layers/layer_000/qtip2/codes.npy"
    plane_before = (os.stat(plane).st_ino, plane.read_bytes())
    old_quant = json.loads((pack / "config.json").read_text())["quantization_config"]

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-weights",
                "--refresh-metadata",
            ]
        )
        == 0
    )
    refreshed = json.loads(capsys.readouterr().out)
    config = json.loads((pack / "config.json").read_text())
    manifest = json.loads((pack / "BANANA_PACK_MANIFEST.json").read_text())

    assert refreshed["status"] == "PASS"
    assert refreshed["mode"] == "refresh-metadata"
    assert refreshed["base_weights_shards"] == len(shards)
    assert refreshed["base_weights_index"] is True
    assert refreshed["tensor_payloads_rewritten"] is False
    assert config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert config["hidden_size"] == 4096
    assert config["rope_scaling"] == {"type": "yarn", "factor": 16}
    assert config["expert_dtype"] == "fp4"
    assert {
        key: config["quantization_config"][key]
        for key in old_quant
    } == old_quant
    assert config["quantization_config"]["activation_scheme"] == "dynamic"
    assert config["quantization_config"]["fmt"] == "e4m3"
    assert config["quantization_config"]["scale_fmt"] == "ue8m0"
    assert config["quantization_config"]["weight_block_size"] == [128, 128]
    # metadata-only: quantized planes must be byte- and inode-identical
    assert (os.stat(plane).st_ino, plane.read_bytes()) == plane_before
    roles = {row["path"]: row["role"] for row in manifest["files"]}
    for shard in shards:
        assert roles[shard] == "base_weights_shard"
        assert (pack / shard).read_bytes() == (serving_model / shard).read_bytes()
        assert os.stat(pack / shard).st_ino == os.stat(serving_model / shard).st_ino
    assert roles["model.safetensors.index.json"] == "base_weights_index"
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        assert (pack / name).read_bytes() == (serving_model / name).read_bytes()
    assert main(["verify", str(pack)]) == 0
    capsys.readouterr()

    # idempotent re-refresh replaces stale base-weight rows instead of duplicating
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-weights",
                "--refresh-metadata",
            ]
        )
        == 0
    )
    rerefreshed = json.loads(capsys.readouterr().out)
    assert rerefreshed["status"] == "PASS"
    manifest2 = json.loads((pack / "BANANA_PACK_MANIFEST.json").read_text())
    shard_rows = [
        row
        for row in manifest2["files"]
        if row["role"] in ("base_weights_shard", "base_weights_index")
    ]
    assert len(shard_rows) == len(shards) + 1
    assert (os.stat(plane).st_ino, plane.read_bytes()) == plane_before
    assert main(["verify", str(pack)]) == 0


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="atomic metadata refresh requires Linux renameat2(RENAME_EXCHANGE)",
)
def test_smash_refresh_metadata_resolves_symlinked_shards(
    tmp_path: Path, capsys
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    serving_model = _write_serving_model(tmp_path / "serving-model")
    shards = _write_symlinked_base_weights(serving_model, tmp_path / "shard-store")
    pack = tmp_path / "pack"
    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-symlink-weights",
                "--link-mode",
                "copy",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "export",
                "--source-root",
                str(source),
                "--serving-model-root",
                str(serving_model),
                "--output",
                str(pack),
                "--model-id",
                "fixture-model",
                "--instance-id",
                "refresh-symlink-weights",
                "--refresh-metadata",
            ]
        )
        == 0
    )
    refreshed = json.loads(capsys.readouterr().out)
    assert refreshed["status"] == "PASS"
    assert refreshed["base_weights_shards"] == len(shards)
    for shard in shards:
        packed = pack / shard
        # pack entry must be a REGULAR file (hardlink to the resolved target)
        assert packed.is_file() and not packed.is_symlink()
        real = tmp_path / "shard-store" / shard
        assert packed.read_bytes() == real.read_bytes()
        assert os.stat(packed).st_ino == os.stat(real).st_ino
    assert main(["verify", str(pack)]) == 0
