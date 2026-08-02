from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from banana_smasher.contract import (
    export_pack,
    refresh_serving_metadata,
    unpack_index_rows,
)


def _save(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    np.save(root / name, value, allow_pickle=False)
    return {
        "file": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _write_p1016_fixture(root: Path) -> dict[str, np.ndarray]:
    root.mkdir()
    experts = 256
    slots = list(range(experts))
    families = [2] * experts
    tiers = ["d4_k1024"] * experts
    expected: dict[str, np.ndarray] = {}
    payloads: dict[str, dict[str, object]] = {}
    for projection, suffix in (("fused13", "13"), ("down", "2")):
        values = np.arange(experts * 2 * 4, dtype=np.int16).reshape(experts, 2, 4) % 1024
        expected[projection] = values
        expert_ids = np.arange(experts, dtype=np.int16)
        tensors = {
            "codes": _save(root, f"layer_000.d4_k1024.{suffix}.codes.npy", values),
            "expert_ids": _save(
                root,
                f"layer_000.d4_k1024.{suffix}.expert_ids.npy",
                expert_ids,
            ),
        }
        payloads[projection] = {
            "d4_k1024": {
                "family": "d4",
                "d": 4,
                "k": 1024,
                "schema": "p972-vq-expert-group-v2",
                "tensors": tensors,
            }
        }
    meta = {
        "format": "p1016-true-c-native-planes-v1",
        "layer": 0,
        "E": experts,
        "family_codes": {"qtip2": 0, "qtip3": 1, "d4": 2, "native": 3},
        "family13": families,
        "family2": families,
        "tier13": tiers,
        "tier2": tiers,
        "slot13": slots,
        "slot2": slots,
        "payloads": payloads,
    }
    (root / "layer_000.meta.json").write_text(json.dumps(meta))
    return expected


def test_little_endian_row_codec_matches_v4_reference_bytes() -> None:
    values = np.array(
        [[0x001, 0x3FF, 0x155, 0x2AA], [0x3AB, 0x000, 0x3FF, 0x123]],
        dtype=np.uint16,
    )
    # The V4 producer packs each row independently, least-significant bit first.
    expected = np.array(
        [[0x01, 0xFC, 0x5F, 0x95, 0xAA], [0xAB, 0x03, 0xF0, 0xFF, 0x48]],
        dtype=np.uint8,
    )
    packed = np.packbits(
        ((values[..., None] >> np.arange(10, dtype=np.uint16)) & 1)
        .astype(np.uint8)
        .reshape(values.shape[0], -1),
        axis=-1,
        bitorder="little",
    )
    assert np.array_equal(packed, expected)
    assert np.array_equal(unpack_index_rows(packed, bits=10, values_per_row=4), values)


def test_export_physically_packs_d4_codes_and_records_lossless_decode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _write_p1016_fixture(source)
    output = tmp_path / "pack"

    manifest = export_pack(
        source_root=source,
        output=output,
        model_id="fixture",
        instance_id="fixture-1",
        runtime_floor_bytes=0,
        link_mode="copy",
    )

    selection = manifest["selected_payloads"]
    assert selection["producer_stage"] == "smash export:v4-row-packed-selected-wire-v1"
    for projection in ("fused13", "down"):
        spec = selection["layers"]["0"][projection]["payloads"]["d4_k1024"][
            "tensors"
        ]["codes"]
        assert spec["encoding"] == "little-endian-packed-index-rows-v1"
        assert spec["index_bits"] == 10
        assert spec["decoded_dtype"] == "int16"
        assert spec["decoded_shape"] == [256, 2, 4]
        packed = np.load(output / "planes" / spec["file"], allow_pickle=False)
        assert packed.dtype == np.uint8
        assert list(packed.shape) == [256, 2, 5]
        decoded = unpack_index_rows(packed, bits=10, values_per_row=4).reshape(
            expected[projection].shape
        )
        assert np.array_equal(decoded.astype(np.int16), expected[projection])
        assert spec["data_bytes"] < spec["decoded_data_bytes"]


def test_metadata_refresh_canonically_resets_runtime_floor_without_rewriting_payloads(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_p1016_fixture(source)
    output = tmp_path / "pack"
    manifest_before = export_pack(
        source_root=source,
        output=output,
        model_id="fixture",
        instance_id="fixture-refresh",
        runtime_floor_bytes=999,
        link_mode="copy",
    )
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        (serving / name).write_text("{}\n")
    payload_spec = manifest_before["selected_payloads"]["layers"]["0"]["fused13"][
        "payloads"
    ]["d4_k1024"]["tensors"]["codes"]
    payload = output / "planes" / payload_spec["file"]
    before = (payload.stat().st_ino, payload.read_bytes())

    receipt = refresh_serving_metadata(
        output,
        serving_model_root=serving,
        runtime_floor_bytes=123,
    )
    manifest = json.loads((output / "BANANA_PACK_MANIFEST.json").read_text())

    assert receipt["runtime_floor_bytes"] == 123
    assert manifest["selected_payloads"]["runtime_floor_bytes"] == 123
    assert (payload.stat().st_ino, payload.read_bytes()) == before
