from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.cli import _parser
from banana_smasher.contract import export_pack, verify_pack
from banana_smasher.repair import (
    CodebookRepair,
    RepairBundle,
    materialize_codebook_plane,
    validate_repair_state,
)
from test_contract import _write_qtip2_source


def _wire_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _fixture_bundle(old: np.ndarray, replacement: np.ndarray) -> RepairBundle:
    old_sha = _wire_sha(old)
    return RepairBundle(
        checkpoint_path=Path("/sealed/UPDATE_012.pt"),
        checkpoint_sha256="1" * 64,
        active_overlay_path=Path("/sealed/ACTIVE_OVERLAY.json"),
        active_overlay_sha256="2" * 64,
        assignment_path=Path("/sealed/ASSIGNMENT.json"),
        assignment_sha256="3" * 64,
        checkpoint_format="bs-basic-repair-v1",
        mechanism="physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains",
        update=12,
        codebooks={
            old_sha: CodebookRepair(
                checkpoint_key=f"L0/d4_k4__2_{old_sha}",
                source_wire_sha256=old_sha,
                array=np.ascontiguousarray(replacement, dtype=np.float16),
            )
        },
        dense_tensors={
            "norms/model.norm": np.arange(4, dtype=np.float32),
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.125, dtype=np.float32
            ),
        },
        norm_count=1,
        output_count=1,
    )


def test_repair_export_replaces_matching_codebook_and_binds_dense_state(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    codebook = source / "layers/layer_000/qtip2/codebooks.npy"
    old = np.load(codebook, allow_pickle=False)
    replacement = np.full(old.shape, 7.0, dtype=np.float16)
    bundle = _fixture_bundle(old, replacement)
    pack = tmp_path / "pack"

    manifest = export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="bs-pack-repair-0001",
        link_mode="hardlink",
        repair=bundle,
    )
    receipt = verify_pack(pack)

    repaired = np.load(
        pack / "planes/layers/layer_000/qtip2/codebooks.npy", allow_pickle=False
    )
    original = np.load(codebook, allow_pickle=False)
    assert np.array_equal(repaired, replacement)
    assert np.array_equal(original, old)
    assert manifest["repair"]["status"] == "MATERIALIZED"
    assert manifest["repair"]["codebook_checkpoint_keys"] == 1
    assert manifest["repair"]["codebook_target_files"] == 1
    assert receipt["repair"]["status"] == "PASS"
    assert receipt["repair"]["norms"] == 1
    assert receipt["repair"]["outputs"] == 1
    quant = json.loads((pack / "config.json").read_text())["quantization_config"]
    assert quant["repair_manifest"] == "repair/REPAIR_MANIFEST.json"
    assert quant["repair_state"] == "repair/repair_state.safetensors"


def test_repair_export_fails_closed_when_checkpoint_codebook_has_no_plane(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    old = np.arange(16, dtype=np.float16).reshape(4, 4)
    bundle = _fixture_bundle(old + 100, old + 200)

    with pytest.raises(ValueError, match="checkpoint codebooks were not materialized"):
        export_pack(
            source_root=source,
            output=tmp_path / "pack",
            model_id="fixture-model",
            instance_id="bs-pack-repair-missing",
            link_mode="copy",
            repair=bundle,
        )
    assert not (tmp_path / "pack").exists()


def test_materializer_replaces_each_index_in_multi_codebook_plane(
    tmp_path: Path,
) -> None:
    base = np.stack(
        [
            np.arange(8, dtype=np.float16).reshape(4, 2),
            np.arange(8, 16, dtype=np.float16).reshape(4, 2),
        ]
    )
    source = tmp_path / "layer_000.d4_k4.2.codebooks.npy"
    destination = tmp_path / "pack/planes/layer_000.d4_k4.2.codebooks.npy"
    np.save(source, base, allow_pickle=False)
    repairs = {}
    for index in range(2):
        source_sha = _wire_sha(base[index])
        repairs[source_sha] = CodebookRepair(
            checkpoint_key=f"L0/codebook_{index}_{source_sha}",
            source_wire_sha256=source_sha,
            array=np.full((4, 2), index + 20, dtype=np.float16),
        )

    rows = materialize_codebook_plane(source, destination, repairs)

    assert rows is not None
    assert [row["codebook_index"] for row in rows] == [0, 1]
    repaired = np.load(destination, allow_pickle=False)
    assert np.all(repaired[0] == 20)
    assert np.all(repaired[1] == 21)


def test_validate_repair_state_rejects_nonfinite_and_surface_drift() -> None:
    valid = {
        "codebooks": {
            "L0": {
                "cb_" + "a" * 64: np.ones((2, 2), dtype=np.float32),
            }
        },
        "norms": {"model.norm": np.ones(2, dtype=np.float32)},
        "outputs": {
            "model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.0, dtype=np.float32
            )
        },
    }
    result = validate_repair_state(valid, expected_counts=(1, 1, 1))
    assert len(result["codebooks"]) == 1

    valid["outputs"][
        "model.layers.0.self_attn.o_b_proj.output_log_gain"
    ] = np.asarray(np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        validate_repair_state(valid, expected_counts=(1, 1, 1))


def test_smash_export_parser_exposes_bound_repair_inputs() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "export",
            "--source-root",
            "/planes",
            "--output",
            "/pack",
            "--model-id",
            "DeepSeek-V4-Flash-BQ3",
            "--instance-id",
            "u012-v5",
            "--repair-checkpoint",
            "/sealed/UPDATE_012.pt",
            "--repair-checkpoint-sha256",
            "1" * 64,
            "--active-overlay",
            "/sealed/ACTIVE.json",
            "--active-overlay-sha256",
            "2" * 64,
            "--assignment",
            "/sealed/ASSIGNMENT.json",
            "--assignment-sha256",
            "3" * 64,
            "--repair-update",
            "12",
        ]
    )
    assert args.repair_checkpoint == Path("/sealed/UPDATE_012.pt")
    assert args.repair_update == 12
