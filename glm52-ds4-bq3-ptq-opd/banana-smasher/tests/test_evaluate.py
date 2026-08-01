from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.bank import build_bank
from banana_smasher.durability import sha256_file, tree_identity
from banana_smasher.evaluate import EvaluationError, evaluate_paired, verify_evaluation
from real_axis_fixtures import real_axis_fixture


def _bank(paths: dict[str, Path]) -> None:
    build_bank(
        model_root=paths["model"],
        corpus=paths["corpus"],
        windows_manifest=paths["windows"],
        output=paths["bank"],
        instrument_profile=paths["instrument"],
    )


def _evaluate(paths: dict[str, Path], **kwargs):
    return evaluate_paired(
        model_root=paths["model"],
        candidate=paths["candidate"],
        reference=paths["reference"],
        bank=paths["bank"],
        output=paths["evaluation"],
        **kwargs,
    )


def test_paired_evaluation_persists_kld_top1_and_artifact_manifests(
    tmp_path: Path,
) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    result = _evaluate(paths)
    assert result["status"] == "COMPLETE"
    assert result["mode"] == "paired_real_axis"
    assert set(result["arms"]) == {"candidate", "reference"}
    assert result["arms"]["candidate"]["global_kld"] >= 0.0
    assert 0.0 <= result["arms"]["candidate"]["top1_rate"] <= 1.0
    assert "window_deltas" in result["paired"]

    seal = verify_evaluation(paths["evaluation"])
    receipt = seal["receipt"]
    assert receipt["population"] == {
        "ordered_window_ids_sha256": receipt["population"]["ordered_window_ids_sha256"],
        "windows": 2,
        "positions": 8,
        "classes": {"code": 1, "reasoning": 1},
    }
    for arm in ("candidate", "reference"):
        assert receipt["arms"][arm]["logits_manifest"]["members"] == 2
        assert receipt["arms"][arm]["kld_manifest"] == receipt["arms"][arm][
            "logits_manifest"
        ]
        assert set(receipt["arms"][arm]["kld"]["per_class"]) == {
            "code",
            "reasoning",
        }
        assert receipt["arms"][arm]["top1_parity"]["positions"] == 8
    assert (paths["evaluation"] / "EVALUATION_COMPLETE").is_file()


def test_resume_uses_only_greatest_common_pair_checkpoint(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=4)
    _bank(paths)

    def stop_after_layer_one(layer: int) -> None:
        if layer == 1:
            raise EvaluationError("SYNTHETIC_LAYER_DEATH")

    with pytest.raises(EvaluationError, match="SYNTHETIC_LAYER_DEATH"):
        _evaluate(paths, layer_hook=stop_after_layer_one)
    assert not (paths["evaluation"] / "EVALUATION_COMPLETE").exists()
    assert (
        paths["evaluation"] / "checkpoints/layer_001/PAIR_COMPLETE"
    ).is_file()

    result = _evaluate(paths)
    assert result["status"] == "COMPLETE"
    assert result["resume"]["started_from_layer"] == 2
    assert result["resume"]["checkpoint_sha256"]


def test_unpaired_suffix_is_not_a_resume_boundary(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=3)
    _bank(paths)

    def stop(layer: int) -> None:
        if layer == 1:
            raise EvaluationError("STOP")

    with pytest.raises(EvaluationError):
        _evaluate(paths, layer_hook=stop)
    (paths["evaluation"] / "checkpoints/layer_001/PAIR_COMPLETE").unlink()
    result = _evaluate(paths)
    assert result["resume"]["started_from_layer"] == 1


def test_explicit_resume_requires_a_valid_contiguous_pair(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=3)
    _bank(paths)
    with pytest.raises(ValueError):
        _evaluate(paths, resume_from_layer=2)


def test_final_receipt_without_completion_marker_does_not_skip(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=2)
    _bank(paths)
    _evaluate(paths)
    (paths["evaluation"] / "EVALUATION_COMPLETE").unlink()
    result = _evaluate(paths)
    assert result["status"] == "COMPLETE"
    assert result["resume"]["started_from_layer"] == 2


def test_evaluation_spec_mismatch_preserves_existing_completion(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    marker = paths["evaluation"] / "EVALUATION_COMPLETE"
    receipt = paths["evaluation"] / "evaluation.json"
    before = (marker.read_bytes(), receipt.read_bytes())
    with pytest.raises(EvaluationError, match="EVALUATION_SPEC_MISMATCH"):
        evaluate_paired(
            model_root=paths["model"],
            candidate=paths["reference"],
            reference=paths["candidate"],
            bank=paths["bank"],
            output=paths["evaluation"],
        )
    assert (marker.read_bytes(), receipt.read_bytes()) == before
    verify_evaluation(paths["evaluation"])


def test_layer_12_descriptor_is_resolved_independently(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=13)
    _bank(paths)
    _evaluate(paths)
    receipt = json.loads((paths["evaluation"] / "evaluation.json").read_text())
    layers = receipt["topology"]["layers"]
    assert layers[12]["candidate"]["descriptor"]["source_shard"] == (
        "divergent-shard-12"
    )
    assert layers[11]["candidate"]["sha256"] != layers[12]["candidate"]["sha256"]
    checkpoint = json.loads(
        (
            paths["evaluation"]
            / "checkpoints/layer_012/candidate/manifest.json"
        ).read_text()
    )
    assert checkpoint["layer_descriptor"]["descriptor"]["source_shard"] == (
        "divergent-shard-12"
    )


def test_arm_swap_flips_paired_delta_and_ratio(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    forward = _evaluate(paths)
    reverse = evaluate_paired(
        model_root=paths["model"],
        candidate=paths["reference"],
        reference=paths["candidate"],
        bank=paths["bank"],
        output=tmp_path / "evaluation-reversed",
    )
    assert reverse["paired"]["mean_window_delta"] == pytest.approx(
        -forward["paired"]["mean_window_delta"], abs=1e-12
    )
    assert reverse["paired"]["improvement_ratio"] == pytest.approx(
        1.0 / forward["paired"]["improvement_ratio"], rel=1e-10
    )


def test_identical_arms_have_zero_paired_delta_and_equal_parity(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    result = evaluate_paired(
        model_root=paths["model"],
        candidate=paths["candidate"],
        reference=paths["candidate"],
        bank=paths["bank"],
        output=paths["evaluation"],
    )
    assert result["paired"]["mean_window_delta"] == 0.0
    assert result["paired"]["improvement_ratio"] == 1.0
    assert result["arms"]["candidate"]["top1_rate"] == result["arms"]["reference"][
        "top1_rate"
    ]


def test_evaluate_rejects_incomplete_bank_before_loading_packs(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    paths["bank"].mkdir()
    with pytest.raises(ValueError, match="BANK_INCOMPLETE"):
        _evaluate(paths)


def test_unexpected_arm_artifact_prevents_completion(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    unexpected = paths["evaluation"] / "arms/candidate/unbound.npz"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_bytes(b"not-manifest-bound")

    with pytest.raises(EvaluationError, match="ARM_FILE_SET_MISMATCH"):
        _evaluate(paths)
    assert not (paths["evaluation"] / "EVALUATION_COMPLETE").exists()


def test_resealed_arm_manifest_cannot_drift_from_paired_population(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    arm_path = root / "arms/candidate/manifest.json"
    arm = json.loads(arm_path.read_text())
    arm["members"][0]["window_id"] = "drifted-window"
    arm_path.write_text(json.dumps(arm, sort_keys=True) + "\n")

    receipt_path = root / "evaluation.json"
    receipt = json.loads(receipt_path.read_text())
    arm_identity = receipt["arms"]["candidate"]["logits_manifest"]
    arm_identity["sha256"] = sha256_file(arm_path)
    receipt["arms"]["candidate"]["kld_manifest"] = dict(arm_identity)
    receipt["artifacts"]["tree_sha256"] = tree_identity(
        root,
        excluded_names=(
            ".banana-smasher.lock",
            "EVALUATION_COMPLETE",
            "EVALUATION_PROGRESS.json",
            "evaluation.json",
        ),
    )["sha256"]
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    marker_path = root / "EVALUATION_COMPLETE"
    marker = json.loads(marker_path.read_text())
    marker["evaluation_sha256"] = sha256_file(receipt_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")

    with pytest.raises(EvaluationError, match="ARM_POPULATION_MISMATCH"):
        verify_evaluation(root)
