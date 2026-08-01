from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.bank import build_bank
from banana_smasher.durability import canonical_sha256, sha256_file, tree_identity
from banana_smasher.evaluate import EvaluationError, evaluate_paired, verify_evaluation
from banana_smasher.metrics import aggregate_windows, paired_summary
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


def _reseal_evaluation(root: Path, receipt: dict) -> None:
    receipt["artifacts"]["tree_sha256"] = tree_identity(
        root,
        excluded_names=(
            ".banana-smasher.lock",
            "EVALUATION_COMPLETE",
            "EVALUATION_PROGRESS.json",
            "evaluation.json",
        ),
    )["sha256"]
    receipt_path = root / "evaluation.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    marker_path = root / "EVALUATION_COMPLETE"
    marker = json.loads(marker_path.read_text())
    marker["evaluation_id"] = receipt["evaluation_id"]
    marker["evaluation_spec_sha256"] = receipt["evaluation_spec_sha256"]
    marker["evaluation_sha256"] = sha256_file(receipt_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")


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
    assert "paired_ci95" in result["paired"]
    assert result["performance"]["fallback_used"] is False
    assert result["performance"]["window_batch_size"] == 2

    seal = verify_evaluation(paths["evaluation"])
    receipt = seal["receipt"]
    performance = receipt["performance"]
    assert set(performance) == {
        "tokens_per_second",
        "wall_seconds",
        "peak_vram_bytes",
        "quality_result",
        "kernel",
        "fallback_used",
        "fallback_status",
        "window_batch_size",
        "layer_forwards_per_arm",
        "head_forwards_per_arm",
    }
    assert performance["tokens_per_second"] > 0.0
    assert performance["wall_seconds"] > 0.0
    assert performance["peak_vram_bytes"] == 0
    assert performance["quality_result"]["status"] == "PASS"
    assert performance["kernel"] == "numpy-concatenated-all-window-gemm"
    assert performance["fallback_used"] is False
    assert performance["fallback_status"] == "none"
    assert performance["window_batch_size"] == 2
    assert performance["layer_forwards_per_arm"] == 3
    assert performance["head_forwards_per_arm"] == 1
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
        manifest_path = paths["evaluation"] / receipt["arms"][arm]["logits_manifest"][
            "path"
        ]
        arm_manifest = json.loads(manifest_path.read_text())
        assert [row["positions"] for row in arm_manifest["members"]] == [4, 4]
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 999),
        ("operation", "forged"),
        ("mode", "single_arm"),
        ("unexpected", "not-contractual"),
    ],
)
def test_resealed_evaluation_rejects_core_contract_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    receipt = json.loads((root / "evaluation.json").read_text())
    receipt[field] = value
    _reseal_evaluation(root, receipt)

    with pytest.raises(EvaluationError, match="EVALUATION_COMPLETION_INVALID"):
        verify_evaluation(root)


def test_resealed_evaluation_rejects_performance_receipt_drift(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    receipt = json.loads((root / "evaluation.json").read_text())
    receipt["performance"]["fallback_used"] = True
    _reseal_evaluation(root, receipt)

    with pytest.raises(EvaluationError, match="EVALUATION_PERFORMANCE_INVALID"):
        verify_evaluation(root)


def test_resealed_evaluation_rejects_aggregate_and_evaluation_id_drift(
    tmp_path: Path,
) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    receipt = json.loads((root / "evaluation.json").read_text())
    receipt["arms"]["candidate"]["kld"]["global"] = -999.0
    receipt["arms"]["candidate"]["top1_parity"]["rate"] = 2.0
    receipt["evaluation_id"] = "f" * 64
    _reseal_evaluation(root, receipt)

    with pytest.raises(EvaluationError, match="EVALUATION_(METRICS|ID)_MISMATCH"):
        verify_evaluation(root)


def test_pack_manifest_seal_binds_real_axis_descriptor(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    runtime_path = paths["candidate"] / "real_axis.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["layers"][0]["descriptor"]["source_shard"] = "unsealed-drift"
    runtime_path.write_text(json.dumps(runtime, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="PACK_MANIFEST_REAL_AXIS_IDENTITY_MISMATCH"):
        _evaluate(paths)


def test_resealed_kld_vector_is_recomputed_from_persisted_logprob(
    tmp_path: Path,
) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    receipt = json.loads((root / "evaluation.json").read_text())
    arm_path = root / "arms/candidate/manifest.json"
    arm = json.loads(arm_path.read_text())
    metric_rows = []
    for row in arm["members"]:
        artifact = root / row["path"]
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["kld"] = np.zeros_like(arrays["kld"])
        np.savez(
            artifact,
            candidate_logprob=arrays["candidate_logprob"],
            candidate_argmax=arrays["candidate_argmax"],
            kld=arrays["kld"],
            top1_equal=arrays["top1_equal"],
            teacher_logprob=arrays["teacher_logprob"],
            teacher_argmax=arrays["teacher_argmax"],
        )
        row["bytes"] = artifact.stat().st_size
        row["sha256"] = sha256_file(artifact)
        metric_rows.append(
            {
                "window_id": row["window_id"],
                "class": row["class"],
                "kld": arrays["kld"],
                "top1_equal": arrays["top1_equal"],
            }
        )
    arm_path.write_text(json.dumps(arm, sort_keys=True) + "\n")
    identity = receipt["arms"]["candidate"]["logits_manifest"]
    identity["sha256"] = sha256_file(arm_path)
    receipt["arms"]["candidate"]["kld_manifest"] = dict(identity)
    candidate_metrics = aggregate_windows(metric_rows)
    receipt["arms"]["candidate"]["kld"] = candidate_metrics["kld"]
    receipt["arms"]["candidate"]["top1_parity"] = candidate_metrics["top1_parity"]
    receipt["arms"]["candidate"]["per_window"] = candidate_metrics["per_window"]
    receipt["paired"] = paired_summary(
        candidate_metrics,
        {
            "kld": receipt["arms"]["reference"]["kld"],
            "top1_parity": receipt["arms"]["reference"]["top1_parity"],
            "per_window": receipt["arms"]["reference"]["per_window"],
        },
    )
    receipt["evaluation_id"] = canonical_sha256(
        {
            "evaluation_spec_sha256": receipt["evaluation_spec_sha256"],
            "candidate_manifest_sha256": receipt["arms"]["candidate"][
                "logits_manifest"
            ]["sha256"],
            "reference_manifest_sha256": receipt["arms"]["reference"][
                "logits_manifest"
            ]["sha256"],
        }
    )
    _reseal_evaluation(root, receipt)

    with pytest.raises(EvaluationError, match="ARM_MEMBER_KLD_MISMATCH"):
        verify_evaluation(root)


@pytest.mark.parametrize("drift", ["topology", "instrument", "arm_identity"])
def test_resealed_evaluation_rejects_seal_metadata_drift(
    tmp_path: Path, drift: str
) -> None:
    paths = real_axis_fixture(tmp_path)
    _bank(paths)
    _evaluate(paths)
    root = paths["evaluation"]
    receipt = json.loads((root / "evaluation.json").read_text())
    if drift == "topology":
        receipt["topology"]["layers"][0]["candidate"]["descriptor"][
            "source_shard"
        ] = "forged"
    elif drift == "instrument":
        receipt["instrument"]["support"] += 1
    else:
        receipt["arms"]["candidate"]["artifact"]["layer_count"] += 1
    _reseal_evaluation(root, receipt)

    with pytest.raises(
        EvaluationError,
        match="EVALUATION_(TOPOLOGY|INSTRUMENT|ARM_IDENTITY|SPEC)_MISMATCH",
    ):
        verify_evaluation(root)
