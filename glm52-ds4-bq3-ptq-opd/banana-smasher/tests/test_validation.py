from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.validation import ValidationError, validate_artifact


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_validation_fixture(root: Path) -> tuple[Path, Path]:
    bank = root / "holdout512_v1"
    bank.mkdir(parents=True)
    teacher = bank / "teacher_logits.npz"
    np.savez(
        teacher,
        sample_ids=np.asarray(["s0", "s1", "s2", "s3"]),
        classes=np.asarray(["code", "code", "reasoning", "reasoning"]),
        logits=np.asarray(
            [[3.0, 1.0, 0.0], [1.0, 3.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
            dtype=np.float32,
        ),
    )
    bank_manifest = {
        "schema": "bs-validation-bank-v1",
        "bank_id": "holdout512_v1",
        "sample_count": 4,
        "members": {
            "teacher_logits.npz": {
                "bytes": teacher.stat().st_size,
                "sha256": _sha(teacher),
                "role": "teacher_logits",
            }
        },
        "baselines": {
            "native-mxfp4": {"global_kld": 0.02},
            "qtip2": {"global_kld": 0.04},
        },
    }
    (bank / "bank.json").write_text(json.dumps(bank_manifest))

    artifact = root / "artifact"
    validation = artifact / "validation" / "holdout512_v1"
    validation.mkdir(parents=True)
    student = validation / "student_logits.npz"
    np.savez(
        student,
        sample_ids=np.asarray(["s0", "s1", "s2", "s3"]),
        logits=np.asarray(
            [[2.8, 1.1, 0.0], [1.2, 2.8, 0.1], [1.8, 0.1, 1.0], [0.1, 1.8, 1.1]],
            dtype=np.float32,
        ),
    )
    (artifact / "training_sample_ids.json").write_text(json.dumps(["train-0", "train-1"]))
    return artifact, bank


def test_validate_runs_student_comparison_and_seals_receipt(tmp_path: Path) -> None:
    artifact, bank = _write_validation_fixture(tmp_path)
    receipt_path = tmp_path / "VALIDATION_RECEIPT.json"

    receipt = validate_artifact(
        artifact,
        bank=bank,
        check_exposure=True,
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["bank"] == "holdout512_v1"
    assert receipt["sample_count"] == 4
    assert set(receipt["per_class_kld"]) == {"code", "reasoning"}
    assert set(receipt["comparison"]) == {"native-mxfp4", "qtip2"}
    assert receipt["exposure"]["overlap_count"] == 0
    assert receipt["provenance"]["teacher_logits_sha256"] == _sha(
        bank / "teacher_logits.npz"
    )
    assert receipt["receipt"]["sha256"] == _sha(receipt_path)


def test_validate_refuses_training_exposure(tmp_path: Path) -> None:
    artifact, bank = _write_validation_fixture(tmp_path)
    (artifact / "training_sample_ids.json").write_text(json.dumps(["s2"]))

    with pytest.raises(ValidationError, match="EXPOSURE_OVERLAP"):
        validate_artifact(artifact, bank=bank, check_exposure=True)


def test_validate_missing_teacher_companion_offers_one_time_bank_command(
    tmp_path: Path,
) -> None:
    artifact, bank = _write_validation_fixture(tmp_path)
    (bank / "teacher_logits.npz").unlink()

    with pytest.raises(ValidationError, match="--bank-teacher-logits"):
        validate_artifact(artifact, bank=bank, check_exposure=True)
