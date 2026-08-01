from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.bank import BankError, build_bank, verify_bank
from banana_smasher.durability import sha256_file
from banana_smasher.real_axis import RealAxisError, RealAxisRunner
from real_axis_fixtures import real_axis_fixture


def _build(paths: dict[str, Path], *, runner=None) -> dict:
    return build_bank(
        model_root=paths["model"],
        corpus=paths["corpus"],
        windows_manifest=paths["windows"],
        output=paths["bank"],
        instrument_profile=paths["instrument"],
        runner=runner,
    )


def _reseal_manifest(bank: Path, manifest: dict) -> None:
    manifest_path = bank / "bank.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    marker_path = bank / "BANK_COMPLETE"
    marker = json.loads(marker_path.read_text())
    marker["bank_manifest_sha256"] = sha256_file(manifest_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True) + "\n")


def test_complete_bank_is_exact_and_automatically_idempotent(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    first = _build(paths)
    assert first["status"] == "COMPLETE"
    assert first["generated_members"] == 2
    assert first["reused_members"] == 0
    assert (paths["bank"] / "bank.json").is_file()
    assert (paths["bank"] / "BANK_COMPLETE").is_file()
    seal = verify_bank(paths["bank"])
    assert [row["window_id"] for row in seal["members"]] == [100, 107]
    before = [row["sha256"] for row in seal["members"]]

    # BANK_COMPLETE is the bank seal; an operation-receipt crash window must
    # regenerate the receipt without touching members or the marker.
    (paths["bank"] / "BANK_RECEIPT.json").unlink()
    second = _build(paths)
    assert second["status"] == "COMPLETE"
    assert second["generated_members"] == 0
    assert second["reused_members"] == 2
    assert (paths["bank"] / "BANK_RECEIPT.json").is_file()
    assert [row["sha256"] for row in verify_bank(paths["bank"])["members"]] == before


class _FailAfter:
    def __init__(self, runner: RealAxisRunner, calls: int) -> None:
        self.runner = runner
        self.remaining = calls

    def identity(self):
        return self.runner.identity()

    def walk(self, hidden):
        if self.remaining == 0:
            raise RealAxisError("SYNTHETIC_BANK_INTERRUPTION")
        self.remaining -= 1
        return self.runner.walk(hidden)


def test_incomplete_bank_resumes_only_valid_members(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    failing = _FailAfter(RealAxisRunner(paths["model"]), calls=1)
    with pytest.raises(RealAxisError, match="SYNTHETIC_BANK_INTERRUPTION"):
        _build(paths, runner=failing)
    assert not (paths["bank"] / "BANK_COMPLETE").exists()
    assert (paths["bank"] / "members/window_000000.npz").is_file()

    resumed = _build(paths)
    assert resumed["status"] == "COMPLETE"
    assert resumed["generated_members"] == 1
    assert resumed["reused_members"] == 1


def test_corrupt_partial_member_is_regenerated_not_skipped(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    failing = _FailAfter(RealAxisRunner(paths["model"]), calls=1)
    with pytest.raises(RealAxisError):
        _build(paths, runner=failing)
    member = paths["bank"] / "members/window_000000.npz"
    member.write_bytes(member.read_bytes()[:32])

    resumed = _build(paths)
    assert resumed["generated_members"] == 2
    assert resumed["reused_members"] == 0
    verify_bank(paths["bank"])


def test_orphan_member_and_missing_completion_marker_recover_safely(
    tmp_path: Path,
) -> None:
    paths = real_axis_fixture(tmp_path)
    member = paths["bank"] / "members/window_000000.npz"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"orphaned-before-sidecar")
    result = _build(paths)
    assert result["generated_members"] == 2
    assert result["reused_members"] == 0

    (paths["bank"] / "BANK_COMPLETE").unlink()
    recovered = _build(paths)
    assert recovered["generated_members"] == 0
    assert recovered["reused_members"] == 2
    verify_bank(paths["bank"])


def test_complete_bank_mutation_fails_closed(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _build(paths)
    member = paths["bank"] / "members/window_000000.npz"
    member.write_bytes(member.read_bytes() + b"drift")
    with pytest.raises(BankError, match="BYTES_MISMATCH"):
        verify_bank(paths["bank"])


def test_incomplete_bank_never_verifies(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    failing = _FailAfter(RealAxisRunner(paths["model"]), calls=0)
    with pytest.raises(RealAxisError):
        _build(paths, runner=failing)
    with pytest.raises(BankError, match="BANK_INCOMPLETE"):
        verify_bank(paths["bank"])


def test_existing_progress_rejects_population_drift(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    failing = _FailAfter(RealAxisRunner(paths["model"]), calls=0)
    with pytest.raises(RealAxisError):
        _build(paths, runner=failing)
    progress = paths["bank"] / "BANK_PROGRESS.json"
    before = progress.read_bytes()
    manifest = paths["windows"]
    manifest.write_text(manifest.read_text().replace("fixture-corpus", "other-corpus"))
    with pytest.raises(BankError, match="BUILD_SPEC_MISMATCH"):
        _build(paths)
    assert progress.read_bytes() == before


def test_complete_bank_spec_mismatch_preserves_existing_seal(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _build(paths)
    marker = paths["bank"] / "BANK_COMPLETE"
    manifest = paths["bank"] / "bank.json"
    before = (marker.read_bytes(), manifest.read_bytes())
    windows = paths["windows"]
    windows.write_text(windows.read_text().replace("fixture-corpus", "other-corpus"))
    with pytest.raises(BankError, match="BUILD_SPEC_MISMATCH"):
        _build(paths)
    assert (marker.read_bytes(), manifest.read_bytes()) == before
    verify_bank(paths["bank"])


def test_unexpected_member_can_never_publish_complete(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    failing = _FailAfter(RealAxisRunner(paths["model"]), calls=0)
    with pytest.raises(RealAxisError):
        _build(paths, runner=failing)
    extra = paths["bank"] / "members/unexpected.npz"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unexpected")
    with pytest.raises(BankError, match="FILE_SET_MISMATCH"):
        _build(paths)
    assert not (paths["bank"] / "BANK_COMPLETE").exists()


def test_resealed_manifest_cannot_drift_member_class_from_population(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _build(paths)
    manifest = json.loads((paths["bank"] / "bank.json").read_text())
    manifest["members"][0]["class"] = "drifted-class"
    sidecar = paths["bank"] / manifest["members"][0]["sidecar"]
    sidecar.write_text(json.dumps(manifest["members"][0], sort_keys=True) + "\n")
    _reseal_manifest(paths["bank"], manifest)

    with pytest.raises(BankError, match="BANK_MEMBER_CLASS_MISMATCH"):
        verify_bank(paths["bank"])


def test_resealed_manifest_cannot_forge_population_or_build_spec(tmp_path: Path) -> None:
    paths = real_axis_fixture(tmp_path)
    _build(paths)
    manifest = json.loads((paths["bank"] / "bank.json").read_text())
    manifest["population"]["ordered_window_ids_sha256"] = "0" * 64
    _reseal_manifest(paths["bank"], manifest)
    with pytest.raises(BankError, match="BANK_POPULATION_SHA256_MISMATCH"):
        verify_bank(paths["bank"])

    manifest = json.loads((paths["bank"] / "bank.json").read_text())
    manifest["population"]["ordered_window_ids_sha256"] = manifest["corpus"][
        "ordered_window_ids_sha256"
    ]
    manifest["instrument"]["support"] -= 1
    _reseal_manifest(paths["bank"], manifest)
    with pytest.raises(BankError, match="BANK_BUILD_SPEC_DIGEST_MISMATCH"):
        verify_bank(paths["bank"])
