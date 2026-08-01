from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from banana_smasher import solver_qtip_profile as qtip


MODEL_INDEX_BYTES = b'{"weight_map":{}}'
BASIS = hashlib.sha256(MODEL_INDEX_BYTES).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "SHARDS.json").write_text(
        json.dumps({"intended_basis": {"index_sha256": BASIS}}) + "\n"
    )
    return run_root


def _config(path: Path, expert: int, projection: str) -> dict[str, object]:
    model_root = path.parent.parent / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "model.safetensors.index.json").write_bytes(MODEL_INDEX_BYTES)
    value: dict[str, object] = {
        "layer": 9,
        "expert": expert,
        "projection": projection,
        "geometry": {"L": 16, "K": 3, "V": 2},
        "model_root": str(model_root),
        "input_identity": {"model_index": {"sha256": BASIS}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


def _sealed_unit(
    root: Path, config_path: Path, expert: int, projection: str
) -> tuple[Path, Path]:
    """Write one canonical-format PASS unit: payload + hash-bound receipt."""
    out = root / "solve" / "L009" / f"E{expert:03d}_{projection}"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "QTIP_UNIT.pt"
    trellis = torch.tensor(
        [[expert, 0 if projection == "fused13" else 1]], dtype=torch.int64
    )
    torch.save(
        {
            "schema": "ds4-qtip-hyb-bounded36-unit-v1",
            "shape": [2, 2],
            "trellis": trellis,
            "SU": torch.ones(2, dtype=torch.float16),
            "SV": torch.ones(2, dtype=torch.float16),
            "Wscale": torch.tensor(1.0),
            "tlut": torch.ones((512, 2), dtype=torch.float16),
            "geometry": {
                "L": 16,
                "K": 3,
                "V": 2,
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
                "td_x": 16,
                "td_y": 16,
            },
        },
        artifact,
    )
    receipt_path = out / "QTIP_SOLVE_RECEIPT.json"
    receipt = {
        "schema": "banana-smasher-qtip-solve-v1",
        "status": "PASS",
        "layer": 9,
        "expert": expert,
        "projection": projection,
        "config_sha256": _sha256(config_path),
        "basis_gate": {
            "schema": "banana-smasher-qtip-basis-gate-v1",
            "status": "PASS",
            "index_sha256": BASIS,
            "intended_basis": BASIS,
        },
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "assignment_sha256": qtip._tensor_sha256(trellis),
        "total_wall_seconds": 1.0,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return artifact, receipt_path


def test_resident_batch_preserves_479_pass_units_and_resumes_at_unit_480(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """479 valid PASS units keep identical bytes AND mtimes; unit 480 onward runs."""
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    paths: list[Path] = []
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for unit in range(512):
        expert, projection_index = divmod(unit, 2)
        projection = ("fused13", "down")[projection_index]
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        paths.append(path)
        if unit < 479:
            artifact, receipt = _sealed_unit(run_root, path, expert, projection)
            sealed.append(
                (
                    artifact,
                    receipt,
                    artifact.read_bytes(),
                    receipt.read_bytes(),
                    artifact.stat().st_mtime_ns,
                    receipt.stat().st_mtime_ns,
                )
            )
    assert len(sealed) == 479

    calls: list[Path] = []

    def fake_main(path: Path, root: Path, layer: int, *, profile_mode: bool):
        assert root == run_root
        assert layer == 9
        assert profile_mode is False
        calls.append(path)
        value = json.loads(path.read_text())
        return {
            "status": "PASS",
            "layer": 9,
            "expert": value["expert"],
            "projection": value["projection"],
            "assignment_sha256": hashlib.sha256(path.name.encode()).hexdigest(),
            "total_wall_seconds": 2.0,
        }

    monkeypatch.setattr(qtip, "main", fake_main)
    batch = qtip.main_many(
        config_root,
        run_root,
        9,
        tier="qtip3",
        all_cells=True,
        profile_mode=False,
    )

    # Execution begins with unit 480 (index 479) and continues thereafter.
    assert calls == paths[479:]
    assert len(calls) == 33
    assert batch["status"] == "PASS"
    assert batch["units"] == 512
    assert batch["resumed_units"] == 479
    assert batch["computed_units"] == 33
    # Every ordered assignment survives, resumed and computed alike.
    assert len(batch["ordered_assignments"]) == 512
    # Sealed units are byte-for-byte and mtime-for-mtime untouched.
    for (
        artifact,
        receipt,
        artifact_bytes,
        receipt_bytes,
        artifact_mtime,
        receipt_mtime,
    ) in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime


def test_resident_batch_validates_every_existing_unit_before_first_new_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    missing = config_root / "E000_fused13.json"
    corrupt = config_root / "E000_down.json"
    _config(missing, 0, "fused13")
    _config(corrupt, 0, "down")
    artifact, _receipt = _sealed_unit(run_root, corrupt, 0, "down")
    artifact.write_bytes(b"corrupt")

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError(
            "preflight must reject corruption before computing a missing unit"
        )

    monkeypatch.setattr(qtip, "main", must_not_compute)
    with pytest.raises(RuntimeError, match="existing QTIP unit payload hash drift"):
        qtip.main_many(config_root, run_root, 9, profile_mode=False)


@pytest.mark.parametrize(
    "corruption",
    ["missing-receipt", "missing-payload", "payload-hash", "receipt-json"],
)
def test_existing_unit_fails_loudly_on_partial_or_corrupt_state(
    tmp_path: Path, corruption: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt = _sealed_unit(run_root, config_path, 0, "fused13")
    if corruption == "missing-receipt":
        receipt.unlink()
    elif corruption == "missing-payload":
        artifact.unlink()
    elif corruption == "payload-hash":
        artifact.write_bytes(b"drift")
    else:
        receipt.write_text("{not json")

    with pytest.raises(RuntimeError, match="existing QTIP unit"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        ("status", "identity drift"),
        ("expert", "identity drift"),
        ("config", "config hash drift"),
        ("artifact-path", "artifact path drift"),
        ("assignment", "assignment digest drift"),
        ("basis-gate", "basis drift"),
    ],
)
def test_existing_unit_fails_loudly_on_divergence(
    tmp_path: Path, mutate: str, error: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    receipt = json.loads(receipt_path.read_text())
    if mutate == "status":
        receipt["status"] = "FAIL"
    elif mutate == "expert":
        receipt["expert"] = 1
    elif mutate == "config":
        config = json.loads(config_path.read_text())
        config["rht_seed"] = 7
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    elif mutate == "artifact-path":
        receipt["artifact"] = str(artifact.parent / "elsewhere.pt")
    elif mutate == "assignment":
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        payload["trellis"] = torch.tensor([[99, 99]], dtype=torch.int64)
        torch.save(payload, artifact)
        receipt["artifact_sha256"] = _sha256(artifact)
    else:
        receipt["basis_gate"]["index_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match=error):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


def test_existing_unit_rehashes_live_model_index_against_run_basis(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    config = _config(config_path, 0, "fused13")
    _sealed_unit(run_root, config_path, 0, "fused13")
    model_index = Path(str(config["model_root"])) / "model.safetensors.index.json"
    model_index.write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="live model basis drift"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize("drift", ["payload-geometry", "payload-tensor"])
def test_existing_unit_rejects_incomplete_or_inconsistent_payload(
    tmp_path: Path, drift: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if drift == "payload-geometry":
        payload["geometry"]["K"] = 2
    else:
        del payload["SU"]
    torch.save(payload, artifact)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_sha256"] = _sha256(artifact)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="payload schema is invalid"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize("timing", [True, -1.0, float("nan"), float("inf"), "1.0"])
def test_existing_unit_rejects_invalid_timing(tmp_path: Path, timing: object) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    _artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    receipt = json.loads(receipt_path.read_text())
    receipt["total_wall_seconds"] = timing
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="timing is invalid"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


def test_missing_unit_returns_none_and_profile_mode_never_resumes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    # Nothing durable exists: fresh compute is required.
    assert (
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)
        is None
    )
    # A sealed solve unit never short-circuits profiling.
    _sealed_unit(run_root, config_path, 0, "fused13")
    assert (
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=True)
        is None
    )


def test_resident_batch_skips_39_valid_units_and_starts_at_first_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    paths: list[Path] = []
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for unit in range(40):
        expert, projection_index = divmod(unit, 2)
        projection = ("fused13", "down")[projection_index]
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        paths.append(path)
        if unit < 39:
            artifact, receipt = _sealed_unit(run_root, path, expert, projection)
            sealed.append(
                (
                    artifact,
                    receipt,
                    artifact.read_bytes(),
                    receipt.read_bytes(),
                    artifact.stat().st_mtime_ns,
                    receipt.stat().st_mtime_ns,
                )
            )

    calls: list[Path] = []

    def fake_main(path: Path, root: Path, layer: int, *, profile_mode: bool):
        calls.append(path)
        value = json.loads(path.read_text())
        return {
            "status": "PASS",
            "layer": layer,
            "expert": value["expert"],
            "projection": value["projection"],
            "assignment_sha256": hashlib.sha256(path.name.encode()).hexdigest(),
            "total_wall_seconds": 1.0,
        }

    monkeypatch.setattr(qtip, "main", fake_main)
    batch = qtip.main_many(config_root, run_root, 9, profile_mode=False)

    assert calls == [paths[39]]
    assert batch["resumed_units"] == 39
    assert batch["computed_units"] == 1
    for artifact, receipt, artifact_bytes, receipt_bytes, artifact_mtime, receipt_mtime in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime


def test_resident_batch_skips_a_complete_layer_without_rewriting_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for expert, projection in ((0, "fused13"), (0, "down")):
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        artifact, receipt = _sealed_unit(run_root, path, expert, projection)
        sealed.append(
            (
                artifact,
                receipt,
                artifact.read_bytes(),
                receipt.read_bytes(),
                artifact.stat().st_mtime_ns,
                receipt.stat().st_mtime_ns,
            )
        )

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("a complete hash-valid layer must skip all unit compute")

    monkeypatch.setattr(qtip, "main", must_not_compute)
    batch = qtip.main_many(config_root, run_root, 9, profile_mode=False)

    assert batch["resumed_units"] == 2
    assert batch["computed_units"] == 0
    for artifact, receipt, artifact_bytes, receipt_bytes, artifact_mtime, receipt_mtime in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime
