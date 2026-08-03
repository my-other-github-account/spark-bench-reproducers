from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

from banana_smasher.cli import _parser, main


def _fake_profile(argv, *, emit_summary=False):
    args = dict(zip(argv[::2], argv[1::2], strict=True))
    root = Path(args["--root"])
    assert root.is_dir(), "workflow must create the tier root before solver dispatch"
    layer = int(args["--layer"])
    tiers = args["--tiers"].split(",")
    out = root / "profile" / f"L{layer:03d}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "schema": "banana-smasher-solver-cell-tier-v1",
            "layer": layer,
            "expert": 0,
            "projection": "13",
            "cell": f"L{layer:03d}.E000.P13",
            "tier": tiers[0],
            "variant": "base",
            "weighted_sse": float(layer + 1),
            "teacher_energy": 10.0,
            "relative_weighted_error": float(layer + 1) / 10.0,
        }
    ]
    scientific = out / "SCIENTIFIC_ROWS.json"
    scientific.write_text(json.dumps(rows))
    scientific_sha256 = hashlib.sha256(scientific.read_bytes()).hexdigest()
    profile_rows = out / "PROFILE_ROWS.jsonl"
    profile_rows.write_text("".join(json.dumps(row) + "\n" for row in rows))
    objective = {
        "selected_cells": 1,
        "assignment_sha256": f"{layer + 1:064x}",
        "sum_relative_weighted_error": float(layer + 1) / 10.0,
        "sum_weighted_sse": float(layer + 1),
    }
    objective_path = out / "OBJECTIVE.json"
    objective_path.write_text(
        json.dumps({"schema": "banana-smasher-objective-v1", **objective})
    )
    summary = {
        "schema": "banana-smasher-solver-profile-v1",
        "status": "PASS",
        "implementation": args["--implementation"],
        "layer": layer,
        "tiers": tiers,
        "windows": int(args["--windows"]),
        "outer_wall_s": 1.0,
        "bucket_seconds": {"codebook_distance_sweeps": 0.5},
        "objective": objective,
        "exact_vectors": 1,
        "scientific_rows": str(scientific),
        "scientific_rows_sha256": scientific_sha256,
        "profile_rows": str(profile_rows),
        "profile_rows_sha256": hashlib.sha256(profile_rows.read_bytes()).hexdigest(),
        "objective_path": str(objective_path),
        "objective_sha256": hashlib.sha256(objective_path.read_bytes()).hexdigest(),
    }
    (out / "PROFILE_SUMMARY.json").write_text(json.dumps(summary))
    return summary


def test_fresh_model_solve_parser_accepts_layers_tiers_and_detach() -> None:
    args = _parser().parse_args(
        [
            "solve",
            "--root",
            "/tmp/run",
            "--source-root",
            "/tmp/source",
            "--layers",
            "0,2",
            "--tiers",
            "d4_k2048,d4_k4096",
            "--prices-root",
            "/prices",
            "--detach",
        ]
    )
    assert args.layers == "0,2"
    assert args.tiers == "d4_k2048,d4_k4096"
    assert str(args.prices_root) == "/prices"
    assert args.detach is True
    assert args.output is None


def test_public_qtip_solve_parser_accepts_sealed_config() -> None:
    args = _parser().parse_args(
        [
            "solve",
            "--root",
            "/tmp/run",
            "--source-root",
            "/tmp/source",
            "--layer",
            "39",
            "--qtip-profile-config",
            "/tmp/qtip.json",
        ]
    )
    assert args.layer == 39
    assert str(args.qtip_profile_config) == "/tmp/qtip.json"
    assert args.qtip_units is None


def test_public_qtip_config_directory_dispatches_one_resident_batch(
    tmp_path: Path, monkeypatch
) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    captured: dict[str, object] = {}
    fake = types.ModuleType("banana_smasher.solver_qtip_profile")

    def should_not_dispatch_one(*_args, **_kwargs):
        raise AssertionError("a config directory must use the resident batch fast path")

    def fake_many(config_root_arg, root, layer, *, limit, profile_mode):
        captured.update(
            config_root=config_root_arg,
            root=root,
            layer=layer,
            limit=limit,
            profile_mode=profile_mode,
        )
        return {"status": "PASS", "units": limit}

    setattr(fake, "main", should_not_dispatch_one)
    setattr(fake, "main_many", fake_many)
    monkeypatch.setitem(sys.modules, "banana_smasher.solver_qtip_profile", fake)
    run_root = tmp_path / "run"
    source_root = tmp_path / "source"

    assert (
        main(
            [
                "solve",
                "--root",
                str(run_root),
                "--source-root",
                str(source_root),
                "--layer",
                "39",
                "--qtip-profile-config",
                str(config_root),
                "--qtip-units",
                "64",
            ]
        )
        == 0
    )
    assert captured == {
        "config_root": config_root,
        "root": run_root,
        "layer": 39,
        "limit": 64,
        "profile_mode": False,
    }


def test_qtip_fit_bank_binds_exact_hessian_layer_manifest(tmp_path: Path) -> None:
    from banana_smasher.solver_qtip_profile import _bind_hessian_layer_manifest

    capture_root = tmp_path / "captures"
    capture_root.mkdir()
    members = []
    for window in range(2):
        capture = capture_root / f"xmoe_L039_win{window:04d}.pt"
        done = capture.with_suffix(capture.suffix + ".DONE.json")
        capture.write_bytes(f"capture-{window}".encode())
        done.write_text(json.dumps({"md5": f"done-{window}"}))
        members.append(
            {
                "window": window,
                "capture": {"path": str(capture), "bytes": capture.stat().st_size},
                "capture_done": {"path": str(done), "bytes": done.stat().st_size},
            }
        )
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-hessian-layer-manifest-v1",
                "status": "PASS",
                "layer": 39,
                "windows": 2,
                "capture_root": str(capture_root),
                "members": members,
            }
        )
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    root, windows, binding = _bind_hessian_layer_manifest(
        {
            "hessian_layer_manifest": str(manifest_path),
            "hessian_layer_manifest_sha256": manifest_sha,
            "fit_capture_root": str(capture_root),
            "fit_windows": 2,
        },
        layer=39,
    )
    assert root == capture_root.resolve()
    assert windows == 2
    assert binding["sha256"] == manifest_sha

    with pytest.raises(ValueError, match="hash drift"):
        _bind_hessian_layer_manifest(
            {
                "hessian_layer_manifest": str(manifest_path),
                "hessian_layer_manifest_sha256": "0" * 64,
                "fit_capture_root": str(capture_root),
                "fit_windows": 2,
            },
            layer=39,
        )


def test_public_capture_parser_accepts_local_source_contract() -> None:
    args = _parser().parse_args(
        [
            "capture",
            "--run-root",
            "/run/front-half",
            "--model-root",
            "/models/ds4",
            "--meta-root",
            "/models/ds4",
            "--corpus",
            "/inputs/train.json",
            "--builder",
            "/inputs/builder.py",
            "--layers",
            "3,13",
            "--windows",
            "32",
            "--detach",
        ]
    )
    assert args.layers == "3,13"
    assert args.windows == 32
    assert args.detach is True


def test_capture_injection_adds_capture_only_without_source_mutation() -> None:
    from banana_smasher.capture_source import (
        FORWARD_ANCHOR,
        PARSER_ANCHOR,
        READOUT_ANCHOR,
        inject_builder,
    )

    patched = inject_builder(PARSER_ANCHOR + FORWARD_ANCHOR + READOUT_ANCHOR)
    assert "--capture-only" in patched
    assert "banana-smasher-public-capture-v1" in patched
    assert "if a.capture_only:" in patched


def test_solve_adopts_complete_pricing_rows_without_reprofiling(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("sealed pricing input should bypass profiling")

    monkeypatch.setattr(workflow, "solver_profile_main", should_not_run)
    source = tmp_path / "source"
    source.mkdir()
    prices = source / "prices" / "L000"
    prices.mkdir(parents=True)
    rows = []
    for tier in ("d4_k2048", "d4_k4096"):
        for expert in range(256):
            for projection in ("13", "2"):
                rows.append(
                    {
                        "schema": "solver-pricing-v2-cell-tier-v1",
                        "cell": f"L000.E{expert:03d}.P{projection}",
                        "layer": 0,
                        "expert": expert,
                        "projection": projection,
                        "tier": tier,
                        "variant": "base",
                        "n_windows": 32,
                        "relative_weighted_error": float(expert + 1) / 1000.0,
                    }
                )
    (prices / "prices.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (prices / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "solver-pricing-v2-layer-complete-v1",
                "status": "PASS",
                "layer": 0,
                "windows": 32,
                "rows": len(rows),
                "expected_rows": len(rows),
            }
        )
    )

    root = tmp_path / "run"
    assert (
        main(
            [
                "solve",
                "--root",
                str(root),
                "--source-root",
                str(source),
                "--prices-root",
                str(source / "prices"),
                "--layers",
                "0",
                "--tiers",
                "d4_k2048,d4_k4096",
                "--windows",
                "32",
            ]
        )
        == 0
    )
    for tier in ("d4_k2048", "d4_k4096"):
        summary = json.loads(
            (
                root
                / "solve"
                / tier
                / "profile"
                / "L000"
                / "PROFILE_SUMMARY.json"
            ).read_text()
        )
        assert summary["source_kind"] == "sealed-pricing-v2"
        assert summary["solver_rows"] == 512
        assert len(summary["scientific_rows_sha256"]) == 64


def test_partial_prices_root_profiles_layers_without_sealed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    calls = 0

    def counted_profile(argv, *, emit_summary=False):
        nonlocal calls
        calls += 1
        return _fake_profile(argv, emit_summary=emit_summary)

    monkeypatch.setattr(workflow, "solver_profile_main", counted_profile)
    source = tmp_path / "source"
    prices = source / "prices"
    prices.mkdir(parents=True)
    receipt = workflow.run_fresh_solve(
        run_root=tmp_path / "run",
        source_root=source,
        layers=[1],
        tiers=["d4_k2048"],
        windows=32,
        staging_root=None,
        prices_root=prices,
        reference_search=False,
        hessian_manifest=None,
    )
    assert receipt["status"] == "PASS"
    assert calls == 1


def test_fresh_solve_writes_per_tier_and_aggregate_manifests(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "solver_profile_main", _fake_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()

    assert (
        main(
            [
                "solve",
                "--root",
                str(root),
                "--source-root",
                str(source),
                "--layers",
                "0,2",
                "--tiers",
                "d4_k2048,d4_k4096",
                "--windows",
                "32",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["tiers"] == ["d4_k2048", "d4_k4096"]
    manifest = json.loads((root / "solve" / "MANIFEST.json").read_text())
    assert manifest["schema"] == "banana-smasher-vq-solve-manifest-v1"
    assert manifest["layers"] == [0, 2]
    assert [row["tier"] for row in manifest["tier_manifests"]] == [
        "d4_k2048",
        "d4_k4096",
    ]
    for tier in manifest["tier_manifests"]:
        tier_path = Path(tier["path"])
        assert tier_path == root / "solve" / tier["tier"] / "MANIFEST.json"
        tier_obj = json.loads(tier_path.read_text())
        assert tier_obj["status"] == "PASS"
        assert len(tier_obj["layers"]) == 2
        assert all(row["summary"]["sha256"] for row in tier_obj["layers"])
    chain = json.loads((root / "WORKFLOW_CHAIN.json").read_text())
    assert chain["solve_manifest"]["sha256"] == workflow.sha256_file(
        root / "solve" / "MANIFEST.json"
    )


def test_anchor_consumes_solve_manifest_and_binds_outputs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "solver_profile_main", _fake_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    workflow.run_fresh_solve(
        run_root=root,
        source_root=source,
        layers=[0, 2],
        tiers=["d4_k2048", "d4_k4096"],
        windows=32,
        staging_root=None,
        reference_search=False,
        hessian_manifest=None,
    )

    assert main(["anchor", "--run-root", str(root)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    manifest = json.loads((root / "anchors" / "MANIFEST.json").read_text())
    assert manifest["schema"] == "banana-smasher-anchor-manifest-v1"
    assert manifest["input_solve_manifest"]["sha256"] == workflow.sha256_file(
        root / "solve" / "MANIFEST.json"
    )
    assert [row["tier"] for row in manifest["anchors"]] == [
        "d4_k2048",
        "d4_k4096",
    ]
    for row in manifest["anchors"]:
        obj = json.loads(Path(row["path"]).read_text())
        assert obj["measurement_label"] == "MEASURED_32_WINDOW_WEIGHTED_ERROR_NOT_MODEL_KLD"
        assert obj["layers"] == [0, 2]
        tier = row["tier"]
        named_manifest = root / "anchors" / f"ANCHOR_{tier}_MANIFEST.json"
        assert named_manifest.is_file()
        named = json.loads(named_manifest.read_text())
        assert named["schema"] == "banana-smasher-fixed-anchor-manifest-v1"
        assert named["tier"] == tier
        assert named["fixed_tier"] is True
        assert named["warm_start"] is False
        assert named["input_tier_solve_manifest"]["sha256"]
        assert [layer["layer"] for layer in named["layer_receipts"]] == [0, 2]
        for layer in named["layer_receipts"]:
            receipt_path = Path(layer["path"])
            assert receipt_path.is_file()
            receipt = json.loads(receipt_path.read_text())
            assert receipt["schema"] == "banana-smasher-fixed-anchor-layer-receipt-v1"
            assert receipt["tier"] == tier
            assert receipt["layer"] == layer["layer"]
            assert receipt["assignment_sha256"]
    chain = json.loads((root / "WORKFLOW_CHAIN.json").read_text())
    assert chain["anchor_manifest"]["sha256"] == workflow.sha256_file(
        root / "anchors" / "MANIFEST.json"
    )


def test_anchor_refuses_missing_canonical_profile_members(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "solver_profile_main", _fake_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    workflow.run_fresh_solve(
        run_root=root,
        source_root=source,
        layers=[37],
        tiers=["d4_k4096"],
        windows=32,
        staging_root=None,
        reference_search=False,
        hessian_manifest=None,
    )
    (
        root
        / "solve"
        / "d4_k4096"
        / "profile"
        / "L037"
        / "PROFILE_ROWS.jsonl"
    ).unlink()

    with pytest.raises(ValueError, match="missing canonical profile member.*PROFILE_ROWS"):
        workflow.run_anchor(run_root=root)
    assert not (root / "anchors" / "MANIFEST.json").exists()


def test_anchor_writes_hash_complete_canonical_pass_for_single_tier(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "solver_profile_main", _fake_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    workflow.run_fresh_solve(
        run_root=root,
        source_root=source,
        layers=[37],
        tiers=["d4_k4096"],
        windows=32,
        staging_root=None,
        reference_search=False,
        hessian_manifest=None,
    )
    workflow.run_anchor(run_root=root)

    pass_path = root / "layers" / "L037_PASS.json"
    canonical_pass = json.loads(pass_path.read_text())
    assert canonical_pass["schema"] == "banana-smasher-fixed-anchor-pass-v1"
    assert canonical_pass["status"] == "PASS"
    assert canonical_pass["tier"] == "d4_k4096"
    assert canonical_pass["layer"] == 37
    assert set(canonical_pass["members"]) == {
        "PROFILE_SUMMARY",
        "PROFILE_ROWS",
        "SCIENTIFIC_ROWS",
        "OBJECTIVE",
    }
    for record in canonical_pass["members"].values():
        path = Path(record["path"])
        assert record == workflow.artifact(path)


def test_status_reports_live_process_with_matching_startticks(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "process_startticks", lambda pid: 123456)
    root = tmp_path / "run"
    launch_dir = root / "run"
    launch_dir.mkdir(parents=True)
    startticks = 123456
    (launch_dir / "solve.launch.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-launch-v1",
                "status": "RUNNING",
                "verb": "solve",
                "pid": os.getpid(),
                "startticks": startticks,
            }
        )
    )

    assert main(["status", "--run-root", str(root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "RUNNING"
    assert status["launches"][0]["identity_matches"] is True
    assert status["launches"][0]["live"] is True


def test_status_treats_dead_capture_launch_as_completed_when_manifest_passes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "process_startticks", lambda _pid: None)
    root = tmp_path / "run"
    launch_dir = root / "run"
    launch_dir.mkdir(parents=True)
    (launch_dir / "capture.launch.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-launch-v1",
                "status": "RUNNING",
                "verb": "capture",
                "pid": 999999,
                "startticks": 123456,
            }
        )
    )
    captures = root / "captures"
    captures.mkdir()
    (captures / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-public-capture-manifest-v1",
                "status": "PASS",
            }
        )
    )

    assert main(["status", "--run-root", str(root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "IN_PROGRESS"
    assert status["manifests"][0]["schema"] == (
        "banana-smasher-public-capture-manifest-v1"
    )


def test_anchor_rejects_tier_path_escape(tmp_path: Path) -> None:
    from banana_smasher import workflow

    root = tmp_path / "run"
    tier_path = root / "solve" / "d4_k2048" / "MANIFEST.json"
    tier_path.parent.mkdir(parents=True)
    tier_path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-vq-tier-solve-manifest-v1",
                "status": "PASS",
                "tier": "../../../escaped",
                "windows": 32,
                "layers": [],
            }
        )
    )
    aggregate = root / "solve" / "MANIFEST.json"
    aggregate.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-vq-solve-manifest-v1",
                "status": "PASS",
                "windows": 32,
                "tiers": ["d4_k2048"],
                "tier_manifests": [
                    {"tier": "d4_k2048", **workflow.artifact(tier_path)}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="tier"):
        workflow.run_anchor(run_root=root)
    assert not (tmp_path / "escaped" / "ANCHOR.json").exists()


def test_anchor_rejects_tier_manifest_outside_solve_root(tmp_path: Path) -> None:
    from banana_smasher import workflow

    root = tmp_path / "run"
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-vq-tier-solve-manifest-v1",
                "status": "PASS",
                "tier": "d4_k2048",
                "windows": 32,
                "layers": [],
            }
        )
    )
    aggregate = root / "solve" / "MANIFEST.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-vq-solve-manifest-v1",
                "status": "PASS",
                "windows": 32,
                "tiers": ["d4_k2048"],
                "tier_manifests": [
                    {"tier": "d4_k2048", **workflow.artifact(external)}
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="outside solve root"):
        workflow.run_anchor(run_root=root)


def test_anchor_label_uses_manifest_window_count(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    monkeypatch.setattr(workflow, "solver_profile_main", _fake_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    workflow.run_fresh_solve(
        run_root=root,
        source_root=source,
        layers=[0],
        tiers=["d4_k2048"],
        windows=64,
        staging_root=None,
        prices_root=None,
        reference_search=False,
        hessian_manifest=None,
    )
    workflow.run_anchor(run_root=root)
    anchor = json.loads(
        (root / "anchors" / "d4_k2048" / "ANCHOR.json").read_text()
    )
    assert anchor["measurement_label"] == (
        "MEASURED_64_WINDOW_WEIGHTED_ERROR_NOT_MODEL_KLD"
    )


def test_status_fails_dead_launch_and_never_passes_partial_workflow(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    root = tmp_path / "dead"
    launch = root / "run" / "solve.launch.json"
    launch.parent.mkdir(parents=True)
    launch.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-launch-v1",
                "status": "RUNNING",
                "verb": "solve",
                "pid": 999999,
                "startticks": 123,
            }
        )
    )
    monkeypatch.setattr(workflow, "process_startticks", lambda _pid: None)
    assert workflow.workflow_status(run_root=root)["status"] == "FAIL"

    partial = tmp_path / "partial"
    manifest = partial / "solve" / "MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"schema": "x", "status": "PASS"}))
    assert workflow.workflow_status(run_root=partial)["status"] == "IN_PROGRESS"


def test_resume_rejects_tampered_scientific_rows(
    tmp_path: Path, monkeypatch
) -> None:
    from banana_smasher import workflow

    calls = 0

    def counted_profile(argv, *, emit_summary=False):
        nonlocal calls
        calls += 1
        return _fake_profile(argv, emit_summary=emit_summary)

    monkeypatch.setattr(workflow, "solver_profile_main", counted_profile)
    root = tmp_path / "run"
    source = tmp_path / "source"
    source.mkdir()
    kwargs = dict(
        run_root=root,
        source_root=source,
        layers=[0],
        tiers=["d4_k2048"],
        windows=32,
        staging_root=None,
        prices_root=None,
        reference_search=False,
        hessian_manifest=None,
    )
    workflow.run_fresh_solve(**kwargs)
    scientific = root / "solve" / "d4_k2048" / "profile" / "L000" / "SCIENTIFIC_ROWS.json"
    scientific.write_text(scientific.read_text() + "tampered")
    workflow.run_fresh_solve(**kwargs)
    assert calls == 2


def test_chain_invalidates_anchor_when_solve_manifest_changes(tmp_path: Path) -> None:
    from banana_smasher import workflow

    root = tmp_path / "run"
    hessian = root / "hessians" / "MANIFEST.json"
    solve_a = root / "solve" / "MANIFEST.json"
    solve_b = root / "solve" / "MANIFEST.v2.json"
    anchor = root / "anchors" / "MANIFEST.json"
    for path, marker in (
        (hessian, "hessian"),
        (solve_a, "solve-a"),
        (solve_b, "solve-b"),
        (anchor, "anchor-a"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker)

    chain_path = workflow._write_chain(
        root,
        hessian_manifest=workflow.artifact(hessian),
        solve_manifest=workflow.artifact(solve_a),
        anchor_manifest=workflow.artifact(anchor),
    )
    assert json.loads(chain_path.read_text())["status"] == "PASS"

    workflow._write_chain(root, solve_manifest=workflow.artifact(solve_b))
    chain = json.loads(chain_path.read_text())
    assert chain["status"] == "IN_PROGRESS"
    assert "anchor_manifest" not in chain
