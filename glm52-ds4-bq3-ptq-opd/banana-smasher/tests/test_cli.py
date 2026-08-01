from __future__ import annotations

import json
from pathlib import Path

import banana_smasher.cli as cli_module
import banana_smasher.update as update_module
from banana_smasher.cli import _parser, main
from test_contract import _write_qtip2_source


def test_smash_help_exposes_solve_before_update() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert action.choices is not None
    commands = list(action.choices)
    assert commands[:5] == ["export", "verify", "serve-check", "validate", "bootstrap"]
    assert commands.index("solve") < commands.index("update")


def test_smash_update_defaults_to_accelerated_full_depth_eight_segments() -> None:
    args = _parser().parse_args(
        [
            "update",
            "--runtime-root",
            "/runtime",
            "--model-root",
            "/model",
            "--aot",
            "/aot/_C.so",
            "--output",
            "/updated.pt",
        ]
    )
    assert args.command == "update"
    assert args.tokens == 1024
    assert args.layers == 43
    assert args.segments == 8
    assert args.backend == "accelerated"
    assert args.output == Path("/updated.pt")
    assert args.source_windows is None
    assert args.restart is False
    assert args.verbose_receipts is False


def test_smash_update_parses_explicit_logical_source_windows() -> None:
    args = _parser().parse_args(
        [
            "update",
            "--output",
            "/updated.pt",
            "--source-windows",
            "27,38,39,43",
        ]
    )
    assert args.source_windows == (27, 38, 39, 43)


def test_smash_update_retains_accumulation_segments_compatibility_alias() -> None:
    args = _parser().parse_args(
        [
            "update",
            "--output",
            "/updated.pt",
            "--accumulation-segments",
            "5",
        ]
    )
    assert args.segments == 5


def test_smash_update_dispatches_accelerated_product_contract(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return {"status": "PASS_UPDATE"}

    monkeypatch.setattr(update_module, "run_minimal_update", fake_update)
    output = tmp_path / "updated.pt"
    assert main(
        [
            "update",
            "--runtime-root",
            "/runtime",
            "--model-root",
            "/model",
            "--aot",
            "/aot/_C.so",
            "--output",
            str(output),
        ]
    ) == 0
    assert captured["backend"] == "accelerated"
    assert captured["output"] == output
    assert captured["accumulation_segments"] == 8
    assert json.loads(capsys.readouterr().out)["status"] == "PASS_UPDATE"


def test_smash_update_fsyncs_full_traceback_failure_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    receipt = tmp_path / "update.json"

    def fail_update(**_kwargs):
        raise RuntimeError("synthetic full-depth failure")

    monkeypatch.setattr(update_module, "run_minimal_update", fail_update)
    assert (
        main(
            [
                "update",
                "--runtime-root",
                "/runtime",
                "--model-root",
                "/model",
                "--aot",
                "/aot/_C.so",
                "--output",
                str(tmp_path / "updated.pt"),
                "--receipt",
                str(receipt),
            ]
        )
        == 2
    )

    failure = receipt.with_name("update.failure.json")
    payload = json.loads(failure.read_text())
    assert payload["status"] == "FAIL_EXCEPTION"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic full-depth failure"
    assert "Traceback (most recent call last):" in payload["traceback"]
    assert "synthetic full-depth failure" in payload["traceback"]
    assert "synthetic full-depth failure" in capsys.readouterr().err


def test_smash_update_seals_failure_receipt_for_runtime_key_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    receipt = tmp_path / "update.json"

    def fail_update(**_kwargs):
        raise KeyError("BR_TRAIN")

    monkeypatch.setattr(update_module, "run_minimal_update", fail_update)
    assert main(
        [
            "update",
            "--runtime-root",
            "/runtime",
            "--model-root",
            "/model",
            "--aot",
            "/aot/_C.so",
            "--output",
            str(tmp_path / "updated.pt"),
            "--receipt",
            str(receipt),
        ]
    ) == 2
    payload = json.loads((tmp_path / "update.failure.json").read_text())
    assert payload["error_type"] == "KeyError"
    assert "BR_TRAIN" in payload["error"]
    assert "BR_TRAIN" in capsys.readouterr().err


def test_smash_update_keyboard_interrupt_seals_resumable_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output = tmp_path / "updated.pt"
    receipt = tmp_path / "update.json"

    def interrupt_update(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(update_module, "run_minimal_update", interrupt_update)
    assert main(
        [
            "update",
            "--runtime-root",
            "/runtime",
            "--model-root",
            "/model",
            "--aot",
            "/aot/_C.so",
            "--output",
            str(output),
            "--receipt",
            str(receipt),
        ]
    ) == 130
    failure = json.loads((tmp_path / "update.failure.json").read_text())
    assert failure["status"] == "INTERRUPTED_RESUMABLE"
    assert failure["resume_location"] == str(Path(f"{output}.checkpoint").resolve())
    assert "KeyboardInterrupt" in capsys.readouterr().err


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


def test_smash_bootstrap_defaults_to_candidate_tag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_bootstrap(**kwargs):
        captured.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(cli_module, "bootstrap_container", fake_bootstrap)
    assert (
        main(
            [
                "bootstrap",
                "--recipe",
                str(tmp_path / "Dockerfile"),
                "--context",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert captured["image"] == "banana_smasher-serve:banana-smasher-candidate"
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


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
