from __future__ import annotations

import json
from pathlib import Path

import banana_smasher.cli as cli_module
from banana_smasher.cli import _parser, main
from test_contract import _write_qtip2_source


def test_smash_help_exposes_update_as_sixth_command() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert list(action.choices) == [
        "export",
        "verify",
        "serve-check",
        "validate",
        "bootstrap",
        "update",
    ]


def test_smash_update_requires_explicit_runtime_inputs() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "update",
            "--runtime-root",
            "/runtime",
            "--model-root",
            "/model",
            "--aot",
            "/aot/_C.so",
            "--receipt",
            "/receipt.json",
        ]
    )
    assert args.command == "update"
    assert args.tokens == 1024
    assert args.layers == 43
    assert args.accumulation_segments == 8
    assert args.source_windows is None
    assert args.hard_abort_seconds == 250.0


def test_smash_update_parses_explicit_logical_source_windows() -> None:
    args = _parser().parse_args(
        [
            "update",
            "--receipt",
            "/receipt.json",
            "--source-windows",
            "27,38,39,43",
        ]
    )
    assert args.source_windows == (27, 38, 39, 43)


def test_smash_update_seals_accumulation_audit(tmp_path: Path, capsys) -> None:
    receipt = tmp_path / "audit.json"
    assert main(["update", "--audit-accumulation-only", "--receipt", str(receipt)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "PASS"
    assert output["segments"] == 8
    assert output["optimizer_steps"] == 1
    assert receipt.is_file()


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
