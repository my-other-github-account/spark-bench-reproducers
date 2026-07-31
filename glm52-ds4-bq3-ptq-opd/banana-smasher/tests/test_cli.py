from __future__ import annotations

import json
from pathlib import Path

import banana_smasher.cli as cli_module
from banana_smasher.cli import _parser, main
from test_contract import _write_qtip2_source


def test_smash_help_exposes_exactly_five_verbs() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert list(action.choices) == [
        "export",
        "verify",
        "serve-check",
        "validate",
        "bootstrap",
    ]


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
