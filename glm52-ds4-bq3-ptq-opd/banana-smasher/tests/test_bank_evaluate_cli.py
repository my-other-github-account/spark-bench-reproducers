from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.cli import _parser, main
from real_axis_fixtures import real_axis_fixture


def test_cli_exposes_public_bank_and_paired_evaluate_verbs() -> None:
    parser = _parser()
    action = next(action for action in parser._actions if getattr(action, "choices", None))
    assert list(action.choices)[-2:] == ["bank", "evaluate"]
    evaluate_help = action.choices["evaluate"].format_help()
    assert "--candidate" in evaluate_help
    assert "--reference" in evaluate_help
    for forbidden in (
        "host-claim",
        "gpu-exclusivity",
        "first-three",
        "mission-sha",
        "self-check",
    ):
        assert forbidden not in evaluate_help


def test_cli_bank_then_evaluate_real_axis_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = real_axis_fixture(tmp_path, layer_count=3)
    assert (
        main(
            [
                "bank",
                "--model-root",
                str(paths["model"]),
                "--corpus",
                str(paths["corpus"]),
                "--windows-manifest",
                str(paths["windows"]),
                "--instrument-profile",
                str(paths["instrument"]),
                "--output",
                str(paths["bank"]),
            ]
        )
        == 0
    )
    bank_result = json.loads(capsys.readouterr().out)
    assert bank_result["status"] == "COMPLETE"
    assert bank_result["generated_members"] == 2

    assert (
        main(
            [
                "evaluate",
                "--model-root",
                str(paths["model"]),
                "--candidate",
                str(paths["candidate"]),
                "--reference",
                str(paths["reference"]),
                "--bank",
                str(paths["bank"]),
                "--output",
                str(paths["evaluation"]),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE"
    assert result["mode"] == "paired_real_axis"
    assert "topology" not in result
    assert "evaluation" not in result
    assert set(result["arms"]) == {"candidate", "reference"}


def test_cli_evaluate_cannot_be_invoked_as_single_arm() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "evaluate",
                "--model-root",
                "/model",
                "--candidate",
                "/candidate",
                "--bank",
                "/bank",
                "--output",
                "/evaluation",
            ]
        )
