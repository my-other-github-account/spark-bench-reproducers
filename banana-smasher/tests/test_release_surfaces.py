from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.cli import main
from banana_smasher.contract import export_pack
from test_contract import _write_qtip2_source


ROOT = Path(__file__).parents[1]
EXPECTED_RELEASE_COMMANDS = [
    'smash export --source-root /path/to/materialized-quant-source --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" --serving-model-root /path/to/base-model --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy',
    "smash verify /model",
    "vllm serve /model",
]


def _bash_commands(markdown: str) -> list[str]:
    commands: list[str] = []
    in_bash = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line == "```bash":
            assert not in_bash
            in_bash = True
        elif line == "```" and in_bash:
            in_bash = False
        elif in_bash and line and not line.startswith("#"):
            commands.append(line)
    assert not in_bash
    return commands


def test_release_readme_is_literal_three_command_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert _bash_commands(readme) == EXPECTED_RELEASE_COMMANDS
    assert "The first sealed model instance has no special framework name" in readme


def test_pack_format_documents_versioned_layout_and_auto_detection() -> None:
    pack_format = (ROOT / "PACK_FORMAT.md").read_text(encoding="utf-8")
    for required in (
        "# PACK FORMAT — bs-pack v1",
        "BANANA_PACK_MANIFEST.json",
        "planes/",
        "meta.json",
        "config.json",
        "vLLM auto-detection keys",
        "byte count and SHA-256",
        "wrong/unknown schema, version",
    ):
        assert required in pack_format


def _fixture_pack(tmp_path: Path, name: str) -> Path:
    source = _write_qtip2_source(tmp_path / f"source-{name}")
    pack = tmp_path / f"pack-{name}"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id=f"bs-pack-release-{name}",
        link_mode="copy",
    )
    return pack


@pytest.mark.parametrize("failure", ["missing", "hash", "schema", "bytes"])
def test_validate_pack_fails_closed_on_release_identity_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    pack = _fixture_pack(tmp_path, failure)
    if failure == "missing":
        (pack / "PACK_COMPLETE").unlink()
        expected_error = "missing PACK_COMPLETE marker"
    elif failure == "hash":
        config_path = pack / "config.json"
        original = config_path.read_bytes()
        drifted = original.replace(b'"hidden_size": 16', b'"hidden_size": 17')
        assert len(drifted) == len(original) and drifted != original
        config_path.write_bytes(drifted)
        expected_error = "sha256 mismatch"
    elif failure == "schema":
        manifest_path = pack / "BANANA_PACK_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema"] = "unknown-pack"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        expected_error = "manifest schema mismatch"
    else:
        config_path = pack / "config.json"
        config_path.write_bytes(config_path.read_bytes() + b"\n")
        expected_error = "byte count mismatch"

    assert main(["validate-pack", str(pack)]) == 2
    failure_receipt = json.loads(capsys.readouterr().err)
    assert failure_receipt["status"] == "FAIL"
    assert failure_receipt["command"] == "validate-pack"
    assert expected_error in failure_receipt["error"]
