from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from banana_smasher.bootstrap import bootstrap_container, container_recipe_path


def _fake_docker(path: Path, log: Path) -> Path:
    script = path / "docker"
    script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {str(log)!r}\n"
        "if [ \"$1 $2\" = \"image inspect\" ]; then\n"
        "  printf '%s\\n' 'sha256:fixture-image-id'\n"
        "fi\n"
    )
    script.chmod(0o755)
    return script


def _seal_recipe(root: Path, paths: tuple[Path, ...]) -> None:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    (root / "RECIPE_MANIFEST.json").write_text(
        json.dumps(
            {"schema": "genesis-golden-recipe-manifest-v1", "files": rows}
        )
    )


def test_bootstrap_builds_stock_vllm_image_and_seals_receipt(tmp_path: Path) -> None:
    recipe = tmp_path / "Dockerfile"
    recipe.write_text("FROM scratch\nENTRYPOINT [\\\"vllm\\\"]\n")
    log = tmp_path / "docker.log"
    docker = _fake_docker(tmp_path, log)
    receipt_path = tmp_path / "BOOTSTRAP.json"

    receipt = bootstrap_container(
        recipe=recipe,
        context=tmp_path,
        image="genesis-serve:test",
        docker_bin=str(docker),
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["image"] == "genesis-serve:test"
    assert receipt["image_id"] == "sha256:fixture-image-id"
    calls = log.read_text().splitlines()
    assert calls[0].startswith("build --file ")
    assert "--tag genesis-serve:test" in calls[0]
    assert calls[1] == "image inspect --format {{.Id}} genesis-serve:test"
    assert json.loads(receipt_path.read_text())["image_id"] == "sha256:fixture-image-id"


def test_bootstrap_uses_recipe_build_script_when_present(tmp_path: Path) -> None:
    recipe_root = tmp_path / "golden-container"
    recipe_root.mkdir()
    recipe = recipe_root / "Dockerfile"
    recipe.write_text('FROM scratch\nCMD ["vllm", "serve", "/model"]\n')
    script_log = tmp_path / "script.log"
    build_script = recipe_root / "build_golden.sh"
    build_script.write_text(
        "#!/bin/sh\n"
        f"printf '%s|%s\\n' \"$IMAGE\" \"$OUT\" > {str(script_log)!r}\n"
    )
    build_script.chmod(0o755)
    _seal_recipe(recipe_root, (recipe, build_script))
    docker = _fake_docker(tmp_path, tmp_path / "docker.log")
    receipt_path = tmp_path / "receipts/BOOTSTRAP.json"

    receipt = bootstrap_container(
        recipe=recipe,
        context=tmp_path,
        image="genesis-serve:candidate",
        docker_bin=str(docker),
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["command"] == [str(build_script.resolve())]
    assert script_log.read_text().strip() == (
        f"genesis-serve:candidate|{receipt_path.parent.resolve()}"
    )


def test_bootstrap_refuses_manifest_sealed_recipe_drift(tmp_path: Path) -> None:
    recipe_root = tmp_path / "golden-container"
    recipe_root.mkdir()
    recipe = recipe_root / "Dockerfile"
    recipe.write_text('FROM scratch\nCMD ["vllm", "serve", "/model"]\n')
    build_script = recipe_root / "build_golden.sh"
    build_script.write_text("#!/bin/sh\nexit 0\n")
    build_script.chmod(0o755)
    _seal_recipe(recipe_root, (recipe, build_script))
    build_script.write_text("#!/bin/sh\nexit 9\n")

    with pytest.raises(ValueError, match="recipe file sha256 mismatch"):
        bootstrap_container(
            recipe=recipe,
            context=tmp_path,
            image="genesis-serve:candidate",
            docker_bin=str(_fake_docker(tmp_path, tmp_path / "docker.log")),
            receipt_path=tmp_path / "BOOTSTRAP.json",
        )


def test_packaged_recipe_preserves_normal_vllm_entrypoint() -> None:
    recipe = container_recipe_path()
    assert recipe.is_file()
    assert recipe.parent.name == "golden-container"
    source = recipe.read_text()
    assert 'CMD ["vllm", "serve", "/model"' in source
    assert "ENTRYPOINT" not in source

    patch_source = (recipe.parent / "patch_quant_method_defaults.py").read_text()
    assert 'quant_cfg.get("moe_quant_algo")' in patch_source
    assert 'quant_cfg.get("moe_pack_root"' in patch_source

    manifest = json.loads((recipe.parent / "RECIPE_MANIFEST.json").read_text())
    assert manifest["schema"] == "genesis-golden-recipe-manifest-v2"
    assert manifest["truth_label"] == "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C"
