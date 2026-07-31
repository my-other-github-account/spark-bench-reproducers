from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def container_recipe_path() -> Path:
    """Return the recipe shipped with the package."""
    source_checkout = Path(__file__).resolve().parents[2] / "golden-container"
    installed_package = Path(__file__).with_name("golden-container")
    root = source_checkout if source_checkout.is_dir() else installed_package
    return root / "Dockerfile"


def _verify_recipe_manifest(root: Path) -> dict[str, Any] | None:
    manifest_path = root / "RECIPE_MANIFEST.json"
    if not manifest_path.exists():
        return None
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"recipe manifest is not a regular file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read recipe manifest: {exc}") from exc
    if manifest.get("schema") not in {
        "genesis-golden-recipe-manifest-v1",
        "genesis-golden-recipe-manifest-v2",
    }:
        raise ValueError(f"unsupported recipe manifest schema: {manifest.get('schema')!r}")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recipe manifest files must be a non-empty list")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("malformed recipe manifest file row")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in seen:
            raise ValueError(f"unsafe/duplicate recipe manifest path: {relative}")
        seen.add(row["path"])
        physical = root / relative
        if not physical.is_file() or physical.is_symlink():
            raise ValueError(f"recipe file is missing/non-regular: {relative}")
        if physical.stat().st_size != row.get("bytes"):
            raise ValueError(f"recipe file byte count mismatch: {relative}")
        if _sha256_file(physical) != row.get("sha256"):
            raise ValueError(f"recipe file sha256 mismatch: {relative}")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "file_count": len(rows),
    }


def bootstrap_container(
    *,
    recipe: str | Path,
    context: str | Path,
    image: str,
    docker_bin: str = "docker",
    receipt_path: str | Path = "BOOTSTRAP_RECEIPT.json",
    pull: bool = False,
) -> dict[str, Any]:
    """Build or pull the stock-semantics vLLM container and seal a receipt."""
    started = time.time()
    recipe_path = Path(recipe).expanduser().resolve()
    context_path = Path(context).expanduser().resolve()
    if not recipe_path.is_file() or recipe_path.is_symlink():
        raise ValueError(f"container recipe is missing/non-regular: {recipe_path}")
    if not context_path.is_dir():
        raise ValueError(f"container build context is not a directory: {context_path}")
    if not image or any(character.isspace() for character in image):
        raise ValueError(f"invalid image reference: {image!r}")

    recipe_manifest = _verify_recipe_manifest(recipe_path.parent)
    output = Path(receipt_path).expanduser().resolve()
    build_script = recipe_path.with_name("build_golden.sh")
    environment = None
    if pull:
        command = [docker_bin, "pull", image]
        build_driver = "docker-pull"
    elif (
        recipe_manifest is not None
        and build_script.is_file()
        and not build_script.is_symlink()
    ):
        command = [str(build_script)]
        build_driver = "recipe-script"
        environment = os.environ.copy()
        environment.update(
            {
                "IMAGE": image,
                "OUT": str(output.parent),
                "DOCKER_BIN": docker_bin,
            }
        )
    else:
        command = [
            docker_bin,
            "build",
            "--file",
            str(recipe_path),
            "--tag",
            image,
            str(context_path),
        ]
        build_driver = "docker-build"
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=recipe_path.parent if build_driver == "recipe-script" else None,
        env=environment,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"container bootstrap failed rc={completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    inspect = subprocess.run(
        [docker_bin, "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if inspect.returncode != 0 or not inspect.stdout.strip().startswith("sha256:"):
        raise ValueError(
            f"cannot resolve built image id rc={inspect.returncode}: "
            f"{inspect.stderr[-4000:]}"
        )
    image_id = inspect.stdout.strip()
    receipt = {
        "schema": "bs-bootstrap-receipt-v1",
        "status": "PASS",
        "mode": "pull" if pull else "build",
        "build_driver": build_driver,
        "image": image,
        "image_id": image_id,
        "recipe": {
            "path": str(recipe_path),
            "sha256": _sha256_file(recipe_path),
        },
        "recipe_manifest": recipe_manifest,
        "context": str(context_path),
        "command": command,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "started_unix": started,
        "completed_unix": time.time(),
    }
    _atomic_json(output, receipt)
    result = dict(receipt)
    result["receipt"] = {"path": str(output), "sha256": _sha256_file(output)}
    return result
