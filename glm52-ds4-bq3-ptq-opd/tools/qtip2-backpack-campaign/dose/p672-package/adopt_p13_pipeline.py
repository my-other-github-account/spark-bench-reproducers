#!/usr/bin/env python3
"""Atomic P672 p13 pipeline overlay for a stopped P649-adopted mission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

SCHEMA = "p672-p13-pipeline-adoption-v1"
SOURCE_TASK = "PUBLIC_TASK"
BASE_SHA = "59cacb05b547c58329809b8f1d7c3a52011ecdd58b2eddb2302c0e3cb521125d"
BANANA_SMASHER_BASIC_SHA = "991370498c153988dda7df3fc5a23c40a4d58a48ba3d8bb4d596a3d9fa6a17cc"
FUSED_SHA = "5850caafaaba60502899da3ec713ed813a53505898cbeb410eef4e0a276e29d8"
P649_PHYSICAL_SHA = "7b075170e405ad54b0487f6649923cba4abcaf8592eeaadfde942409b2270a9f"
P672_PHYSICAL_SHA = "6d86f2e7ac658d365adfe20f04502e0697bc97a0fff8c972abb947d98c2c0661"
SAFE_BOUNDARY = (
    "Apply or rollback only while the target mission is stopped and already uses "
    "the exact P649 r4_c1_p2_m64n32w8 postimage. P672 changes only "
    "banana_smasher_physical_surface.py. Enable with P672_P13_PIPELINE=1 and "
    "P672_P13_GROUP=1; P649 p2 tile/grouping and P643 B32/eight-stream attention "
    "remain unchanged. B40, CONFIGS49, and multi-host PROOF-1 are forbidden."
)
RUNTIME_ENV = {
    "P672_P13_PIPELINE": "1",
    "P672_P13_GROUP": "1",
    "P649_EXPERT_RESIDENT_SCOPE": "4",
    "P649_DEQ_CHUNK": "1",
    "P649_NATIVE_CHUNK": "1",
    "P649_P2_BLOCK_M": "64",
    "P649_P2_BLOCK_N": "32",
    "P649_P2_NUM_WARPS": "8",
    "BANANA_SMASHER_REPAIR_MEM_FLOOR_BYTES": str(32 * 1024**3),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def root() -> Path:
    return Path(__file__).resolve().parent


def verify_payloads() -> dict[str, str]:
    expected = {
        "payload/banana_smasher_physical_surface.py": P672_PHYSICAL_SHA,
        "rollback/banana_smasher_physical_surface.py": P649_PHYSICAL_SHA,
    }
    actual: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = root() / relative
        if not path.is_file():
            raise RuntimeError(f"missing pinned bundle payload: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != wanted:
            raise RuntimeError(
                f"payload SHA mismatch {relative}: expected={wanted} actual={actual[relative]}"
            )
    return actual


def target_hashes(mission: Path) -> dict[str, str]:
    paths = {
        "base_binrepair_e2e.py": mission / "code/base_binrepair_e2e.py",
        "banana_smasher_basic_repair.py": mission / "code/banana_smasher_basic_repair.py",
        "banana_smasher_physical_surface.py": mission / "code/banana_smasher_physical_surface.py",
        "fused_expert_linear.py": mission / "code/fused_expert_linear.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise RuntimeError(f"missing target source: {path}")
    return {name: sha256(path) for name, path in paths.items()}


def classify(hashes: dict[str, str]) -> str:
    if hashes.get("base_binrepair_e2e.py") != BASE_SHA:
        return "unknown"
    if hashes.get("banana_smasher_basic_repair.py") != BANANA_SMASHER_BASIC_SHA:
        return "unknown"
    if hashes.get("fused_expert_linear.py") != FUSED_SHA:
        return "unknown"
    physical = hashes.get("banana_smasher_physical_surface.py")
    if physical == P649_PHYSICAL_SHA:
        return "p649_routed_experts"
    if physical == P672_PHYSICAL_SHA:
        return "p672_p13_pipeline"
    return "unknown"


def assert_stopped(mission: Path) -> None:
    prefix = str(mission.resolve()) + "/"
    offenders: list[dict[str, object]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for cmdline in proc.glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline.parent.name)
            if pid == os.getpid():
                continue
            argv = [
                part.decode(errors="replace")
                for part in cmdline.read_bytes().split(b"\0")
                if part
            ]
        except (OSError, ValueError):
            continue
        rooted = [argument for argument in argv[1:] if argument.startswith(prefix)]
        if rooted:
            offenders.append({"pid": pid, "argv": argv[:6], "rooted_args": rooted[:3]})
    if offenders:
        raise RuntimeError(
            "target mission is not stopped: " + json.dumps(offenders, sort_keys=True)
        )


def apply(mission: Path) -> dict[str, object]:
    bundle_shas = verify_payloads()
    assert_stopped(mission)
    before = target_hashes(mission)
    state = classify(before)
    if state == "unknown":
        raise RuntimeError(
            "refusing non-P649/mixed target; no bytes changed: "
            + json.dumps(before, sort_keys=True)
        )
    if state == "p672_p13_pipeline":
        return {
            "schema": SCHEMA,
            "action": "apply",
            "status": "ALREADY_ADOPTED",
            "mission": str(mission),
            "target_shas": before,
            "runtime_env": RUNTIME_ENV,
            "safe_boundary": SAFE_BOUNDARY,
        }
    backup_dir = mission / ".p672_p13_pipeline_rollback"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "banana_smasher_physical_surface.py"
    if backup.exists() and sha256(backup) != P649_PHYSICAL_SHA:
        raise RuntimeError(f"rollback backup collision: {backup}")
    if not backup.exists():
        atomic_copy(mission / "code/banana_smasher_physical_surface.py", backup)
    atomic_json(
        backup_dir / "MANIFEST.json",
        {
            "schema": SCHEMA,
            "mission": str(mission),
            "created_unix": time.time(),
            "original_state": "p649_routed_experts",
            "original_sha256": P649_PHYSICAL_SHA,
            "source_task": SOURCE_TASK,
        },
    )
    atomic_copy(
        root() / "payload/banana_smasher_physical_surface.py",
        mission / "code/banana_smasher_physical_surface.py",
    )
    after = target_hashes(mission)
    if classify(after) != "p672_p13_pipeline":
        raise RuntimeError(f"post-copy verification failed: {after}")
    receipt = {
        "schema": SCHEMA,
        "action": "apply",
        "status": "ADOPTED",
        "mission": str(mission),
        "completed_unix": time.time(),
        "before_shas": before,
        "after_shas": after,
        "bundle_shas": bundle_shas,
        "rollback_dir": str(backup_dir),
        "runtime_env": RUNTIME_ENV,
        "safe_boundary": SAFE_BOUNDARY,
    }
    atomic_json(mission / "P672_P13_PIPELINE_ADOPTION.json", receipt)
    return receipt


def rollback(mission: Path) -> dict[str, object]:
    verify_payloads()
    assert_stopped(mission)
    before = target_hashes(mission)
    state = classify(before)
    if state == "p649_routed_experts":
        return {
            "schema": SCHEMA,
            "action": "rollback",
            "status": "ALREADY_ROLLED_BACK",
            "mission": str(mission),
            "target_shas": before,
        }
    if state != "p672_p13_pipeline":
        raise RuntimeError(
            "refusing rollback from non-P672/mixed target; no bytes changed: "
            + json.dumps(before, sort_keys=True)
        )
    backup_dir = mission / ".p672_p13_pipeline_rollback"
    backup = backup_dir / "banana_smasher_physical_surface.py"
    manifest_path = backup_dir / "MANIFEST.json"
    if not manifest_path.is_file() or not backup.is_file():
        raise RuntimeError(f"missing rollback material: {backup_dir}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("original_state") != "p649_routed_experts"
        or manifest.get("original_sha256") != P649_PHYSICAL_SHA
        or sha256(backup) != P649_PHYSICAL_SHA
    ):
        raise RuntimeError("invalid rollback material")
    atomic_copy(backup, mission / "code/banana_smasher_physical_surface.py")
    after = target_hashes(mission)
    if classify(after) != "p649_routed_experts":
        raise RuntimeError(f"rollback verification failed: {after}")
    receipt = {
        "schema": SCHEMA,
        "action": "rollback",
        "status": "ROLLED_BACK",
        "mission": str(mission),
        "completed_unix": time.time(),
        "before_shas": before,
        "after_shas": after,
    }
    atomic_json(mission / "P672_P13_PIPELINE_ROLLBACK.json", receipt)
    return receipt


def status(mission: Path) -> dict[str, object]:
    bundle_shas = verify_payloads()
    hashes = target_hashes(mission)
    return {
        "schema": SCHEMA,
        "action": "status",
        "status": classify(hashes).upper(),
        "mission": str(mission),
        "target_shas": hashes,
        "bundle_shas": bundle_shas,
        "runtime_env": RUNTIME_ENV,
        "safe_boundary": SAFE_BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("apply", "rollback", "status"))
    parser.add_argument("mission", type=Path)
    arguments = parser.parse_args()
    mission = arguments.mission.expanduser().resolve()
    if arguments.action == "apply":
        result = apply(mission)
    elif arguments.action == "rollback":
        result = rollback(mission)
    else:
        result = status(mission)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"REFUSED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
