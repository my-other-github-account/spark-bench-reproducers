#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    freeze = json.loads((ROOT / "provenance/SOURCE_VERSIONS.json").read_text())
    expected = {
        **{f"runtime/{name}": value for name, value in freeze["portable_runtime_sha256"].items()},
        **{f"artifacts/{name}": value for name, value in freeze["artifact_sha256"].items()},
    }
    missing = sorted(name for name in expected if not (ROOT / name).is_file())
    actual = {
        name: digest(ROOT / name)
        for name in expected
        if (ROOT / name).is_file()
    }
    failed = {
        name: {"expected": expected[name], "actual": actual.get(name)}
        for name in expected
        if actual.get(name) != expected[name]
    }
    manifest = json.loads((ROOT / "artifacts/MANIFEST.json").read_text())
    checks = {
        "all_files_present": not missing,
        "all_hashes_exact": not failed,
        "package_bytes": manifest["sources"]["genesis_package"]["bytes"] == 101_346_700_411,
        "package_files": manifest["sources"]["genesis_package"]["files"] == 1_645,
        "four_kernel_classes": len(manifest["kernel_classes"]) == 4,
        "four_tiers_64_each": set(manifest["tier_map_counts"].values()) == {64},
        "no_quality_claim": manifest["quality_claim"] is False,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing": missing,
        "failed_hashes": failed,
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
