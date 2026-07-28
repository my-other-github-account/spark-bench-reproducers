#!/usr/bin/env python3
"""Rebuild the vendored-source index and complete package manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"

SOURCE_OVERRIDES = {
    "vendor/calibration/fit_wire_calibration.py": {
        "source_sha256": "98e0877586cf3f209bbd7f95a98bdaa126c5be7a6878301461f9b4297dad4edb",
        "transformation": "privacy-only task label substitution",
    },
    "vendor/calibration/pricing_v3_surface.py": {
        "source_sha256": "04b7d53935362b7d71622fba53e2d8170f51e70a74303e47081948d3203e0fc9",
        "transformation": "privacy-only task label substitution",
    },
    "vendor/calibration/verify_corrected_pricing.py": {
        "source_sha256": "7c14b813e1f1e7a16a13f5a20ad807eb580eaeb6888903b3ae37e787bd2b64cf",
        "transformation": "removed dead external-parent expression",
    },
    "vendor/eval/generate_http.py": {
        "source_sha256": "060d9d3d14f5980e4ea9a5440ae6e210875910754d0467d099b86eb17cf9f039",
        "transformation": "private claim location replaced by environment-selected package contract",
    },
    "vendor/eval/score_evalplus.sh": {
        "source_sha256": "30d7e65bdef0471e0086ff05138d23c0d51e1dccd02c45ad06477990aad6058f",
        "transformation": "private claim and container-home locations replaced",
    },
    "vendor/evalplus/evalplus/eval/__init__.py": {
        "source_sha256": "3c833c39b842e33f251c83db4347e0a95191909f23b390c12d73ab29a28a4daf",
        "transformation": "comment-only package-local reference wording",
    },
    "vendor/evalplus/tools/_experimental/generate_big_input.py": {
        "source_sha256": "2564bbd510b77095a7c80bce67ba6720db325ddd39ae74a1e4b4b36258fe4c3f",
        "transformation": "private developer location replaced by neutral container location",
    },
    "vendor/repair/sealed_wire_seed.py": {
        "source_sha256": "8fb046b659aee2fb2ae798219ff10a9dacf42cc7a3ca9adda2727ad3761e39ab",
        "transformation": "corrected sealed-wire inventory logic adapted to a model-agnostic CLI",
    },
    "vendor/runtime/mixed_tier_backend.py": {
        "source_sha256": "db14f3603ff2372229d0b34ea290413ef26be6a08acf766006b48ade8633f1e8",
        "transformation": "upstream identity pinned; shipped file is the portable public runtime",
    },
    "vendor/runtime/mixed_prefill_server.py": {
        "source_sha256": "ffe5224742cde697599f43ff56b5c37459e39da9cd60759607c8a5a40bf4edcc",
        "transformation": "upstream identity pinned; shipped file is the portable public runtime",
    },
}

CAPABILITIES = {
    "qtip_rep16": "REP16, rate-contract, LDLQ, and Viterbi QTIP builders",
    "vq_d4": "true D=4 vector-quantized builders and wire materializers",
    "mxfp4": "DeepSeek-v4 MXFP4 quantization and backend selectors",
    "fortress_measure": "Balanced64 and full-rail measurement mechanics",
    "solver_scip": "envelope-exact seeded assignment and corrected-grid solvers",
    "calibration_p930": "residual fitting, interaction, retrodiction, and corrected pricing",
    "repair_p959": "sealed-wire inventory seed plus sparse expert repair",
    "packers": "streaming pack, overlay, planes, and physical-wire packers",
    "runtime": "mixed-tier backend, patch, prefill server, pack contract, and serve shim",
    "pipeline_acceleration": "8-stream mover, hardlink materializer, four-way sharding, per-codebook streaming, replay and revoke gates",
    "recovered_research": "privacy-scrubbed recovered source, receipts, adoption gates, and provenance for P234/P486/P526/P530/P948/P950/P951/P959/P963/P968",
    "kernel": "frozen CUDA extension source and build contract",
    "evalplus": "pinned EvalPlus source, dataset, endpoint generator, paired sanitizer, and scorer",
    "receipt_schemas": "JSON schemas for profiles, assignments, measurements, serving, eval, and receipts",
}

RUNTIME_UPSTREAM = {
    "mixed_tier_backend.py": "db14f3603ff2372229d0b34ea290413ef26be6a08acf766006b48ade8633f1e8",
    "mixed_tier_patch.py": "80696f626254fb3f2be6c95035e1ba13a17ae3ab7a8d50c366f8056cba66dd27",
    "mixed_prefill_server.py": "ffe5224742cde697599f43ff56b5c37459e39da9cd60759607c8a5a40bf4edcc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def capability(relative: str) -> str:
    if relative.startswith("schemas/"):
        return "receipt_schemas"
    if relative.startswith("vendor/builders/qtip-rep16/"):
        return "qtip_rep16"
    if relative.startswith("vendor/builders/mxfp4/"):
        return "mxfp4"
    if relative.startswith("vendor/builders/packers/"):
        return "packers"
    if relative.startswith("vendor/builders/"):
        return "vq_d4"
    if relative.startswith("vendor/solver/"):
        return "solver_scip"
    if relative.startswith("vendor/calibration/"):
        return "calibration_p930"
    if relative.startswith("vendor/repair/"):
        return "repair_p959"
    if relative.startswith("vendor/measure/"):
        return "fortress_measure"
    if relative.startswith("vendor/runtime/"):
        return "runtime"
    if relative.startswith("vendor/pipeline/"):
        return "pipeline_acceleration"
    if relative.startswith("vendor/recovered/"):
        return "recovered_research"
    if relative.startswith("vendor/kernel/"):
        return "kernel"
    if relative.startswith("vendor/eval"):
        return "evalplus"
    raise ValueError("unclassified vendored file: " + relative)


def vendor_files() -> Iterable[Path]:
    roots = [VENDOR, ROOT / "schemas"]
    for base in roots:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path == VENDOR / "VENDOR_INDEX.json":
                continue
            yield path


def package_files() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts and relative.parts[0] == "workspace":
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.as_posix() in {"TOOLS_MANIFEST.json", "PACKAGE_MANIFEST.json"}:
            continue
        yield path


def build_vendor_index() -> Dict[str, Any]:
    recovery_manifest_path = VENDOR / "recovered" / "glm52_research_source_bundle_v1" / "RECOVERY_MANIFEST.json"
    recovery_sources: Dict[str, Dict[str, Any]] = {}
    if recovery_manifest_path.is_file():
        recovery_manifest = json.loads(recovery_manifest_path.read_text())
        recovery_root = recovery_manifest_path.parent.relative_to(ROOT).as_posix()
        recovery_sources = {
            f"{recovery_root}/{row['path']}": row
            for row in recovery_manifest["files"]
        }
    rows: List[Dict[str, Any]] = []
    capability_files: Dict[str, List[str]] = {name: [] for name in CAPABILITIES}
    for path in vendor_files():
        relative = path.relative_to(ROOT).as_posix()
        shipped = sha256(path)
        cap = capability(relative)
        override = SOURCE_OVERRIDES.get(relative, {})
        recovered = recovery_sources.get(relative)
        source = (
            recovered["recovered_bundle_sha256"]
            if recovered is not None
            else override.get("source_sha256", shipped)
        )
        row = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": shipped,
            "source_sha256": source,
            "capability": cap,
            "transformed": source != shipped,
        }
        if override:
            row["transformation"] = override["transformation"]
        elif recovered is not None and recovered["transformed"]:
            row["transformation"] = "privacy-only redaction recorded in RECOVERY_MANIFEST.json"
        rows.append(row)
        capability_files[cap].append(relative)
    capabilities = {
        name: {"description": CAPABILITIES[name], "files": capability_files[name]}
        for name in sorted(CAPABILITIES)
    }
    result: Dict[str, Any] = {
        "schema": "banana-smasher-vendor-index-v1",
        "runtime_upstream_sha256": RUNTIME_UPSTREAM,
        "evalplus": {
            "commit": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e",
            "humanevalplus_release": "HumanEvalPlus-v0.1.10",
            "humanevalplus_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
        },
        "capabilities": capabilities,
        "files": rows,
        "privacy_law": "source and shipped identities are both retained when only privacy or package-local substitutions were required",
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result


def build_tools_manifest() -> Dict[str, Any]:
    vendor_index = json.loads((VENDOR / "VENDOR_INDEX.json").read_text())
    vendored = {row["path"]: row for row in vendor_index["files"]}
    rows = []
    for path in package_files():
        relative = path.relative_to(ROOT).as_posix()
        shipped = sha256(path)
        row = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": shipped,
            "shipped_sha256": shipped,
            "source_sha256": vendored.get(relative, {}).get("source_sha256", shipped),
        }
        if relative in vendored:
            row["capability"] = vendored[relative]["capability"]
            row["transformed"] = vendored[relative]["transformed"]
        rows.append(row)
    result: Dict[str, Any] = {
        "schema": "banana-smasher-tools-manifest-v1",
        "scope": "every package file except this self-referential manifest, generated workspace, and bytecode caches",
        "files": rows,
    }
    result["aggregate_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    vendor_index = build_vendor_index()
    write_json(VENDOR / "VENDOR_INDEX.json", vendor_index)
    (ROOT / "SOURCE_MANIFEST.sha256").write_text("".join(
        "{}  {}\n".format(row["sha256"], row["path"])
        for row in vendor_index["files"]
    ))
    tools_manifest = build_tools_manifest()
    write_json(ROOT / "TOOLS_MANIFEST.json", tools_manifest)
    package_manifest = {
        "schema": "banana-smasher-package-manifest-v1",
        "tools_manifest_sha256": sha256(ROOT / "TOOLS_MANIFEST.json"),
        "tools_aggregate_sha256": tools_manifest["aggregate_sha256"],
        "source_manifest_sha256": sha256(ROOT / "SOURCE_MANIFEST.sha256"),
        "package_files": len(tools_manifest["files"]),
    }
    package_manifest["aggregate_sha256"] = canonical_sha256(package_manifest)
    write_json(ROOT / "PACKAGE_MANIFEST.json", package_manifest)
    print(json.dumps({
        "status": "PASS",
        "vendor_files": len(vendor_index["files"]),
        "package_files": len(tools_manifest["files"]),
        "vendor_aggregate_sha256": vendor_index["aggregate_sha256"],
        "tools_aggregate_sha256": tools_manifest["aggregate_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
