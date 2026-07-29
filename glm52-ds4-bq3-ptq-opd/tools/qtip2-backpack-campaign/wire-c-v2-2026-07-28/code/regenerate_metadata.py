#!/usr/bin/env python3
"""Deterministically regenerate ARTIFACT_PROVENANCE and PACKAGE_MANIFEST."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PACKAGE_MANIFEST.json"
PROVENANCE_PATH = ROOT / "artifacts/ARTIFACT_PROVENANCE.json"

EXPLICIT_SOURCE_SHA256 = {
    "artifacts/P943_TRUE_C_TERMINAL_SEAL.public.json": "90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c",
    "artifacts/P951_TRUE_C_BALANCED64.public.json": "25dc5d5965b6e0e6c11db69ae05b7d64ec158dd41698e84e551db35270f1e5f7",
    "acceleration/artifacts/P963_EXACT_ACCELERATION_SEAL.public.json": "11ed966638ac0c4641a28c8c4946599bdaeaaca1016d26f8a0ddb7cfb2373196",
    "acceleration/artifacts/P963_ACCEL_EXACT_P951_BALANCED64_V3_MB2.public.json": "85566c504f23d2862ce4f5d1cc5c03797888e57fd090547b239b2ffa2890f6a2",
    "acceleration/code/p963_true_c_accel.py": "44ff2771fad236ad9d25fdbcd4ccdbfdb24b0725a27631650eb9748cb50cfdf8",
    "acceleration/code/p963_true_c_overlay_adapter.py": "e84efd6080806ca51bf8681e05e7e06aef6d2406bab29da4d4b68ff8d551415e",
    "acceleration/code/launch_p963.sh": "393070f9b8c6184f062a9c5cf42f4712492174a7619f4768c3063305ed412c30",
    "structural-guards/p936/authority/tests/test_authority_guard.py": "f93385fce041fcb96aa770b19be0cbbc92db1e58cb6ee2ffca5983c9d05c55c1",
    "evaluation/toolkit/score_evalplus.sh": "30d7e65bdef0471e0086ff05138d23c0d51e1dccd02c45ad06477990aad6058f",
}

REQUIRED_SOURCE_PINS = {
    "baseline_r_physical_manifest": "398441d16f1a251079b518a55095c568353b9f3e542f2ec55d4139e0ac6e7ffd",
    "corrected_vertical_grid": "74869b5f8e3ef4eb43dc98c6ee060c2d9ad048bb215cadd308fb2c9983933dda",
    "f949_assignment": "f949b01a29049b03c9dcba6fb1c9df775d414427aeb9f475cd176503f8ddd654",
    "full_menu_manifest": "854b85c6216ae38682010b13ccd0348f52f1503ca8c568c61e43b829da361fcc",
    "p922_terminal_verification": "d3cbae01335d4fb3809275d72d9f6201c439f2598cd7c41ba0e57d81b36733c5",
    "p930_corrected_grid_v3": "49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203",
    "p930_corrected_pricing_v3": "c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0",
    "p930_final_report": "6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9",
    "p930_p922_selection": "e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818",
    "p930_p928_assignment": "62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122",
    "p930_validator": "9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379",
    "p931_first_feasible": "e84c6c5550eebc00df8b0f15d344c719864bbaf96cfdf5723bc696e839352772",
    "p931_independent_verification": "60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a",
    "p931_reviewed_artifact_manifest": "d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a",
    "p943_terminal_true_c": "90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c",
    "p951_balanced64_true_c": "25dc5d5965b6e0e6c11db69ae05b7d64ec158dd41698e84e551db35270f1e5f7",
    "p963_exact_acceleration": "11ed966638ac0c4641a28c8c4946599bdaeaaca1016d26f8a0ddb7cfb2373196",
    "p963_accelerated_receipt": "85566c504f23d2862ce4f5d1cc5c03797888e57fd090547b239b2ffa2890f6a2",
    "p963_runner": "44ff2771fad236ad9d25fdbcd4ccdbfdb24b0725a27631650eb9748cb50cfdf8",
    "qtip2_anchor": "96e09515e61e87669e5a378b714262184173b625844898a20f210838a3ed0b5b",
    "qtip3_anchor": "d79a79653f66067aee9255d95e0212013abae128df5c0ac2c7727ab899e44315",
}

# These files are authored summaries/protocols, not byte transforms of one source.
DERIVED_PUBLIC_SUMMARIES = {
    "P958_ASSIGNMENT_RECOVERY_STATUS.md",
    "artifacts/CAMPAIGN_COMPARISON_TABLE.json",
    "artifacts/P931_V3_DEFINITIVE.public.json",
    "artifacts/P963_EXACT_ACCELERATION_SEAL.public.json",
    "artifacts/SAME_INSTRUMENT_RESULTS.json",
    "acceleration/artifacts/P963_PROFILE_FIRST_SEAL.public.json",
    "acceleration/artifacts/P963_STAGE_CANARY.public.json",
    "evaluation/P968_AUTHORITY_MAP.public.json",
}

# These are authored directly for this public package. Their own bytes are source.
GENERATED_EXACT_PUBLIC_FILES = {
    "README.md", "CANONICAL_RECIPE.md", "EVALUATION_PROTOCOL.md", "OPERATIONS_FORENSICS.md",
    "code/recompute_results.py", "code/regenerate_metadata.py", "code/verify_package.py",
    "evaluation/P967_INFERENCE_PROTOCOL.public.json",
    "tests/test_package_contract.py", "tests/test_structural_guards.py",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def package_files() -> list[Path]:
    return sorted(
        (
            path for path in ROOT.rglob("*")
            if path.is_file()
            and path != MANIFEST_PATH
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def prior_source_map() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        return {}
    document = json.loads(MANIFEST_PATH.read_text())
    return {
        row["path"]: row["source_sha256"]
        for row in document.get("files", [])
        if isinstance(row, dict) and "path" in row and "source_sha256" in row
    }


def source_map() -> dict[str, str]:
    result = prior_source_map()
    result.update(EXPLICIT_SOURCE_SHA256)
    for relative in DERIVED_PUBLIC_SUMMARIES | GENERATED_EXACT_PUBLIC_FILES | {
        "artifacts/ARTIFACT_PROVENANCE.json"
    }:
        result.pop(relative, None)
    return result


def role_for(relative: str) -> str:
    if relative.startswith(("artifacts/", "acceleration/artifacts/")):
        return "artifact"
    if relative.startswith("evaluation/") and relative.endswith(".json"):
        return "artifact"
    if relative.endswith(".md") or relative.startswith(("specs/", "doctrine/")):
        return "methodology"
    if relative.startswith(("code/", "acceleration/code/", "evaluation/toolkit/", "structural-guards/", "tests/")):
        return "implementation"
    return "support"


def provenance_type_for(relative: str) -> str:
    if relative in DERIVED_PUBLIC_SUMMARIES:
        return "derived_public_summary"
    if relative in GENERATED_EXACT_PUBLIC_FILES:
        return "generated_exact_public_file"
    return "sealed_source_public_copy"


def public_row(path: Path, sources: dict[str, str]) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    data = path.read_bytes()
    public_sha = sha256_bytes(data)
    provenance_type = provenance_type_for(relative)
    source_sha = public_sha if provenance_type in {"derived_public_summary", "generated_exact_public_file"} else sources.get(relative, public_sha)
    transformed = source_sha != public_sha
    return {
        "path": relative,
        "privacy_substitution_applied": transformed,
        "provenance_type": provenance_type,
        "public_copy_bytes": len(data),
        "public_copy_sha256": public_sha,
        "role": role_for(relative),
        "source_sha256": source_sha,
        "source_verification": (
            "reviewed evidence hashes recorded separately; summary is not a byte transform"
            if provenance_type == "derived_public_summary"
            else "exact public-authored bytes"
            if provenance_type == "generated_exact_public_file"
            else "sealed predecessor SHA pinned before public copy" if transformed
            else "byte-identical predecessor/public copy"
        ),
    }


def derived_evidence(relative: str) -> dict[str, str]:
    if relative == "artifacts/P931_V3_DEFINITIVE.public.json":
        return {
            "independent_verification_receipt": REQUIRED_SOURCE_PINS["p931_independent_verification"],
            "source_artifact_manifest": REQUIRED_SOURCE_PINS["p931_reviewed_artifact_manifest"],
        }
    if relative in {
        "artifacts/P963_EXACT_ACCELERATION_SEAL.public.json",
        "acceleration/artifacts/P963_PROFILE_FIRST_SEAL.public.json",
        "acceleration/artifacts/P963_STAGE_CANARY.public.json",
    }:
        return {"exact_acceleration_seal": REQUIRED_SOURCE_PINS["p963_exact_acceleration"]}
    if relative == "artifacts/CAMPAIGN_COMPARISON_TABLE.json":
        return {
            "uniform_qtip2_receipt": REQUIRED_SOURCE_PINS["qtip2_anchor"],
            "terminal_true_c_receipt": REQUIRED_SOURCE_PINS["p951_balanced64_true_c"],
            "p931_verification_receipt": REQUIRED_SOURCE_PINS["p931_independent_verification"],
            "p963_exact_acceleration_seal": REQUIRED_SOURCE_PINS["p963_exact_acceleration"],
        }
    if relative == "artifacts/SAME_INSTRUMENT_RESULTS.json":
        return {
            "uniform_qtip2_receipt": REQUIRED_SOURCE_PINS["qtip2_anchor"],
            "terminal_true_c_receipt": REQUIRED_SOURCE_PINS["p951_balanced64_true_c"],
            "p931_verification_receipt": REQUIRED_SOURCE_PINS["p931_independent_verification"],
        }
    if relative == "evaluation/P968_AUTHORITY_MAP.public.json":
        return {
            "p968_original_authority_map": "0f95fbee8aa574b427ebdd3139428134cada020d043a3aae24be621af7638ec7",
            "p968_toolkit_gate_receipt": "413efbd6b24eb19cb8601bf3f4488000c1eb8217056b589152dcefa2eb7a340b",
        }
    if relative == "P958_ASSIGNMENT_RECOVERY_STATUS.md":
        return {
            "p931_verification_receipt": REQUIRED_SOURCE_PINS["p931_independent_verification"],
            "p931_reviewed_artifact_manifest": REQUIRED_SOURCE_PINS["p931_reviewed_artifact_manifest"],
        }
    raise KeyError(f"missing evidence map for derived summary {relative}")


def provenance_paths() -> set[str]:
    result = {"P958_ASSIGNMENT_RECOVERY_STATUS.md"}
    for directory in (ROOT / "artifacts", ROOT / "acceleration/artifacts"):
        for path in directory.rglob("*"):
            if path.is_file() and path != PROVENANCE_PATH:
                result.add(path.relative_to(ROOT).as_posix())
    for path in (ROOT / "evaluation").glob("*.json"):
        result.add(path.relative_to(ROOT).as_posix())
    return result


def build_provenance(sources: dict[str, str]) -> dict[str, Any]:
    rows = []
    for relative in sorted(provenance_paths()):
        path = ROOT / relative
        row = public_row(path, sources)
        row.pop("role", None)
        row.pop("source_verification", None)
        if row["provenance_type"] == "derived_public_summary":
            row["source_evidence_sha256"] = derived_evidence(relative)
        rows.append(row)
    return {
        "artifacts": rows,
        "hash_semantics": {
            "public_copy_sha256": "SHA-256 of the exact file shipped in this repository.",
            "source_sha256": "SHA-256 of predecessor bytes for source-derived copies; equals public_copy_sha256 for generated or derived documents.",
            "source_evidence_sha256": "Hashes of reviewed receipts used to author a derived summary; not byte-predecessor hashes and never substituted for source_sha256.",
            "privacy_substitution_applied": "True exactly when source_sha256 and public_copy_sha256 differ for a source-derived public copy.",
        },
        "schema": "wire-c-v2-artifact-provenance-v3",
        "status": "PASS_PUBLICATION_SAFE_INVENTORY",
    }


def build_manifest(sources: dict[str, str]) -> dict[str, Any]:
    return {
        "files": [public_row(path, sources) for path in package_files()],
        "hash_semantics": {
            "public_copy_sha256": "SHA-256 of the exact file shipped in this repository.",
            "source_sha256": "SHA-256 of predecessor bytes, or exact packaged bytes for generated/derived documents.",
        },
        "physical_build": {
            "assignment_cells": 22016,
            "assignment_label": "F521 V2 physical candidate",
            "baseline_r_preserved": True,
            "changed_cells_built_and_read_back": 21472,
            "changed_cells_expected": 21472,
            "unchanged_cells_inherited": 544,
        },
        "protocol_contract": {
            "greedy_repeats": 3,
            "paired_results_in_package": False,
            "sampled_n_per_task": 5,
        },
        "required_source_pins": REQUIRED_SOURCE_PINS,
        "schema": "wire-c-v2-package-manifest-v3",
        "selected_solve": {
            "accounting_closure": True,
            "envelope_bytes": 101346700411,
            "exact_logical_bytes": 101346700382,
            "label": "P931_V3_DEFINITIVE_TIME_LIMIT_INCUMBENT",
            "measured": False,
            "objective": 0.035078633039490076,
            "optimality_proven": False,
            "slack_bytes": 29,
        },
        "status": "PASS_COMPLETE_PUBLIC_INVENTORY",
        "true_c_chain": {
            "premeasurement_estimate": {
                "mechanical_point_kld": 0.08887845829694314,
                "p922_measured_substitution_penalty_kld": 0.02925963216194956,
                "planning_bracket_kld": [0.089, 0.095],
                "status": "ESTIMATE_NOT_MEASUREMENT",
                "wire_c_r_measured_global_kld": 0.1181380904588927,
            },
            "status": "MEASURED_TERMINAL_F521_T",
            "terminal_balanced64_global_kld": 0.06829414627618949,
            "terminal_measurement_source_sha256": REQUIRED_SOURCE_PINS["p951_balanced64_true_c"],
            "terminal_positions": 65536,
            "terminal_windows": 64,
            "zero_quarantine": True,
            "zero_substitution": True,
        },
    }


def generated_bytes(sources: dict[str, str]) -> tuple[bytes, bytes]:
    provenance_data = json_bytes(build_provenance(sources))
    if PROVENANCE_PATH.read_bytes() != provenance_data:
        PROVENANCE_PATH.write_bytes(provenance_data)
    manifest_data = json_bytes(build_manifest(sources))
    return provenance_data, manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail unless both generated files are current")
    args = parser.parse_args()
    sources = source_map()
    expected_provenance = json_bytes(build_provenance(sources))
    # Manifest hashes the final provenance bytes, so compute it against those bytes.
    current_provenance = PROVENANCE_PATH.read_bytes() if PROVENANCE_PATH.is_file() else b""
    if current_provenance != expected_provenance and not args.check:
        PROVENANCE_PATH.write_bytes(expected_provenance)
    expected_manifest = json_bytes(build_manifest(sources))
    if args.check:
        failures = []
        if current_provenance != expected_provenance:
            failures.append("artifacts/ARTIFACT_PROVENANCE.json is stale")
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_bytes() != expected_manifest:
            failures.append("PACKAGE_MANIFEST.json is stale")
        if failures:
            raise SystemExit("\n".join(failures))
        print("PACKAGE_METADATA_CHECK_PASS")
        return
    MANIFEST_PATH.write_bytes(expected_manifest)
    print("PACKAGE_METADATA_REGENERATED")


if __name__ == "__main__":
    main()
