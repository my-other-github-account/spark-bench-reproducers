#!/usr/bin/env python3
"""Fail-closed verification for the public Wire-C V2 package."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
SELF_FILES = {"PACKAGE_MANIFEST.json"}
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

P931_VALIDITY = {
    "measured": False,
    "optimality_proven": False,
    "physical_checkpoint_scored": False,
    "result_type": "PROJECTED_SOLVER_RESULT",
    "source_payloads_redistributed": False,
    "status": "PROJECTED_DEFINITIVE_FEASIBLE__TIME_LIMIT_INCUMBENT__NOT_PHYSICAL_MEASUREMENT",
    "true_c_measurement_status": "PENDING_DIRECT_P937_P939",
}
P931_SOLVER = {
    "backend": "OR-Tools MPSolver SCIP",
    "best_bound": 0.03507853638367621,
    "envelope_bytes": 101346700411,
    "exact_bytes": 101346700382,
    "integer_status": "FEASIBLE",
    "objective_reweighted": 0.035078633039490076,
    "relative_gap": 2.7554042339551337e-06,
    "slack_bytes": 29,
    "termination_reason": "TIME_LIMIT_WITH_FEASIBLE_INCUMBENT",
    "threads": 16,
    "time_limit_seconds": 3600.0,
    "wall_seconds": 4214.848067998886,
}
P931_CLASSES = {
    "agentic": 0.03958745712003608,
    "chat": 0.008532822455696412,
    "code": 0.05105645114446229,
    "multilingual": 0.04864400347424909,
    "prose": 0.0352992491602176,
    "reasoning": 0.005687227334670042,
}
P931_TIERS = {
    "d4_k1024": 1370,
    "d4_k2048": 4520,
    "d4_k256": 2,
    "d4_k4096": 1478,
    "d4_k512": 4,
    "d4_k8192": 16,
    "d8_k1024": 0,
    "d8_k2048": 0,
    "d8_k256": 0,
    "d8_k4096": 0,
    "d8_k512": 0,
    "native_mxfp4": 39,
    "qtip15_1.509117": 0,
    "qtip2_2.0117": 1829,
    "qtip3_3.0117": 12758,
}
P931_INPUT_SHAS = {
    "p922_selection": "e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818",
    "p928_assignment": "62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122",
    "p930_final_report": "6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9",
    "pre_v3_grid": "74869b5f8e3ef4eb43dc98c6ee060c2d9ad048bb215cadd308fb2c9983933dda",
    "pricing_v3": "c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0",
    "v3_grid": "49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203",
    "v3_validation": "9666d979b79ec576f55a4ea685bb1311b29910875fc227a24f470370e516b379",
}
P931_CODE_SHAS = {
    "launch_p931_v3": "bc2cc6a9775f361ad75532a62b1edebf6e127765f01a4f74517418e7f8bbe6ef",
    "pricing_v3_surface": "04b7d53935362b7d71622fba53e2d8170f51e70a74303e47081948d3203e0fc9",
    "retrodiction_pricing": "466bba58afd00cc12d6b157281eda8a5d06fb5807b89874b5a6497ee29c3047e",
    "solve_p693_turbo": "a6052cabce24d218f7ba2161910f70e77abddf51d953a07a7132158b75dc2c04",
    "solve_p760_turbo": "db1cdc17dc9aae78d2283e26f7dff241d02b81b5a04af662ab213346dbcdf7a9",
    "solve_p924_reweighted_unchanged_step1": "fc99323ecea76d52b3640e96894ded0ee684987d66abea38cde6cc8be4dfd355",
    "solve_p931_v3": "c70985c1b35bae48b09945db6c343c9d6099550c409006e5694b4b9a8aced1eb",
    "verify_p931_v3": "7dca34898869db1d2501ba1761b4a3429062659b7f32fd1f7ea281430959838a",
}
P931_SOURCE_OUTPUT_SHAS = {
    "outputs/P931_V3_DEFINITIVE/ASSIGNMENT_QTIP2_QTIP3.json": "b8f07185e54f018af4bcc2b6831b457b1ada2c97ad166226ee19e3e8e91bbd8d",
    "outputs/P931_V3_DEFINITIVE/DOMINANCE_PRUNING.json": "5f6f35d8ec5aaa7e26d9b882df22732688bab46d86661b2942e699cc8a616e76",
    "outputs/P931_V3_DEFINITIVE/DONE.json": "a469a4aa27279b24348bed4d7b68f9355a78f026fb4b566c8adf3737d3f84068",
    "outputs/P931_V3_DEFINITIVE/FIRST_FEASIBLE.json": "e84c6c5550eebc00df8b0f15d344c719864bbaf96cfdf5723bc696e839352772",
    "outputs/P931_V3_DEFINITIVE/PAIRED_GREEDY_LEDGER.jsonl": "b295eef46adcc2010579ec5b88d5d51ad224ccfde17743ca5b385fd5ab32f0d1",
    "outputs/P931_V3_DEFINITIVE/PAIRED_GREEDY_SEED.json": "d478665266aae5b2efa6583b6a5acae0502d5c2c4acc463883c1a85f6e15e927",
    "outputs/P931_V3_DEFINITIVE/PROGRESS.json": "540c186662cf0813aee2d1bba7607041f933bf5ab6d45642087bde13edca4805",
    "outputs/P931_V3_DEFINITIVE/PURCHASE_TABLE.json": "1825ded640a86d421895f8d727f89fae955728817a394678976037b87e692eca",
    "outputs/P931_V3_DEFINITIVE/RESULT.json": "b61a3bcd955d8ce62141abb71a3307e5be72be74deb575358d1d1e9867dcee08",
    "outputs/P931_V3_DEFINITIVE/SANITY.json": "d2ee4af35212cbf78cba7ae9efc1999685a5b774d65c96710cdb47e8d501b342",
    "outputs/P931_V3_DEFINITIVE/SCIP.log": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "outputs/P931_V3_DEFINITIVE/STDOUT.jsonl": "f8a811a3a57fec8d3f64aa2beb8bc948b7b04456c12ca5a8152dc19ddfd0acae",
}
P931_ALIASES = {
    "outputs/P931_V3_DEFINITIVE/FIRST_FEASIBLE.json": "outputs/P931_V3_DEFINITIVE/FIRST_FEASIBLE_PREVIEW.json",
    "outputs/P931_V3_DEFINITIVE/PAIRED_GREEDY_LEDGER.jsonl": "outputs/P931_V3_DEFINITIVE/PAIRED_GREEDY_LEDGER.json",
    "outputs/P931_V3_DEFINITIVE/PAIRED_GREEDY_SEED.json": "outputs/P931_V3_DEFINITIVE/WIRE_C_V3_DEFINITIVE_PAIRED_GREEDY_SEED.json",
}
P951_CLASSES = {
    "agentic": 0.07879656974187459,
    "chat": 0.021183150884045005,
    "code": 0.05501697946566645,
    "multilingual": 0.11238435483229318,
    "prose": 0.09759553403503682,
    "reasoning": 0.014495197391988604,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def expect(failures: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{label}: got {actual!r}, expected {expected!r}")


def validate_manifest(root: Path, failures: list[str]) -> dict[str, dict[str, Any]]:
    try:
        document = load(root, "PACKAGE_MANIFEST.json")
    except (OSError, ValueError) as exc:
        failures.append(f"cannot load PACKAGE_MANIFEST.json: {exc}")
        return {}
    expect(failures, "manifest schema", document.get("schema"), "wire-c-v2-package-manifest-v3")
    expect(failures, "manifest status", document.get("status"), "PASS_COMPLETE_PUBLIC_INVENTORY")
    rows = document.get("files")
    if not isinstance(rows, list):
        failures.append("manifest files must be a list")
        return {}
    paths = [row.get("path") for row in rows if isinstance(row, dict)]
    if len(paths) != len(rows) or len(paths) != len(set(paths)):
        failures.append("manifest paths are absent or duplicated")
    if paths != sorted(paths):
        failures.append("manifest paths are not sorted")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    listed = set(paths)
    extras = sorted(actual_paths - listed - SELF_FILES)
    missing = sorted(listed - actual_paths)
    if extras:
        failures.append(f"unmanifested files: {extras}")
    if missing:
        failures.append(f"manifested files missing from tree: {missing}")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        relative = str(row.get("path", ""))
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            failures.append(f"unsafe manifest path: {relative!r}")
            continue
        source_sha = row.get("source_sha256")
        public_sha = row.get("public_copy_sha256")
        if not isinstance(source_sha, str) or not HEX64.fullmatch(source_sha):
            failures.append(f"invalid source SHA for {relative}")
        if not isinstance(public_sha, str) or not HEX64.fullmatch(public_sha):
            failures.append(f"invalid public SHA for {relative}")
        provenance_type = row.get("provenance_type", "sealed_source_public_copy")
        privacy = row.get("privacy_substitution_applied")
        if provenance_type == "derived_public_summary":
            if privacy is not False or source_sha != public_sha:
                failures.append(f"derived-summary hash semantics invalid for {relative}")
        elif privacy is not (source_sha != public_sha):
            failures.append(f"privacy/hash inequality invariant failed for {relative}")
        path = root / relative
        if path.is_file():
            if path.stat().st_size != row.get("public_copy_bytes"):
                failures.append(f"byte-count mismatch {relative}")
            if sha256(path) != public_sha:
                failures.append(f"hash mismatch {relative}")
        by_path[relative] = row
    return by_path


def validate_p931_document(document: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expect(failures, "P931 schema", document.get("schema"), "p931-v3-definitive-public-summary-v1")
    expect(failures, "P931 status", document.get("status"), "PASS_DEFINITIVE_V3_ASSIGNMENT_REPLAY_AND_COMPARISON")
    expect(failures, "P931 public validity", document.get("public_validity"), P931_VALIDITY)
    expect(failures, "P931 solver", document.get("solver"), P931_SOLVER)
    expect(failures, "P931 prediction vector", document.get("prediction_by_class"), P931_CLASSES)
    assignment = document.get("assignment", {})
    expect(failures, "P931 assignment SHA", assignment.get("assignment_sha256"), "b8f07185e54f018af4bcc2b6831b457b1ada2c97ad166226ee19e3e8e91bbd8d")
    expect(failures, "P931 assignment-map SHA", assignment.get("assignment_map_sha256"), "c260b1a05ad1368a9aa11ee184fbe0a2734e781c7bbf4e1bf40c67821ea8786c")
    expect(failures, "P931 cell count", assignment.get("cell_count"), 22016)
    expect(failures, "P931 changed-cell count", assignment.get("changed_cell_count_vs_original"), 20718)
    expect(failures, "P931 tier counts", assignment.get("tier_counts"), P931_TIERS)
    expect(failures, "P931 assignment payload availability", assignment.get("assignment_payload_redistributed"), False)
    expect(failures, "P931 recovery status", assignment.get("assignment_recovery_status"), "HASHES_ONLY__DURABLE_EXACT_MAP_PUBLICATION_PENDING")
    expect(failures, "P931 lineage input SHAs", document.get("p930_lineage", {}).get("input_shas"), P931_INPUT_SHAS)
    expect(failures, "P931 source-code SHAs", document.get("source_code_shas"), P931_CODE_SHAS)
    verification = document.get("verification", {})
    expect(failures, "P931 verification status", verification.get("status"), "PASS_DEFINITIVE_V3_ASSIGNMENT_REPLAY_AND_COMPARISON")
    expect(failures, "P931 verification receipt SHA", verification.get("source_receipt_sha256"), "60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a")
    expect(failures, "P931 verification receipt path", verification.get("source_receipt_path"), "outputs/P931_V3_DEFINITIVE/P931_DEFINITIVE_VERIFICATION.json")
    expect(failures, "P931 source-output map", verification.get("source_output_shas"), P931_SOURCE_OUTPUT_SHAS)
    expect(failures, "P931 source aliases", verification.get("source_to_reviewed_manifest_aliases"), P931_ALIASES)
    expect(failures, "P931 source output count", verification.get("source_output_count"), 12)
    artifact_manifest = verification.get("artifact_manifest", {})
    expect(failures, "P931 artifact-manifest SHA", artifact_manifest.get("sha256"), "d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a")
    expect(failures, "P931 artifact-manifest count", artifact_manifest.get("entry_count"), 13)
    expect(failures, "P931 gates passed", verification.get("gates_passed"), 17)
    expect(failures, "P931 gates total", verification.get("gates_total"), 17)
    solver = document.get("solver", {})
    if solver.get("envelope_bytes") - solver.get("exact_bytes", 0) != solver.get("slack_bytes"):
        failures.append("P931 byte closure failed")
    return failures


def validate_p931(root: Path, failures: list[str]) -> None:
    document = load(root, "artifacts/P931_V3_DEFINITIVE.public.json")
    failures.extend(validate_p931_document(document))
    same = load(root, "artifacts/SAME_INSTRUMENT_RESULTS.json")
    projection = same.get("solver_projections", {}).get("p931_v3_definitive", {})
    expected = {
        "artifact": "artifacts/P931_V3_DEFINITIVE.public.json",
        "assignment_map_sha256": document["assignment"]["assignment_map_sha256"],
        "assignment_payload_redistributed": False,
        "assignment_recovery_status": document["assignment"]["assignment_recovery_status"],
        "assignment_recovery_status_url": document["assignment"]["recovery_status_url"],
        "assignment_sha256": document["assignment"]["assignment_sha256"],
        "best_bound": document["solver"]["best_bound"],
        "changed_cell_count_vs_original": document["assignment"]["changed_cell_count_vs_original"],
        "exact_bytes": document["solver"]["exact_bytes"],
        "integer_status": document["solver"]["integer_status"],
        "manifest_sha256": document["verification"]["artifact_manifest"]["sha256"],
        "measured": False,
        "objective_reweighted": document["solver"]["objective_reweighted"],
        "optimality_proven": False,
        "physical_checkpoint_scored": False,
        "prediction_by_class": document["prediction_by_class"],
        "relative_gap": document["solver"]["relative_gap"],
        "result_type": "PROJECTED_SOLVER_RESULT",
        "slack_bytes": document["solver"]["slack_bytes"],
        "source_payloads_redistributed": False,
        "status": P931_VALIDITY["status"],
        "termination_reason": document["solver"]["termination_reason"],
        "threads": document["solver"]["threads"],
        "time_limit_seconds": document["solver"]["time_limit_seconds"],
        "verification_sha256": document["verification"]["source_receipt_sha256"],
        "wall_seconds": document["solver"]["wall_seconds"],
        "true_c_measurement_status": "PENDING_DIRECT_P937_P939",
    }
    expect(failures, "same-instrument P931 projection", projection, expected)


def validate_true_c(root: Path, failures: list[str]) -> None:
    p943 = load(root, "artifacts/P943_TRUE_C_TERMINAL_SEAL.public.json")
    expect(failures, "P943 schema", p943.get("schema"), "p943-true-c-terminal-seal-v1")
    expect(failures, "P943 status", p943.get("status"), "PASS_TERMINAL_F521_T")
    expect(failures, "P943 codebooks", p943.get("codebooks"), 80)
    expect(failures, "P943 codebook ledger length", len(p943.get("codebook_ledger", [])), 80)
    expect(failures, "P943 target rows", p943.get("target_rows"), 2860)
    expect(failures, "P943 ledger row sum", sum(int(row["rows"]) for row in p943.get("codebook_ledger", [])), 2860)
    expect(failures, "P943 pack fraction", p943.get("pack_fraction"), 1.0)
    expect(failures, "P943 zero quarantine", p943.get("zero_quarantine"), True)
    expect(failures, "P943 zero substitution", p943.get("zero_substitution"), True)
    expect(failures, "P943 active overlay SHA", p943.get("active_overlay_sha256"), "9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62")
    expect(failures, "P943 delta-manifest SHA", p943.get("delta_manifest_sha256"), "6d13b82d49c49c55c4215b662cad4c488a1b8c81fb39a32e03096562ba604dc6")
    expect(failures, "P943 build identity", p943.get("build_identity_sha256"), "13d1f887f8e6055f1f579730c2cc37be1e6c0754dd02256cf35a3a9f8c2d0a2f")
    expect(failures, "P943 validity", p943.get("public_validity", {}).get("status"), "SEALED_REFIT_AUTHORITY")

    p951 = load(root, "artifacts/P951_TRUE_C_BALANCED64.public.json")
    expect(failures, "P951 schema", p951.get("schema"), "p951-independent-terminal-true-c-balanced64-v1")
    expect(failures, "P951 status", p951.get("status"), "PASS_INDEPENDENT_TERMINAL_TRUE_C_80_OF_80")
    expect(failures, "P951 validity", p951.get("public_validity", {}).get("status"), "MEASURED_TERMINAL_TRUE_C_F521_T_BALANCED64_V1")
    expect(failures, "P951 global KLD", p951.get("global", {}).get("mean"), 0.06829414627618949)
    expect(failures, "P951 windows", p951.get("windows"), 64)
    expect(failures, "P951 positions", p951.get("global", {}).get("n_positions"), 65536)
    expect(failures, "P951 support", p951.get("support"), 8192)
    expect(failures, "P951 cutoff", p951.get("cutoff"), 1024)
    expect(failures, "P951 direction", p951.get("direction"), "KL(teacher||candidate)")
    expect(failures, "P951 class vector", {name: p951.get("six_classes", {}).get(name, {}).get("mean") for name in CLASSES}, P951_CLASSES)
    expect(failures, "P951 class window closure", sum(int(p951["six_classes"][name]["n_windows"]) for name in CLASSES), 64)
    expect(failures, "P951 class position closure", sum(int(p951["six_classes"][name]["n_positions"]) for name in CLASSES), 65536)
    coverage = p951.get("coverage", {})
    expect(failures, "P951 changed cells", coverage.get("changed_cells_applied"), 21472)
    expect(failures, "P951 expected changed cells", coverage.get("changed_cells_expected"), 21472)
    expect(failures, "P951 coverage layers", coverage.get("coverage_layers"), list(range(43)))
    expect(failures, "P951 completed layers", coverage.get("completed_layers"), list(range(43)))
    expect(failures, "P951 overlay retired", coverage.get("overlay_stage_retired"), True)
    exactness = p951.get("exactness", {})
    expect(failures, "P951 pack fraction", exactness.get("pack_fraction"), 1.0)
    expect(failures, "P951 zero quarantine", exactness.get("zero_quarantine"), True)
    expect(failures, "P951 zero substitution", exactness.get("zero_substitution"), True)
    expect(failures, "P951 output-set SHA", p951.get("outputs", {}).get("window_output_set_sha256"), "3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451")
    expect(failures, "P951 instrument identity", p951.get("instrument_id_sha256"), "c71b24d8c94927661d3aecd8899d59f0c825c9e7cd362b509372f202e2d31d50")


def validate_p963_document(document: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expect(failures, "P963 schema", document.get("schema"), "p963-exact-acceleration-public-v1")
    expect(failures, "P963 status", document.get("status"), "PASS_EXACT_EQUAL_ACCELERATION_GE_2X")
    baseline = document.get("baseline", {})
    accelerated = document.get("accelerated", {})
    expect(failures, "P963 baseline output SHA", baseline.get("output_set_sha256"), "3529d33893a12d92dda96beba29c1a0e21adec6d008f2b32ced7d0066662c451")
    expect(failures, "P963 accelerated output SHA", accelerated.get("output_set_sha256"), baseline.get("output_set_sha256"))
    expect(failures, "P963 global KLD", document.get("exactness", {}).get("global_mean"), 0.06829414627618949)
    expect(failures, "P963 class vector", document.get("exactness", {}).get("six_class_means"), P951_CLASSES)
    expect(failures, "P963 tensor exactness", document.get("exactness", {}).get("tensor_exact_windows"), "64/64")
    expect(failures, "P963 max per-position delta", document.get("exactness", {}).get("maximum_absolute_per_position_delta"), 0.0)
    expect(failures, "P963 raw output exact", document.get("exactness", {}).get("raw_output_set_sha_exact"), True)
    base_seconds = baseline.get("elapsed_seconds")
    accel_seconds = accelerated.get("elapsed_seconds")
    if not isinstance(base_seconds, (int, float)) or not isinstance(accel_seconds, (int, float)) or accel_seconds <= 0:
        failures.append("P963 timing fields invalid")
    else:
        speedup = base_seconds / accel_seconds
        reduction = (base_seconds - accel_seconds) / base_seconds * 100.0
        if not math.isclose(speedup, document.get("comparison", {}).get("speedup", -1.0), rel_tol=0.0, abs_tol=1e-12):
            failures.append("P963 speedup arithmetic drift")
        if not math.isclose(reduction, document.get("comparison", {}).get("wall_clock_reduction_percent", -1.0), rel_tol=0.0, abs_tol=1e-12):
            failures.append("P963 wall-reduction arithmetic drift")
    expect(failures, "P963 implementation SHAs", document.get("implementation"), {
        "adapter_sha256": "e84efd6080806ca51bf8681e05e7e06aef6d2406bab29da4d4b68ff8d551415e",
        "canonical_builder_sha256": "873a98a37a6cf854572983ebfdffc15e2292f6599a7fb3206c14cb866f2f8784",
        "canonical_loader_sha256": "155310d1e6701d6cb2d1c04558514366a2304cb2a8d6d26402ed7c800b8b6c89",
        "canonical_scorer_sha256": "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f",
        "launcher_sha256": "393070f9b8c6184f062a9c5cf42f4712492174a7619f4768c3063305ed412c30",
        "runner_sha256": "44ff2771fad236ad9d25fdbcd4ccdbfdb24b0725a27631650eb9748cb50cfdf8",
        "stage_canary_sha256": "afe37a424266679edb5c50fb9fd2ad68ed9c1e8bf306ad7fd6ddafc286684831",
    })
    expect(failures, "P963 source artifact SHA", document.get("public_validity", {}).get("source_artifact_sha256"), "11ed966638ac0c4641a28c8c4946599bdaeaaca1016d26f8a0ddb7cfb2373196")
    return failures


def validate_p963(root: Path, failures: list[str]) -> None:
    document = load(root, "artifacts/P963_EXACT_ACCELERATION_SEAL.public.json")
    failures.extend(validate_p963_document(document))
    detailed = load(root, "acceleration/artifacts/P963_EXACT_ACCELERATION_SEAL.public.json")
    for arm in ("baseline", "accelerated"):
        for key in ("elapsed_seconds", "output_set_sha256", "receipt_sha256", "windows_per_minute"):
            expect(failures, f"P963 detailed {arm}.{key}", detailed.get(arm, {}).get(key), document.get(arm, {}).get(key))
    expect(failures, "P963 detailed comparison", detailed.get("comparison"), document.get("comparison"))
    expect(failures, "P963 detailed exactness", detailed.get("exactness"), document.get("exactness"))


def validate_protocols(root: Path, failures: list[str]) -> None:
    p967 = load(root, "evaluation/P967_INFERENCE_PROTOCOL.public.json")
    expect(failures, "P967 status", p967.get("status"), "PREREGISTERED_NOT_EXECUTED_IN_THIS_PACKAGE")
    expect(failures, "P967 sampled n", p967.get("sampled_extension", {}).get("n_per_task"), 5)
    expect(failures, "P967 sampled seeds", p967.get("sampled_extension", {}).get("seed_ordinals"), [10000, 10004])
    expect(failures, "P967 greedy repeats", p967.get("greedy_instability", {}).get("repeats"), 3)
    expect(failures, "P967 greedy seeds", p967.get("greedy_instability", {}).get("seed_ordinals"), [20000, 20002])
    p968 = load(root, "evaluation/P968_AUTHORITY_MAP.public.json")
    expect(failures, "P968 public status", p968.get("public_validity", {}).get("status"), "PREREGISTERED_PROTOCOL_ONLY")
    expect(failures, "P968 paired-result availability", p968.get("public_validity", {}).get("true_c_paired_results_available"), False)
    expect(failures, "P968 sampled n", p968.get("protocol_preregistration", {}).get("sampled", {}).get("n_per_task"), 5)
    expect(failures, "P968 greedy repeats", p968.get("protocol_preregistration", {}).get("greedy_instability", {}).get("repeats"), 3)
    try:
        namespace = runpy.run_path(str(root / "evaluation/toolkit/p968_common.py"))
        arms = namespace["ARMS"]
    except Exception as exc:
        failures.append(f"cannot load P968 toolkit arms: {exc}")
    else:
        expect(failures, "P968 toolkit sampled arm", arms.get("sampled"), {"n": 5, "temperature": 0.2, "top_p": 0.95, "seed_start": 10000})
        expect(failures, "P968 toolkit greedy arm", arms.get("greedy"), {"n": 3, "temperature": 0.0, "top_p": 1.0, "seed_start": 20000})


def validate_comparison(root: Path, failures: list[str]) -> None:
    table = load(root, "artifacts/CAMPAIGN_COMPARISON_TABLE.json")
    expect(failures, "comparison status", table.get("status"), "PASS_EXPLICIT_COMPARABILITY_GROUPS")
    campaigns = {row.get("campaign"): row for row in table.get("campaigns", [])}
    expect(failures, "comparison campaign set", set(campaigns), {
        "A_forensic_reconstruction", "B_uniform_qtip2_baseline", "C_corrected_pricing_solver",
        "D_exact_refit_and_terminal_true_c", "E_exact_equal_scorer_acceleration", "F_inference_and_eval_protocol",
        "G_direct_iq4_reference",
    })
    expect(failures, "P931/P951 comparability", table.get("one_instrument_verdicts", {}).get("p931_projection_vs_p951_measurement", {}).get("verdict"), "NOT_COMPARABLE")
    expect(failures, "functional evaluation result label", table.get("one_instrument_verdicts", {}).get("p967_p968", {}).get("verdict"), "NO_RESULT")
    direct = table.get("one_instrument_verdicts", {}).get("terminal_true_c_vs_direct_iq4", {})
    expect(failures, "direct IQ4 KLD", direct.get("direct_iq4_global_kld"), 0.07204393760942278)
    expect(failures, "direct true-C KLD", direct.get("terminal_true_c_global_kld"), 0.06829414627618949)
    expect(failures, "direct KLD delta", direct.get("delta_true_c_minus_iq4"), -0.0037497913332332905)
    expect(failures, "direct relative reduction", direct.get("true_c_lower_than_iq4_fraction"), 0.05204867276358931)
    expect(failures, "direct IQ4 receipt", direct.get("direct_iq4_receipt_sha256"), "abb2031865874c0025719889064f5b0e4f7c5a55cfb3ee2916a924ed348bdf07")
    same = load(root, "artifacts/SAME_INSTRUMENT_RESULTS.json")
    iq4 = next((row for row in same.get("rows", []) if row.get("key") == "iq4_reference"), {})
    expect(failures, "registry IQ4 KLD", iq4.get("global_kld"), 0.07204393760942278)
    expect(failures, "registry IQ4 receipt", iq4.get("receipt_sha256"), "abb2031865874c0025719889064f5b0e4f7c5a55cfb3ee2916a924ed348bdf07")
    expect(failures, "registry IQ4 positions", iq4.get("sealed_finite_positions"), 524288)
    baseline = campaigns.get("B_uniform_qtip2_baseline", {}).get("global_kld")
    terminal = campaigns.get("D_exact_refit_and_terminal_true_c", {}).get("global_kld")
    verdict = table.get("one_instrument_verdicts", {}).get("balanced64_uniform_qtip2_vs_terminal_true_c", {})
    if isinstance(baseline, (int, float)) and isinstance(terminal, (int, float)):
        expect(failures, "comparison delta", verdict.get("delta_terminal_minus_baseline"), terminal - baseline)
        if not math.isclose(verdict.get("relative_kld_reduction", -1), (baseline - terminal) / baseline, rel_tol=0.0, abs_tol=1e-15):
            failures.append("comparison relative reduction arithmetic drift")


def validate_runtime_closure(root: Path, failures: list[str]) -> None:
    document = load(root, "artifacts/RUNTIME_CONTAINER_CLOSURE.public.json")
    expect(failures, "runtime closure schema", document.get("schema"), "wire-c-runtime-container-closure-public-v1")
    phases = {row.get("phase"): row for row in document.get("phases", [])}
    expect(failures, "runtime closure phase set", set(phases), {"P970", "P987", "P991", "P993", "P994", "P997"})
    expect(failures, "P987 status", phases.get("P987", {}).get("board_status"), "BLOCKED_REVIEW_REQUIRED")
    expect(
        failures,
        "P987 blocker",
        phases.get("P987", {}).get("blocker_receipt_sha256"),
        "8a71e05d27dd59f0597ef5742a4494325ddb8055afd8408e8b61b0bab4b96322",
    )
    expect(failures, "P991 status", phases.get("P991", {}).get("board_status"), "DONE")
    expect(failures, "P991 endpoint ready", phases.get("P991", {}).get("full_endpoint", {}).get("http_ready"), False)
    expect(failures, "P993 status", phases.get("P993", {}).get("board_status"), "DONE")
    expect(
        failures,
        "P993 ABI seal",
        phases.get("P993", {}).get("receipts", {}).get("abi_closure_seal_sha256"),
        "7fcacaeba2492b33af936abee3c595d80fa29e5084f662ccdea230d46cd607d9",
    )
    expect(failures, "P994 status", phases.get("P994", {}).get("board_status"), "DONE")
    expect(
        failures,
        "P994 winner",
        phases.get("P994", {}).get("winner_receipt_sha256"),
        "4d5f9e75e5e94272719634c8528e558643eeb62ea2d3812cbb0b1ab20dadf343",
    )
    expect(
        failures,
        "P997 status",
        phases.get("P997", {}).get("board_status"),
        "BLOCKED_PARENT__SUCCESSOR_LANES_RUNNING_AT_SNAPSHOT",
    )
    expect(
        failures,
        "P997 G1 status",
        phases.get("P997", {}).get("candidate_g1", {}).get("status"),
        "INCOMPLETE_G1__LEGACY_VLLM_C_ABSENT",
    )
    expect(
        failures,
        "P997 no promotion",
        phases.get("P997", {}).get("promotion_gate", {}).get("status"),
        "FAIL_NO_TAG_MOVE",
    )


def validate_provenance(root: Path, manifest: dict[str, dict[str, Any]], failures: list[str]) -> None:
    provenance = load(root, "artifacts/ARTIFACT_PROVENANCE.json")
    semantics = provenance.get("hash_semantics", {})
    if "not byte-predecessor" not in semantics.get("source_evidence_sha256", ""):
        failures.append("provenance source-evidence semantics missing")
    for row in provenance.get("artifacts", []):
        relative = row.get("path")
        manifest_row = manifest.get(relative)
        if manifest_row is None:
            failures.append(f"provenance path absent from manifest: {relative}")
            continue
        for key in ("source_sha256", "public_copy_sha256", "public_copy_bytes", "privacy_substitution_applied"):
            expect(failures, f"provenance {relative}.{key}", row.get(key), manifest_row.get(key))
        if row.get("provenance_type") == "derived_public_summary" and "source_evidence_sha256" not in row:
            failures.append(f"derived summary lacks source-evidence map: {relative}")
    p931 = next((row for row in provenance.get("artifacts", []) if row.get("path") == "artifacts/P931_V3_DEFINITIVE.public.json"), None)
    if p931 is None:
        failures.append("P931 provenance row missing")
    else:
        expect(failures, "P931 provenance type", p931.get("provenance_type"), "derived_public_summary")
        expect(failures, "P931 provenance evidence", p931.get("source_evidence_sha256"), {
            "independent_verification_receipt": "60e6573f717e78fa8039a64938d7e444661e6583bd2c1549aa15906fcba4703a",
            "source_artifact_manifest": "d13db9c39f2da6620c432ba75ec1a5c45b1852b766418cc2e8d8a2b09e9e312a",
        })


def validate_structural_sources(root: Path, failures: list[str]) -> None:
    required = {
        "structural-guards/p936/authority/authority_guard.py": (
            "class AuthorityStore", "resolve_codebook_binding", "build_protected_index",
            "assert_reclaim_allowed", "assert_seal_dependencies",
        ),
        "structural-guards/p953/immutable_sha_authority.py": (
            "class ImmutableSHAIndex", "bind_stage_specs", "validate_inherited_prefix", "resume_layer_plan",
        ),
    }
    for relative, needles in required.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing structural source {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"structural source {relative} lacks {needle}")


def validate_privacy(root: Path, failures: list[str]) -> None:
    private_names = ("d" + "nola", "mac" + "mini", "Da" + "vid")
    forbidden = re.compile(
        r"/(?:Users|home)/|spark[-_]?[0-9]+|t_[0-9a-f]{8}|"
        + r"\b(?:" + "|".join(private_names) + r")\b|"
        + r"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
        r"192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
        r"100\.(?:6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.(?:[0-9]{1,3}\.)[0-9]{1,3})(?![0-9])",
        re.I,
    )
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if forbidden.search(text):
            failures.append(f"privacy token in {path.relative_to(root).as_posix()}")


def validate_publication_safety(root: Path, failures: list[str]) -> None:
    try:
        namespace = runpy.run_path(str(root / "code/publication_safety.py"))
        receipt = namespace["scan_package"](root)
    except Exception as exc:
        failures.append(f"strict publication-safety scan failed: {exc}")
        return
    expect(failures, "publication-safety status", receipt.get("status"), "PASS")
    expect(
        failures,
        "publication-safety privacy status",
        receipt.get("privacy_scan_status"),
        "PASS_NO_PRIVATE_OR_SECRET_MATERIAL",
    )
    expect(
        failures,
        "publication-safety tree status",
        receipt.get("tree_safety_status"),
        "PASS_STRICT_TEXT_REGULAR_CONTAINED_MANIFEST_CLOSED",
    )


def verify(root: Path) -> list[str]:
    failures: list[str] = []
    manifest = validate_manifest(root, failures)
    try:
        validate_p931(root, failures)
        validate_true_c(root, failures)
        validate_p963(root, failures)
        validate_protocols(root, failures)
        validate_comparison(root, failures)
        validate_runtime_closure(root, failures)
        validate_provenance(root, manifest, failures)
        validate_structural_sources(root, failures)
        validate_privacy(root, failures)
        validate_publication_safety(root, failures)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        failures.append(f"verification exception: {exc}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    failures = verify(root)
    if failures:
        print("WIRE_C_V2_PACKAGE_VERIFY_FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    manifest = load(root, "PACKAGE_MANIFEST.json")
    print(f"WIRE_C_V2_PACKAGE_VERIFY_PASS files={len(manifest['files'])}")


if __name__ == "__main__":
    main()
