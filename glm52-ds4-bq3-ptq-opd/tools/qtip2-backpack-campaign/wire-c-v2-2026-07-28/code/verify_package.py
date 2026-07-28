#!/usr/bin/env python3
"""Fail-closed verification for the public Wire C V2 package."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SELF_FILES = {"PACKAGE_MANIFEST.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    doc = json.loads(MANIFEST.read_text())
    failures: list[str] = []
    rows = doc.get("files", [])
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        failures.append("duplicate path in package manifest")

    listed = set(paths)
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
    }
    extras = sorted(actual - listed - SELF_FILES)
    missing_from_tree = sorted(listed - actual)
    if extras:
        failures.append(f"unmanifested files: {extras}")
    if missing_from_tree:
        failures.append(f"manifested files missing from tree: {missing_from_tree}")

    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            continue
        if path.stat().st_size != int(row["public_copy_bytes"]):
            failures.append(f"byte-count mismatch {row['path']}")
        if sha256(path) != row["public_copy_sha256"]:
            failures.append(f"hash mismatch {row['path']}")

    for name, pin in doc["required_source_pins"].items():
        if not any(row["source_sha256"] == pin for row in rows):
            failures.append(f"missing source pin {name}")

    solve = doc["selected_solve"]
    if solve["envelope_bytes"] - solve["exact_logical_bytes"] != solve["slack_bytes"]:
        failures.append("selected-solve byte closure failed")
    if not solve["accounting_closure"]:
        failures.append("selected-solve accounting closure false")

    build = doc["physical_build"]
    if build["changed_cells_expected"] != build["changed_cells_built_and_read_back"]:
        failures.append("physical changed-cell closure failed")
    if build["changed_cells_expected"] + build["unchanged_cells_inherited"] != build["assignment_cells"]:
        failures.append("physical assignment-cell closure failed")

    true_c = doc["true_c_chain"]
    point = true_c["wire_c_r_measured_global_kld"] - true_c["p922_measured_substitution_penalty_kld"]
    if abs(point - true_c["mechanical_point_kld"]) > 1e-15:
        failures.append("true-C mechanical point closure failed")
    if true_c["status"] != "ESTIMATE_NOT_MEASUREMENT":
        failures.append("true-C estimate mislabeled")

    balanced_a = ROOT / "artifacts/BALANCED64_V1.json"
    balanced_s = ROOT / "specs/BALANCED64_V1.public.json"
    if balanced_a.is_file() and balanced_s.is_file() and balanced_a.read_bytes() != balanced_s.read_bytes():
        failures.append("BALANCED64 spec copies differ")
    measurement_spec = json.loads((ROOT / "specs/WIRE_C_V2_MEASUREMENT_SPEC.public.json").read_text())
    if measurement_spec["instrument"]["windows"] != 64 or measurement_spec["instrument"]["positions"] != 65536:
        failures.append("measurement spec geometry drift")
    if measurement_spec["assignment"]["changed_cells"] + measurement_spec["assignment"]["inherited_cells"] != measurement_spec["assignment"]["assignment_cells"]:
        failures.append("measurement spec assignment closure failed")

    p931_path = ROOT / "artifacts/P931_V3_FIRST_FEASIBLE.public.json"
    if not p931_path.is_file():
        failures.append("missing P931 projected first-feasible receipt")
    else:
        p931 = json.loads(p931_path.read_text())
        validity = p931.get("public_validity", {})
        if validity.get("status") != "PROJECTED_FIRST_FEASIBLE__FINAL_SCIP_PENDING":
            failures.append("P931 first feasible validity label drift")
        if validity.get("measured") is not False:
            failures.append("P931 first feasible mislabeled measured")
        if p931.get("exact_bytes") != 101346700411 or p931.get("envelope_slack_bytes") != 0:
            failures.append("P931 first feasible byte closure drift")
        pricing = p931.get("pricing_v3_surface", {})
        if pricing.get("p922_joined_rows") != 3803:
            failures.append("P931 P922 join-count drift")
        if pricing.get("p928_interaction_application") != "ALREADY_BAKED_INTO_THE_THREE_V3_GRID_ROWS__NOT_ADDED_TWICE":
            failures.append("P931 P928 application drift")
        if pricing.get("input_shas", {}).get("pricing_v3") != "c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0":
            failures.append("P931 P930 pricing pin drift")

    provenance = json.loads((ROOT / "artifacts/ARTIFACT_PROVENANCE.json").read_text())
    manifest_by_path = {row["path"]: row for row in rows}
    for row in provenance["artifacts"]:
        manifest_row = manifest_by_path.get(row["path"])
        if manifest_row is None:
            failures.append(f"provenance path absent from manifest {row['path']}")
            continue
        for key in ("source_sha256", "public_copy_sha256", "public_copy_bytes"):
            if row[key] != manifest_row[key]:
                failures.append(f"provenance mismatch {row['path']}.{key}")

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
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if forbidden.search(text):
            failures.append(f"privacy token in {path.relative_to(ROOT)}")

    if failures:
        print("WIRE_C_V2_PACKAGE_VERIFY_FAIL")
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"WIRE_C_V2_PACKAGE_VERIFY_PASS files={len(rows)}")


if __name__ == "__main__":
    main()
