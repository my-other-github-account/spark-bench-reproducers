#!/usr/bin/env python3
"""Fail-closed verifier for the portable P672 p13 pipeline bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
P649_FINAL_SHA = "6714fb8df38d7eed2c7282caea691545ab2bdf51284d2fe8c40ceba7d8b5a398"
P649_PHYSICAL_SHA = "7b075170e405ad54b0487f6649923cba4abcaf8592eeaadfde942409b2270a9f"
P672_PHYSICAL_SHA = "6d86f2e7ac658d365adfe20f04502e0697bc97a0fff8c972abb947d98c2c0661"
FUSED_SHA = "5850caafaaba60502899da3ec713ed813a53505898cbeb410eef4e0a276e29d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-9) -> None:
    require(math.isfinite(actual), f"{label} is not finite")
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label} drift: {actual} != {expected}",
    )


def main() -> int:
    adoption = json.loads((ROOT / "ADOPTION.json").read_text())
    require(adoption.get("schema") == "p672-p13-pipeline-bundle-v1", "wrong ADOPTION schema")
    require(adoption.get("overall_pass") is True, "bundle is not PASS")

    actual: dict[str, str] = {}
    for relative, expected in adoption["bundle_files_sha256"].items():
        path = ROOT / relative
        require(path.is_file(), f"missing bundle file: {relative}")
        actual[relative] = sha256(path)
        require(actual[relative] == expected, f"SHA mismatch {relative}")
    present = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "ADOPTION.json"
        and "__pycache__" not in path.relative_to(ROOT).parts
    }
    require(present == set(actual), "manifest file-set mismatch")

    p649 = json.loads((ROOT / "receipts/P649_FINAL_GATE.json").read_text())
    profile = json.loads((ROOT / "receipts/INNER_PROFILE.json").read_text())
    smoke = json.loads((ROOT / "receipts/DECODE_SMOKE.json").read_text())
    ladder = json.loads((ROOT / "receipts/P13_PIPELINE_LADDER.json").read_text())
    backward = json.loads((ROOT / "receipts/REAL_ROUTED_BACKWARD_EXACT.json").read_text())
    raw_final = json.loads((ROOT / "receipts/FINAL_GATE_RAW_OVERLAP_EVENTS.json").read_text())
    final = json.loads((ROOT / "receipts/FINAL_GATE.json").read_text())
    p649_repro = json.loads((ROOT / "receipts/P649_REPRO_VERIFY.json").read_text())

    require(sha256(ROOT / "receipts/P649_FINAL_GATE.json") == P649_FINAL_SHA, "P649 gate SHA drift")
    require(p649.get("overall_pass") is True, "P649 gate failed")
    require(p649_repro.get("status") == "PASS", "P649 verifier reproduction failed")
    require(p649_repro.get("selected_grouping") == "r4_c1_p2_m64n32w8", "P649 grouping drift")
    close(float(p649_repro["routed_expert_seconds_after"]), 351.03980721416883, "P649 routed headline")

    integrity = profile.get("profiler_integrity", {})
    require(integrity.get("overall_pass") is True, "profiler integrity failed")
    require(integrity.get("projection_map") == {"2048": "2", "4096": "13"}, "projection map drift")
    require(integrity.get("signed_residual_preserved") is True, "signed residual missing")
    require(integrity.get("child_over_accounting_rejected") is True, "over-accounting rejection missing")
    require(integrity.get("expected_call_parity_enforced") is True, "profile call parity missing")
    require(integrity.get("method") == "deferred CUDA event pairs with one terminal sync per stage", "profiler method drift")
    full43_profile = profile["routed_expert_inner_profile"]["full43_scorer"]
    require(full43_profile.get("call_parity") is True, "profile full43 call parity failed")
    require(full43_profile.get("over_accounting_rejected") is True, "profile over-accounted")
    require(float(full43_profile.get("signed_residual_seconds", -1.0)) >= 0.0, "negative profile residual")
    require(len(full43_profile.get("top3", [])) == 3, "profile top3 incomplete")

    require(smoke.get("overall_pass") is True, "decoder smoke failed")
    require([row["bits"] for row in smoke["rows"]] == [8, 10, 11, 12, 16], "decoder width set drift")
    require(all(row.get("exact") and row.get("max_abs") == 0 for row in smoke["rows"]), "decoder mismatch")

    require(ladder.get("overall_pass") is True, "pipeline ladder failed")
    search = ladder.get("search_space", {})
    require(search.get("bounded_exhaustive_for_safe_p13_groups") is True, "ladder not bounded-exhaustive")
    require(search.get("rows_measured") == search.get("rows_expected") == 4, "ladder row count drift")
    require({row["p13_group"] for row in ladder["rows"][1:]} == {1, 2, 4}, "G ladder drift")
    require(all(row.get("all_exact") and row.get("floor_pass") for row in ladder["rows"]), "ineligible ladder row")
    selected = ladder["selected"]
    require(selected.get("name") == "p672_pipeline_g1", "G1 not selected")
    require(selected.get("p13_group") == 1, "selected group drift")
    require(float(selected.get("p13_payload_speedup_vs_p649", 0.0)) >= 1.5, "ladder target speedup failed")

    require(backward.get("overall_pass") is True, "routed backward/update gate failed")
    require(backward["output"].get("exact_equal") is True, "routed output mismatch")
    require(backward["input_gradient"].get("exact_equal") is True, "routed input-gradient mismatch")
    require(backward.get("all_parameter_gradients_at_decision_precision") is True, "routed parameter-gradient mismatch")
    require(backward.get("all_optimizer_updates_at_decision_precision") is True, "routed update mismatch")
    require(backward.get("all_finite") is True, "routed backward non-finite")
    require(backward.get("p649_reference_sha256") == P649_PHYSICAL_SHA, "backward P649 identity drift")
    require(backward.get("p672_candidate_sha256") == P672_PHYSICAL_SHA, "backward P672 identity drift")

    require(final.get("overall_pass") is True, "canonical full43 gate failed")
    require(raw_final.get("overall_pass") is False, "raw overlap-event receipt must preserve target-only fail")
    require(
        final.get("raw_overlap_event_receipt", {}).get("sha256")
        == sha256(ROOT / "receipts/FINAL_GATE_RAW_OVERLAP_EVENTS.json"),
        "raw overlap-event receipt identity drift",
    )
    identity = final.get("source_identity", {})
    require(identity.get("p649_final_gate_sha256") == P649_FINAL_SHA, "final P649 gate identity drift")
    require(identity.get("p649_physical_sha256") == P649_PHYSICAL_SHA, "final P649 source drift")
    require(identity.get("p672_physical_sha256") == P672_PHYSICAL_SHA, "final P672 source drift")
    require(identity.get("fused_expert_linear_sha256") == FUSED_SHA, "final fused source drift")
    target = final["target_sink"]
    before_target = float(target["before_seconds"]["combined_47_layers"])
    after_target = float(target["after_exposed_seconds"]["combined_47_layers"])
    close(float(target["speedup_combined_47_layers"]), before_target / after_target, "target speedup")
    require(float(target["speedup_combined_47_layers"]) >= 1.5 and target.get("pass") is True, "target speedup failed")
    routed = final["routed_total"]
    close(
        float(routed["speedup_combined_47_layers"]),
        float(routed["before_seconds"]["combined_47_layers"]) / float(routed["after_seconds"]["combined_47_layers"]),
        "routed speedup",
    )
    require(final["memory"].get("floor_pass") is True, "32-GiB floor failed")
    require(float(final["memory"]["minimum_available_gib"]) >= 32.0, "floor arithmetic failed")
    require(final["call_parity"].get("pass") is True, "canonical call parity failed")
    exact = final["exactness"]
    require(exact.get("cache_exact_16_of_16") is True, "cache mismatch")
    require(exact.get("cache_finite_16_of_16") is True, "cache non-finite")
    require(exact.get("scorer_exact_8_of_8") is True, "scorer mismatch")
    require(exact.get("scorer_decision_precision_8_of_8") is True, "scorer decision mismatch")
    require(exact.get("scorer_finite_8_of_8") is True, "scorer non-finite")
    require(exact.get("model_output_input_grad_parameter_grad_update_pass") is True, "model grad/update mismatch")
    require(exact.get("routed_output_input_grad_parameter_grad_update_pass") is True, "routed grad/update mismatch")
    require(exact.get("finite") is True, "canonical non-finite")

    runtime = adoption.get("runtime_env", {})
    require(runtime.get("P672_P13_PIPELINE") == "1", "pipeline env missing")
    require(runtime.get("P672_P13_GROUP") == "1", "G1 env missing")
    require(runtime.get("P649_EXPERT_RESIDENT_SCOPE") == "4", "P649 grouping drift")
    require(runtime.get("BANANA_SMASHER_REPAIR_MEM_FLOOR_BYTES") == str(32 * 1024**3), "memory floor env drift")

    roundtrip = {
        name: json.loads((ROOT / "roundtrip" / f"{name}.json").read_text())
        for name in (
            "STATUS_PREIMAGE", "APPLY", "APPLY_IDEMPOTENT", "STATUS_POSTIMAGE",
            "ROLLBACK", "ROLLBACK_IDEMPOTENT", "STATUS_FINAL",
        )
    }
    expected_status = {
        "STATUS_PREIMAGE": "P649_ROUTED_EXPERTS",
        "APPLY": "ADOPTED",
        "APPLY_IDEMPOTENT": "ALREADY_ADOPTED",
        "STATUS_POSTIMAGE": "P672_P13_PIPELINE",
        "ROLLBACK": "ROLLED_BACK",
        "ROLLBACK_IDEMPOTENT": "ALREADY_ROLLED_BACK",
        "STATUS_FINAL": "P649_ROUTED_EXPERTS",
    }
    require({name: row.get("status") for name, row in roundtrip.items()} == expected_status, "roundtrip status drift")
    require(
        roundtrip["STATUS_PREIMAGE"]["target_shas"]
        == roundtrip["STATUS_FINAL"]["target_shas"],
        "rollback did not restore P649 preimage",
    )
    require(
        roundtrip["STATUS_POSTIMAGE"]["target_shas"]["banana_smasher_physical_surface.py"]
        == P672_PHYSICAL_SHA,
        "apply postimage drift",
    )

    for name in ("ACTIVE_TARGET", "MIXED_TARGET", "TAMPERED_PAYLOAD"):
        negative = json.loads((ROOT / "negative" / f"{name}.json").read_text())
        require(negative.get("refused") is True, f"{name} was not refused")
        require(negative.get("pre_shas") == negative.get("post_shas"), f"{name} changed target bytes")

    print(json.dumps({
        "schema": "p672-p13-pipeline-bundle-verification-v1",
        "status": "PASS",
        "verified_files": len(actual),
        "selected_p13_group": 1,
        "target_seconds_before": before_target,
        "target_seconds_after": after_target,
        "target_speedup": target["speedup_combined_47_layers"],
        "routed_seconds_before": routed["before_seconds"]["combined_47_layers"],
        "routed_seconds_after": routed["after_seconds"]["combined_47_layers"],
        "full43_wall_seconds": final["wall"]["after_seconds"]["full43_scorer"],
        "minimum_available_gib": final["memory"]["minimum_available_gib"],
        "minutes_saved_per_24_updates": final["projected_24_update_dose"]["incremental_minutes_saved_vs_p649"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"P672_VERIFY_ERROR: {error}", file=sys.stderr)
        raise
