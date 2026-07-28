#!/usr/bin/env python3
"""P760 live-inventory solve: P693 config + current QTIP2/QTIP3/QTIP15 columns."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
P693_PATH = ROOT / "code" / "solve_p693_turbo.py"
Q15_DIR = ROOT / "inputs" / "qtip15"
SNAPSHOT = ROOT / "inputs" / "MENU_SNAPSHOT.json"
Q3_AUDIT = ROOT / "inputs" / "qtip3" / "P696_QTIP3_ARCHIVE_AUDIT.json"
OUT = ROOT / "out_p693"
Q15_TIER = "qtip15_1.509117"
Q2_LAYERS = (0, 2, 3, 4, 5, 6, 7, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 42)
Q3_LAYERS = tuple(range(3, 43))
Q15_LAYERS = (0, 4, 35)
Q2_PHYSICAL = {"fused13": 4_213_837, "down": 2_112_559}
Q2_LOGICAL = {"fused13": 4_210_692, "down": 2_109_444}
PRICE_LABEL = "P760_P637_QTIP2_FAMILY_PLUS_P693_QTIP3_PROJECTION_SSE_PLUS_QTIP15_EXACT_PER_UNIT_SSE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256(path)


def load_p693():
    spec = importlib.util.spec_from_file_location("p693_base", P693_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


p693 = load_p693()
p693.EXPECTED_Q3_LAYERS = Q3_LAYERS
p693.PRICE_LABEL = PRICE_LABEL
_original_import_base = p693.import_base


def import_base_patched():
    base = _original_import_base()
    physical = {int(layer): dict(row) for layer, row in base.QTIP_PHYSICAL_BYTES_BY_LAYER.items()}
    for layer in Q2_LAYERS:
        physical.setdefault(layer, dict(Q2_PHYSICAL))
    base.ELIGIBLE = Q2_LAYERS
    base.QTIP_PHYSICAL_BYTES_BY_LAYER = physical
    return base


p693.import_base = import_base_patched


def load_q3_audit() -> dict:
    d = json.loads(Q3_AUDIT.read_text())
    rows = sorted(d["layers"], key=lambda x: int(x["layer"]))
    layers = tuple(int(r["layer"]) for r in rows)
    if d.get("status") != "PASS" or layers != Q3_LAYERS:
        raise RuntimeError({"qtip3_audit_gate": d.get("status"), "layers": layers})
    if int(d.get("canonical_layer_count", -1)) != 40 or int(d.get("total_units", -1)) != 20_480:
        raise RuntimeError("qtip3 all40 closure drift")
    for row in rows:
        if int(row["unit_count"]) != 512 or int(row["logical_bytes"]) != p693.EXPECTED_Q3_LAYER_LOGICAL_BYTES:
            raise RuntimeError({"qtip3_layer_drift": row})
        if float(row["physical_artifact_bpw"]) > 3.0117 + 1e-12:
            raise RuntimeError({"qtip3_physical_cap": row})
    return {
        "eligible_layers": list(layers),
        "receipts": rows,
        "lineage_32_34_36": [r for r in rows if int(r["layer"]) in (32, 34, 36)],
        "logical_bytes_per_layer": p693.EXPECTED_Q3_LAYER_LOGICAL_BYTES,
        "logical_bytes_per_projection_unit": dict(p693.EXPECTED_Q3_BYTES),
        "archive_container_bytes_total": sum(int(r["archive"]["bytes"]) for r in rows),
        "archive_receipt_set_sha256": sha256(Q3_AUDIT),
        "tlut_sha256": "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19",
        "all40_aggregate_output_set_sha256": d["all40_aggregate_output_set_sha256"],
        "source_audit": {"path": str(Q3_AUDIT), "sha256": sha256(Q3_AUDIT)},
    }


p693.load_archive_receipts = load_q3_audit
_original_build_surface = p693.build_surface


def load_q15_selectors() -> tuple[dict[tuple[int, int, str], dict], list[dict]]:
    selectors: dict[tuple[int, int, str], dict] = {}
    receipts = []
    for path in sorted(Q15_DIR.glob("*.json")):
        d = json.loads(path.read_text())
        layer = int(d["layer"])
        rows = d["unit_selectors"]
        if not str(d.get("status", "")).startswith("PASS") or layer not in Q15_LAYERS or len(rows) != 512:
            raise RuntimeError({"bad_qtip15_manifest": str(path)})
        counts = Counter((str(r["identity"]["projection"]), int(r["K"])) for r in rows)
        expected = Counter({("fused13", 1): 128, ("fused13", 2): 128, ("down", 1): 128, ("down", 2): 128})
        if counts != expected:
            raise RuntimeError({"qtip15_allocation_drift": str(path), "counts": counts})
        for row in rows:
            ident = row["identity"]
            key = (layer, int(ident["expert"]), str(ident["projection"]))
            if key in selectors:
                raise RuntimeError({"duplicate_qtip15_cell": key})
            sse = float(row["sse_fp64"])
            if not math.isfinite(sse) or sse < 0:
                raise RuntimeError({"bad_qtip15_sse": key, "sse": sse})
            selectors[key] = dict(row)
        receipts.append({
            "layer": layer,
            "path": str(path),
            "sha256": sha256(path),
            "selector_count": len(rows),
            "physical_artifact_bytes": sum(int(r["artifact_bytes"]) for r in rows),
            "logical_bytes": sum(int(r["logical_bytes"]) for r in rows),
        })
    if tuple(sorted(r["layer"] for r in receipts)) != Q15_LAYERS or len(selectors) != 512 * len(Q15_LAYERS):
        raise RuntimeError({"qtip15_snapshot_closure": receipts, "cells": len(selectors)})
    return selectors, receipts


def build_surface_p760():
    surface = _original_build_surface()
    base, gs, opts = surface["base"], surface["gs"], surface["opts"]
    selectors, receipts = load_q15_selectors()
    ratios = []
    for key, row in selectors.items():
        if base.QTIP_TIER not in opts[key]:
            raise RuntimeError({"qtip15_without_qtip2_family_price": key})
        projection = key[2]
        selected_sse = float(row["sse_fp64"])
        if int(row["K"]) == 2:
            k2_sse = selected_sse
            ratio = 1.0
        else:
            extra = Q2_LOGICAL[projection] - int(row["logical_bytes"])
            rank = float(row["damage_reduction_per_extra_byte"])
            k2_sse = selected_sse - rank * extra
            if extra <= 0 or not math.isfinite(k2_sse) or k2_sse <= 0 or k2_sse > selected_sse + 1e-9:
                raise RuntimeError({"qtip15_k2_sse_reconstruction": key, "k1": selected_sse, "k2": k2_sse, "rank": rank, "extra": extra})
            ratio = selected_sse / k2_sse
        q2 = opts[key][base.QTIP_TIER]
        opts[key][Q15_TIER] = {
            "tier": Q15_TIER,
            "bytes": int(row["artifact_bytes"]),
            "costs": {c: max(0.0, float(q2["costs"][c]) * ratio) for c in gs.CLASSES},
            "pricing_basis": PRICE_LABEL,
            "reference_tier": base.QTIP_TIER,
            "selected_k": int(row["K"]),
            "selected_unit_sse_fp64": selected_sse,
            "same_cell_k2_sse_fp64": k2_sse,
            "selected_to_k2_sse_ratio": ratio,
            "source_artifact_sha256": row["artifact_sha256"],
            "source_done_sha256": row["done_sha256"],
        }
        ratios.append(ratio)
    surface["qtip15"] = {
        "tier": Q15_TIER,
        "eligible_layers": list(Q15_LAYERS),
        "selector_receipts": receipts,
        "selector_count": len(selectors),
        "pricing_basis": PRICE_LABEL,
        "pricing_law": "sealed P637 QTIP2 same-currency per-cell family price multiplied by exact selected-unit build SSE / reconstructed same-cell K2 build SSE; K2-selected ratio=1; clamp>=0",
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
    }
    snapshot = json.loads(SNAPSHOT.read_text())
    if snapshot["status"] != "PASS" or tuple(snapshot["qtip2_2.0117"]["layers"]) != Q2_LAYERS or tuple(snapshot["qtip3_3.0117"]["layers"]) != Q3_LAYERS or tuple(snapshot["qtip15_1.509117"]["layers"]) != Q15_LAYERS:
        raise RuntimeError("menu snapshot drift")
    surface["menu_snapshot"] = {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT), "payload": snapshot}
    surface["checks"].update({
        "p760_qtip2_live_layers_31": tuple(base.ELIGIBLE) == Q2_LAYERS,
        "p760_qtip3_all40": surface["archives"]["eligible_layers"] == list(Q3_LAYERS),
        "p760_qtip15_sealed_layers_3": surface["qtip15"]["eligible_layers"] == list(Q15_LAYERS),
        "p760_menu_snapshot_bound": surface["menu_snapshot"]["sha256"] == "2525376f5f5225f04cda6ad940f65156dfcf3c334a03c72fa3a1257f0c5eaa7c",
    })
    if not all(surface["checks"].values()):
        raise RuntimeError({"p760_surface_checks": surface["checks"]})
    return surface


p693.build_surface = build_surface_p760


def postprocess() -> None:
    surface = p693.build_surface()
    gs, opts, original = surface["gs"], surface["opts"], surface["original"]
    assignment_path = OUT / "ASSIGNMENT_QTIP2_QTIP3.json"
    adoc = json.loads(assignment_path.read_text())
    adoc["schema"] = "p760-qtip2-qtip3-qtip15-assignment-v1"
    adoc["measurement_label"] = PRICE_LABEL
    adoc["menu_snapshot_sha256"] = surface["menu_snapshot"]["sha256"]
    assignment_sha = atomic_json(assignment_path, adoc)
    selected = p693.load_assignment(assignment_path, gs)
    pred, exact_bytes = p693.summarize(gs, opts, original, selected)
    objective = math.fsum(pred.values()) / len(gs.CLASSES)
    result_path = OUT / "RESULT.json"
    result = json.loads(result_path.read_text())
    arm = result["arms"]["with_2bit_plus_3bit"]
    if abs(float(arm["objective"]) - objective) > 1e-12 or int(arm["exact_bytes"]) != exact_bytes:
        raise RuntimeError("postprocess solve reproduction drift")
    q2 = p693.rung_summary(gs, opts, original, selected, surface["base"].QTIP_TIER, Q2_LAYERS)
    q3 = p693.rung_summary(gs, opts, original, selected, p693.QTIP3_TIER, Q3_LAYERS)
    q15 = p693.rung_summary(gs, opts, original, selected, Q15_TIER, Q15_LAYERS)
    qkeys = {k for k, t in selected.items() if t in (surface["base"].QTIP_TIER, p693.QTIP3_TIER, Q15_TIER)}
    qdelta = sum(int(opts[k][selected[k]]["bytes"]) - int(opts[k][original[k]]["bytes"]) for k in qkeys)
    result.update({
        "schema": "p760-live-inventory-expanded-solve-v1",
        "status": "PASS_FEASIBLE_P760_LIVE_QTIP_INVENTORY",
        "measurement_label": PRICE_LABEL,
        "menu_snapshot": surface["menu_snapshot"],
        "qtip15_pricing": surface["qtip15"],
    })
    result["arms"]["with_2bit_plus_3bit"]["label"] = "with_all_live_qtip2_qtip3_qtip15"
    result["rungs"] = {"qtip2_2.0117": q2, "qtip3_3.0117": q3, Q15_TIER: q15}
    result["assignment_receipt_sha256"] = assignment_sha
    result["input_receipts"]["menu_snapshot"] = {"path": str(SNAPSHOT), "sha256": sha256(SNAPSHOT)}
    result["input_receipts"]["qtip15_manifests"] = surface["qtip15"]["selector_receipts"]
    result["bytes"]["all_qtip_net_delta"] = qdelta
    result["bytes"]["all_qtip_net_bytes_freed"] = -qdelta
    ordinary = exact_bytes - p693.ENVELOPE - qdelta
    result["bytes"]["ordinary_tier_net_delta"] = ordinary
    result["bytes"]["ordinary_tier_net_bytes_spent"] = max(0, ordinary)
    result["bytes"]["closure_qtip_plus_ordinary_equals_total"] = qdelta + ordinary == exact_bytes - p693.ENVELOPE
    result_sha = atomic_json(result_path, result)
    done_path = OUT / "DONE.json"
    done = json.loads(done_path.read_text())
    done.update({
        "schema": "p760-done-v1",
        "status": result["status"],
        "result_sha256": result_sha,
        "assignment_receipt_sha256": assignment_sha,
        "qtip2_cells": q2["selected_cells"],
        "qtip3_cells": q3["selected_cells"],
        "qtip15_cells": q15["selected_cells"],
        "menu_snapshot_sha256": sha256(SNAPSHOT),
    })
    done_sha = atomic_json(done_path, done)
    atomic_json(OUT / "PROGRESS.json", {**done, "status": "SOLVER_EXITED_P760_RESULT_READY", "done_sha256": done_sha})
    classes = result["arms"]["with_2bit_plus_3bit"]["prediction_by_class"]
    lines = [
        "# P760 live QTIP inventory solve",
        "",
        f"Status: **{result['status']}**",
        "",
        "| uniform-six | weighted512 | exact bytes | slack | qtip2 cells | qtip3 cells | qtip15 cells |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {objective:.12g} | {result['arms']['with_2bit_plus_3bit']['weighted512_global']:.12g} | {exact_bytes} | {p693.ENVELOPE-exact_bytes} | {q2['selected_cells']} | {q3['selected_cells']} | {q15['selected_cells']} |",
        "",
        "| class | predicted KLD | ceiling |",
        "|---|---:|---:|",
    ]
    for c in gs.CLASSES:
        lines.append(f"| {c} | {classes[c]:.12g} | {result['constraints']['per_class_hard_ceilings'][c]:.12g} |")
    lines += [
        "",
        f"- Inventory: QTIP2 {len(Q2_LAYERS)} layers; QTIP3 {len(Q3_LAYERS)} layers; QTIP15 {len(Q15_LAYERS)} layers.",
        f"- Integer: {result['solver']['integer_status']} / {result['solver']['time_limit_seconds']}s; strongest rigorous bound {result['solver']['strongest_rigorous_lower_bound']:.12g}; relative gap {result['solver']['relative_gap']:.12g}.",
        f"- Assignment: `{assignment_path}` SHA `{assignment_sha}`.",
        f"- Snapshot SHA: `{sha256(SNAPSHOT)}`.",
    ]
    (OUT / "STANDARD_TABLE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({**done, "done_sha256": done_sha, "table": str(OUT / "STANDARD_TABLE.md")}, sort_keys=True))


def main() -> int:
    if os.environ.get("P760_SANITY_ONLY") == "1":
        surface = p693.build_surface()
        receipt = {
            "schema": "p760-sanity-only-v1",
            "status": "PASS",
            "checks": surface["checks"],
            "menu_snapshot": surface["menu_snapshot"],
            "qtip15": surface["qtip15"],
            "qtip2_layers": list(Q2_LAYERS),
            "qtip3_layers": list(Q3_LAYERS),
            "qtip15_layers": list(Q15_LAYERS),
        }
        digest = atomic_json(OUT / "P760_SANITY_ONLY.json", receipt)
        print(json.dumps({"status": "PASS", "sanity_sha256": digest}, sort_keys=True))
        return 0
    rc = int(p693.main() or 0)
    if rc == 0:
        postprocess()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
