#!/usr/bin/env python3
"""Solve Q2-BIN and IQ3-BIN with k8192, corrected k4096, and k2048 VQ tiers.

The solver preserves the resident R8 greedy lower-hull recipe while adding two
more VQ3-family tiers. `vq3` is measured k8192/d4 at 3.5 bpw; `vq3b` is the
corrected k4096/d4 tier at 3.25 bpw, iso-byte with W3; `vq3c` is measured
k2048/d4 at 3.0 bpw. The k2048 tier reuses the measured k8192 per-layer ratio
profile because no k2048 layermap exists. All layer-shared codebook overheads
are charged because all three VQ tiers are in the menu.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import time

TASK = "t_1def6c8b"
ROOT = Path(os.environ.get("IQ4_ROOT", str(Path.home() / "missions/IQ4_BAR_RUN_t_1def6c8b")))
REF = ROOT / "solver_reference"
RIG = ROOT / "rig"
OUT = ROOT / "solve"
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE_SOLVER = REF / "reference_solver_DO_NOT_EXECUTE.py"
K8192_MAP = REF / "VQ3_MEASURED_LAYERMAP_ROWS.jsonl"
K4096_MAP_BASE = REF / "K4096_MEASURED_LAYERMAP_ROWS.jsonl"
K4096_LEDGER_FIXED = REF / "CORRECTED_K4096_VQ3_UNIFORM_LEDGER.jsonl"
K4096_RECEIPT = Path.home() / "missions/K4096_ANCHOR_RCA_t8885886e/CORRECTED_PLANE_RECEIPT.json"
K4096_ANCHOR = Path.home() / "missions/K4096_ANCHOR_RCA_t8885886e/out/K4096_CORRECTED_UNIFORM_MEASURED_ROW.json"
K8192_ANCHOR = REF / "K8192_UNIFORM_MEASURED_ROW.json"
K2048_ANCHOR = REF / "K2048_UNIFORM_MEASURED_ROW.json"
K2048_ANCHOR_MD5 = REF / "K2048_UNIFORM_MEASURED_ROW.json.md5"
K2048_SCORE = REF / "K2048_ANCHOR.SCORE.jsonl"
TEMPLATE = REF / "R8_MIXEDTIER_VQ3_MANIFEST_96G.json"

BUDGETS = [
    {"budget_id": "IQ3-BIN", "expert_gib": "94.4", "total_gb": "101.95",
     "under_reference": "1.05GB UNDER IQ3_XXS 103.0",
     "claim_shape": "smaller than IQ3_XXS, beats next rung"},
    {"budget_id": "Q2-BIN", "expert_gib": "88.2", "total_gb": "95.75",
     "under_reference": "1.05GB UNDER Q2_K_XL 96.8",
     "claim_shape": "smaller than Q2_K_XL, beats next rung"},
]
MENU = ["ternary", "vqa", "w2", "w3", "vq3", "vq3b", "vq3c", "fp4"]


def file_md5(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def gib_bytes(text: str) -> int:
    return int((Decimal(text) * Decimal(1 << 30)).to_integral_value(rounding=ROUND_HALF_UP))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def corrected_k4096_map() -> tuple[Path, dict[int, dict]]:
    rows = {int(r["layer"]): r for r in load_jsonl(K4096_MAP_BASE)}
    assert set(rows) == set(range(43))
    receipt = json.loads(K4096_RECEIPT.read_text())
    fixed = {int(r["layer"]): r for r in receipt["layers"]}
    assert set(fixed) == {0, 1}
    ledger = load_jsonl(K4096_LEDGER_FIXED)
    for layer in (0, 1):
        rec = dict(rows[layer])
        for proj in ("fused13", "down"):
            vals = [
                r for r in ledger
                if isinstance(r.get("layer"), int) and int(r["layer"]) == layer and r.get("proj") == proj
            ]
            assert len(vals) == 256, (layer, proj, len(vals))
            rec[f"{proj}_relrms_mean"] = sum(float(r["relrms_nn_fp16cb"]) for r in vals) / len(vals)
            rec[f"{proj}_w3v2_relrms_mean"] = sum(float(r["relrms_w3v2"]) for r in vals) / len(vals)
        rec["global_vq3_relrms"] = (
            2.0 * rec["fused13_relrms_mean"] + rec["down_relrms_mean"]
        ) / 3.0
        rec["global_w3v2_relrms"] = (
            2.0 * rec["fused13_w3v2_relrms_mean"] + rec["down_w3v2_relrms_mean"]
        ) / 3.0
        rec["pct_delta_vs_w3v2"] = 100.0 * (
            rec["global_vq3_relrms"] / rec["global_w3v2_relrms"] - 1.0
        )
        rec["plane_md5"] = fixed[layer]["canonical_md5"]
        rec["plane_bytes"] = int(fixed[layer]["bytes"])
        rec["plane_source"] = fixed[layer]["source_path"]
        rec["lane_source"] = "spark-5/corrected_build"
        rec["plane_meta_relrms_nn_mean_fp32cb"] = float(fixed[layer]["build_relrms_nn_mean"])
        rec["corrected_by_receipt"] = str(K4096_RECEIPT)
        rows[layer] = rec
    path = OUT / "K4096_CORRECTED_LAYERMAP_ROWS.jsonl"
    path.write_text("".join(json.dumps(rows[i], sort_keys=True) + "\n" for i in range(43)))
    return path, rows


def layer_ratios(path: Path) -> dict[int, dict]:
    result = {}
    for row in load_jsonl(path):
        layer = int(row["layer"])
        assert row.get("measured") is True and row.get("interpolated") is False, row
        assert layer not in result
        result[layer] = {
            "ratio_by_unit": {
                "fused13": float(row["fused13_relrms_mean"]) / float(row["fused13_w3v2_relrms_mean"]),
                "down": float(row["down_relrms_mean"]) / float(row["down_w3v2_relrms_mean"]),
            },
            "plane_md5": row["plane_md5"],
        }
    assert set(result) == set(range(43))
    return result


def lower_hull(points: list[tuple[int, float, str]]) -> list[tuple[int, float, str]]:
    points = sorted(points, key=lambda p: (p[0], p[1], p[2]))
    deduped = []
    for point in points:
        if deduped and deduped[-1][0] == point[0]:
            continue
        deduped.append(point)
    hull = [deduped[0]]
    for point in deduped[1:]:
        if point[1] >= hull[-1][1]:
            continue
        while len(hull) >= 2:
            b1, c1, _ = hull[-2]
            b2, c2, _ = hull[-1]
            if (c2 - c1) / (b2 - b1) >= (point[1] - c2) / (point[0] - b2):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def solve(r8, keys, led2, led3, ratios, normalizers, anchors_raw, budget_bytes):
    pbytes = {unit: dict(values) for unit, values in r8.PBYTES.items()}
    for unit in r8.NW:
        pbytes[unit]["vq3b"] = pbytes[unit]["w3"]
        # Exact 3.0/3.25 wire-bpw ratio. These unit sizes are divisible by 13.
        assert (pbytes[unit]["w3"] * 12) % 13 == 0
        pbytes[unit]["vq3c"] = pbytes[unit]["w3"] * 12 // 13
    items = []
    hull_presence = Counter()
    for key in keys:
        layer, _ = key
        m2, e2 = led2[key]
        m3, e3 = led3[key]
        for unit in ("fused13", "down"):
            c_w2 = r8.ANCHOR["w2"] * m2 * e2[unit] * r8.M2["w2"][unit] / normalizers["w2"]
            c_w3 = r8.ANCHOR["w3"] * m3 * e3[unit] * r8.M2["w3"][unit] / normalizers["w3"]
            damage = {
                "fp4": 0.0,
                "ternary": r8.MEAS_TERN * m2 * ratios["base"]["ternary"][unit] * e2[unit] * r8.M2["w2"][unit] / normalizers["ternary"],
                "vqa": r8.MEAS_VQA * m2 * ratios["base"]["vqa"][unit] * e2[unit] * r8.M2["w2"][unit] / normalizers["vqa"],
                "w2": c_w2,
                "w3": c_w3,
            }
            for tier in ("vq3", "vq3b", "vq3c"):
                damage[tier] = (
                    anchors_raw[tier] * m3 * e3[unit] * r8.M2["vq3"][unit]
                    * ratios[tier][layer]["ratio_by_unit"][unit] / normalizers[tier]
                )
            hull = lower_hull([(pbytes[unit][tier], damage[tier], tier) for tier in MENU])
            for _, _, tier in hull:
                hull_presence[tier] += 1
            steps = [(b[0] - a[0], a[1] - b[1], b[2]) for a, b in zip(hull, hull[1:])]
            items.append(((key, unit), hull[0][1], hull[0][2], steps))

    codebooks = {
        "vqa": r8.VQA_CODEBOOK_BYTES,
        "vq3": 43 * 2 * 8192 * 4 * 2,
        "vq3b": 43 * 2 * 4096 * 4 * 2,
        "vq3c": 43 * 2 * 2048 * 4 * 2,
    }
    overhead = sum(codebooks.values())
    base_bytes = sum(pbytes[iid[1]][tier0] for iid, _, tier0, _ in items)
    spend_cap = budget_bytes - base_bytes - overhead
    assert spend_cap >= 0
    heap = []
    for idx, (_, _, _, steps) in enumerate(items):
        if steps:
            cost, gain, _ = steps[0]
            assert cost > 0
            heapq.heappush(heap, (-gain / cost, idx, 0))
    spent = 0
    removed = 0.0
    skipped = 0
    tier_of = {iid: tier0 for iid, _, tier0, _ in items}
    while heap:
        _, idx, step_idx = heapq.heappop(heap)
        iid, _, _, steps = items[idx]
        cost, gain, tier = steps[step_idx]
        if spent + cost > spend_cap:
            skipped += 1
            if skipped > 8192:
                break
            continue
        spent += cost
        removed += gain
        tier_of[iid] = tier
        if step_idx + 1 < len(steps):
            next_cost, next_gain, _ = steps[step_idx + 1]
            heapq.heappush(heap, (-next_gain / next_cost, idx, step_idx + 1))
    bytes_used = base_bytes + spent + overhead
    counts = {unit: Counter(tier_of[(key, unit)] for key in keys) for unit in ("fused13", "down")}
    return {
        "raw_pred": sum(c0 for _, c0, _, _ in items) - removed,
        "bytes_used": bytes_used,
        "bpw": bytes_used * 8 / r8.N_WEIGHTS,
        "counts": counts,
        "tier_of": tier_of,
        "overhead": overhead,
        "codebook_bytes": codebooks,
        "pbytes": pbytes,
        "hull_presence": hull_presence,
    }


def main():
    required = [REFERENCE_SOLVER, K8192_MAP, K4096_MAP_BASE, K4096_LEDGER_FIXED,
                K4096_RECEIPT, K4096_ANCHOR, K8192_ANCHOR, K2048_ANCHOR,
                K2048_ANCHOR_MD5, K2048_SCORE, TEMPLATE,
                REF / "SOLVE_LEDGER.jsonl", REF / "SOLVE_LEDGER_w3v2.jsonl",
                REF / "VQW2_LEDGER.jsonl"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    receipt = json.loads(K4096_RECEIPT.read_text())
    for row in receipt["layers"]:
        path = Path(row["source_path"])
        assert file_md5(path) == row["canonical_md5"], (path, row)
    corrected_map_path, _ = corrected_k4096_map()

    spec = importlib.util.spec_from_file_location("r8ref", REFERENCE_SOLVER)
    assert spec is not None and spec.loader is not None
    r8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r8)
    r8.IN = REF

    led2, fill2 = r8.fill_noval(r8.load_ledger(REF / "SOLVE_LEDGER.jsonl", "w2"))
    led3, fill3 = r8.fill_noval(r8.load_ledger(REF / "SOLVE_LEDGER_w3v2.jsonl", "w3v2"))
    assert set(led2) == set(led3) and len(led2) == 43 * 256
    keys = sorted(led2)
    base_ratios = r8.load_ratios()
    ratios = {
        "base": base_ratios,
        "vq3": layer_ratios(K8192_MAP),
        "vq3b": layer_ratios(corrected_map_path),
    }
    # No k2048 layermap is available yet; use the same-family k8192 profile.
    ratios["vq3c"] = ratios["vq3"]
    normalizers = {
        "w2": sum(m * sum(e[u] * r8.M2["w2"][u] for u in r8.NW) for m, e in led2.values()),
        "w3": sum(m * sum(e[u] * r8.M2["w3"][u] for u in r8.NW) for m, e in led3.values()),
    }
    for name in base_ratios:
        normalizers[name] = sum(
            m * sum(base_ratios[name][u] * e[u] * r8.M2[name][u] for u in r8.NW)
            for m, e in led2.values()
        )
    for tier in ("vq3", "vq3b", "vq3c"):
        normalizers[tier] = sum(
            m * sum(e[u] * r8.M2["vq3"][u] * ratios[tier][layer]["ratio_by_unit"][u] for u in r8.NW)
            for (layer, _), (m, e) in led3.items()
        )

    k8192_anchor = json.loads(K8192_ANCHOR.read_text())
    k4096_anchor = json.loads(K4096_ANCHOR.read_text())
    k2048_anchor = json.loads(K2048_ANCHOR.read_text())
    assert k8192_anchor["measurement_status"] == "MEASURED"
    assert int(k8192_anchor["n_windows"]) == 512 and int(k8192_anchor["n_positions"]) == 524288
    assert k4096_anchor["measurement_status"] == "MEASURED"
    assert int(k4096_anchor["n_windows"]) == 512 and int(k4096_anchor["n_positions"]) == 524288
    assert k4096_anchor["corpus_md5"] == "1701920b4ba96dea0b18fe9df0151876"
    assert k2048_anchor["measurement_status"] == "MEASURED"
    assert int(k2048_anchor["n_windows"]) == 512 and int(k2048_anchor["n_positions"]) == 524288
    assert k2048_anchor["corpus_md5"] == "1701920b4ba96dea0b18fe9df0151876"
    assert int(k2048_anchor["k"]) == 2048 and int(k2048_anchor["d"]) == 4
    assert float(k2048_anchor["kl_vs_fp8"]) == 0.098564
    assert K2048_ANCHOR_MD5.read_text().split()[0] == file_md5(K2048_ANCHOR)
    calibration = float(json.loads(TEMPLATE.read_text())["lp1_measurement_calibration"])
    assert 0.98 < calibration < 0.99
    measured_anchors = {
        "vq3": float(k8192_anchor["kl_vs_fp8"]),
        "vq3b": float(k4096_anchor["kl_vs_fp8"]),
        "vq3c": float(k2048_anchor["kl_vs_fp8"]),
    }
    anchors_raw = {tier: value / calibration for tier, value in measured_anchors.items()}

    rows = []
    for budget in BUDGETS:
        budget_bytes = gib_bytes(budget["expert_gib"])
        solved = solve(r8, keys, led2, led3, ratios, normalizers, anchors_raw, budget_bytes)
        predicted = solved["raw_pred"] * calibration
        assignment = defaultdict(dict)
        for ((layer, expert), unit), tier in solved["tier_of"].items():
            assignment[str(layer)].setdefault(str(expert), {})[unit] = tier
        label = f"TRIPLEVQ_K2048MENU_{budget['budget_id'].replace('-', '_')}_{budget['expert_gib']}G_EXPERT_{budget['total_gb']}GB_TOTAL"
        manifest = {
            "task": TASK,
            "variant": label,
            "host": "spark-5",
            "phase": "K2048_IN_THREE_RUNG_VQ_MENU",
            "status": "PRED_SOLVED_FROM_THREE_MEASURED_VQ_ANCHORS",
            "measurement_label": "PRED",
            "anchor_measurement_label": "MEASURED",
            "budget_id": budget["budget_id"],
            "expert_gib": float(budget["expert_gib"]),
            "expert_budget_bytes": budget_bytes,
            "expert_bytes_used": solved["bytes_used"],
            "expert_bytes_used_gib": solved["bytes_used"] / (1 << 30),
            "bytes_under_cap": budget_bytes - solved["bytes_used"],
            "total_gb": float(budget["total_gb"]),
            "under_reference": budget["under_reference"],
            "claim_shape": budget["claim_shape"],
            "solve_objective_value_predicted_kld": predicted,
            "solve_objective_value_m2_raw": solved["raw_pred"],
            "bpw": solved["bpw"],
            "lp1_measurement_calibration": calibration,
            "menu": MENU,
            "tier_anchors": {
                "vq3": {"k": 8192, "d": 4, "wire_bpw": 3.5, "measured_kld": measured_anchors["vq3"],
                          "path": str(K8192_ANCHOR), "md5": file_md5(K8192_ANCHOR), "layermap": str(K8192_MAP), "layermap_md5": file_md5(K8192_MAP)},
                "vq3b": {"k": 4096, "d": 4, "wire_bpw": 3.25, "measured_kld": measured_anchors["vq3b"],
                           "path": str(K4096_ANCHOR), "md5": file_md5(K4096_ANCHOR), "layermap": str(corrected_map_path), "layermap_md5": file_md5(corrected_map_path),
                           "corrected_plane_receipt": str(K4096_RECEIPT), "corrected_plane_receipt_md5": file_md5(K4096_RECEIPT)},
                "vq3c": {"k": 2048, "d": 4, "wire_bpw": 3.0, "measured_kld": measured_anchors["vq3c"],
                           "path": str(K2048_ANCHOR), "md5": file_md5(K2048_ANCHOR),
                           "score_path": str(K2048_SCORE), "score_md5": file_md5(K2048_SCORE),
                           "ratio_source": "vq3_profile", "ratio_layermap": str(K8192_MAP),
                           "ratio_layermap_md5": file_md5(K8192_MAP)},
            },
            "vq3_codebook_bytes": solved["codebook_bytes"]["vq3"],
            "vq3b_codebook_bytes": solved["codebook_bytes"]["vq3b"],
            "vq3c_codebook_bytes": solved["codebook_bytes"]["vq3c"],
            "all_codebook_bytes": solved["codebook_bytes"],
            "overhead_bytes": solved["overhead"],
            "vq3_pbytes": solved["pbytes"],
            "counts_fused13": dict(solved["counts"]["fused13"]),
            "counts_down": dict(solved["counts"]["down"]),
            "hull_presence": dict(solved["hull_presence"]),
            "filled_missing_w2_fields": fill2,
            "filled_missing_w3_fields": fill3,
            "assignment": {layer: assignment[layer] for layer in sorted(assignment, key=int)},
            "created_unix": time.time(),
        }
        path = OUT / f"{label}_MANIFEST.json"
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        rows.append({
            "task": TASK, "row": label, "budget_id": budget["budget_id"],
            "measurement_label": "PRED", "expert_gib": float(budget["expert_gib"]),
            "total_gb": float(budget["total_gb"]), "predicted_kld": predicted,
            "bpw": solved["bpw"], "expert_bytes_used": solved["bytes_used"],
            "bytes_under_cap": budget_bytes - solved["bytes_used"],
            "counts_fused13": dict(solved["counts"]["fused13"]),
            "counts_down": dict(solved["counts"]["down"]),
            "manifest": str(path), "manifest_md5": file_md5(path),
        })

    summary = {
        "task": TASK,
        "status": "TRIPLE_VQ_TWO_BUDGETS_RESOLVED",
        "menu": MENU,
        "corrected_k4096_receipt_verified": True,
        "rows": rows,
        "created_unix": time.time(),
    }
    summary_path = OUT / "TRIPLE_VQ_K2048_TWO_BUDGET_SOLVE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "TRIPLE_VQ_K2048_TWO_BUDGET_SOLVE.COMPLETE").write_text(file_md5(summary_path) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
