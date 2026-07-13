#!/usr/bin/env python3
"""t_3d6e422d card-step-3: R7 re-knapsack with IMPROVED anchors.

Damage model verbatim from R6_MANIFEST.json (t_29c4872c, validated
predicted-vs-measured within 2.1%):
  KLD_pred = sum_e c_tier(e);  c_tier(e) = anchor(tier)*mass_e*rel_e/Z_tier

SOLVER VALIDATION: reproduce the sealed 88 GiB solve from the OLD ledgers
+ OLD anchors (counts 5854/5039/115, pred 0.150647, 0 mismatches) before
any new solve — same bar as knapsack_256k.py (t_cf38c8c9).

NEW SOLVES (improved anchors):
  w2 tier: planes_gptq_w2v2 (W2v2-e43 grid + alpha + GPTQ/GPTAQ),
           anchor = measured R4_ds4flash_w2_gptq_v2 KLD (from ledger),
           rel_e  = shipped-arm val errs from SOLVE_LEDGER_w2v2.jsonl
  w3 tier: planes_w3v2_e43, anchor 0.072742,
           rel_e  = shipped-arm val errs from SOLVE_LEDGER_w3v2.jsonl
  fp4 tier: ckpt e2m1 passthrough, anchor 0.0
at budgets 88 GiB (compare vs sealed 0.147488/0.141492) and 94 GiB
(bpw 2.915, the 256K point per the t_cf38c8c9 spec update).

Emits R7_MANIFEST_88G.json + R7_MANIFEST_94G.json with predicted KLD
BEFORE building (R6 doctrine).
"""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

GIB = 1 << 30
BYTES = {"w2": 7_077_888, "w3": 10_223_616, "fp4": 13_369_344}
STEP = 3_145_728
N_WEIGHTS = 277.03e9
GPTQM = os.path.expanduser("~/missions/DS4_GPTQ")
BQ = os.path.expanduser("~/missions/DS4_BESTQ")
TEACH = os.path.expanduser("~/missions/DS4_TEACHER")

OLD_ANCHOR = {"w2": 0.311544, "w3": 0.072742}
SEALED_BUDGET = 94_489_280_512
SEALED_COUNTS = {"w2": 5854, "w3": 5039, "fp4": 115}
SEALED_PRED = 0.150647
BUDGETS = {"88G": SEALED_BUDGET, "94G": 94 * GIB}


def load_ledger(path, arm):
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["layer"], r["expert"])
        mass = r["n_fit"] + r["n_val"]
        errs = {}
        for unit in ("fused13", "down"):
            u = r[arm][unit] if isinstance(r[arm], dict) and \
                unit in r[arm] else None
            if u is None:
                errs[unit] = None
                continue
            ship = u.get("ship")
            v = u.get(f"val_{ship}")
            errs[unit] = v
        out[key] = [mass, errs]
    return out


def fill_noval(led, tag):
    filled = 0
    for unit in ("fused13", "down"):
        med = defaultdict(list)
        for (L, e), (m, errs) in led.items():
            if errs[unit] is not None:
                med[L].append(errs[unit])
        med = {L: sorted(v)[len(v) // 2] for L, v in med.items()}
        for (L, e), rec in led.items():
            if rec[1][unit] is None:
                rec[1][unit] = med[L]
                filled += 1
    print(f"[{tag}] layer-median filled {filled} unit entries")
    return led


def coeffs(led, anchor):
    raw = {k: m * (errs["fused13"] + errs["down"])
           for k, (m, errs) in led.items()}
    Z = sum(raw.values())
    return {k: anchor * v / Z for k, v in raw.items()}


def solve(c2, c3, budget):
    import heapq
    keys = sorted(c2.keys())
    base = len(keys) * BYTES["w2"]
    nsteps = (budget - base) // STEP
    heap = []
    for k in keys:
        heapq.heappush(heap, (-(c2[k] - c3[k]), 0, k))
    tier = {k: "w2" for k in keys}
    taken = 0
    while taken < nsteps and heap:
        negg, stage, k = heapq.heappop(heap)
        if stage == 0:
            tier[k] = "w3"
            heapq.heappush(heap, (-c3[k], 1, k))
        else:
            tier[k] = "fp4"
        taken += 1
    used = base + taken * STEP
    pred = sum(c2[k] if tier[k] == "w2"
               else (c3[k] if tier[k] == "w3" else 0.0) for k in keys)
    counts = Counter(tier.values())
    return tier, dict(counts), used, pred


def measured_kld(variant):
    for ln in open(f"{TEACH}/KLD_LEDGER.jsonl"):
        r = json.loads(ln)
        if r.get("variant") == variant:
            return r["kl_vs_fp8"]
    raise KeyError(variant)


def main():
    # ---- validation against the sealed solve (old ledgers, old anchors)
    led2_old = fill_noval(
        load_ledger(f"{GPTQM}/out/SOLVE_LEDGER.jsonl", "w2"), "w2-old")
    led3 = fill_noval(
        load_ledger(f"{GPTQM}/out/SOLVE_LEDGER_w3v2.jsonl", "w3v2"), "w3v2")
    c2o = coeffs(led2_old, OLD_ANCHOR["w2"])
    c3o = coeffs(led3, OLD_ANCHOR["w3"])
    tier88, counts88, used88, pred88 = solve(c2o, c3o, SEALED_BUDGET)
    sealed = json.load(open(f"{os.path.expanduser('~')}"
                            f"/missions/DS4_R6/R6_MANIFEST.json"))
    mism = 0
    for L, row in sealed["assignment"].items():
        for e, t in row.items():
            if tier88[(int(L), int(e))] != t:
                mism += 1
    ok = (counts88 == SEALED_COUNTS and mism == 0
          and abs(pred88 - SEALED_PRED) < 5e-4)
    print(f"[validate] counts={counts88} pred={pred88:.6f} mism={mism} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    # ---- new coefficients
    w2_anchor = measured_kld("R4_ds4flash_w2_gptq_v2")
    print(f"[anchors] w2(new)={w2_anchor} w3=0.072742 fp4=0.0")
    led2 = fill_noval(
        load_ledger(f"{BQ}/out/SOLVE_LEDGER_w2v2.jsonl", "w2v2"), "w2v2")
    assert set(led2) == set(led3), "ledger key mismatch"
    c2 = coeffs(led2, w2_anchor)
    c3 = coeffs(led3, OLD_ANCHOR["w3"])
    worse = sum(1 for k in c2 if c2[k] > c3[k])
    print(f"c2>c3 on {worse}/{len(c2)} experts")

    for tag, budget in BUDGETS.items():
        tier, counts, used, pred = solve(c2, c3, budget)
        bpw = used * 8 / N_WEIGHTS
        print(f"[{tag}] counts={counts} used={used/GIB:.4f}GiB "
              f"pred={pred:.6f} bpw={bpw:.4f}")
        assignment = defaultdict(dict)
        for (L, e), t in sorted(tier.items()):
            assignment[str(L)][str(e)] = t
        out = {
            "task": "t_3d6e422d",
            "variant": f"r7_bestq_{tag.lower()}",
            "budget_bytes": budget,
            "bytes_used": used,
            "bpw": round(bpw, 4),
            "tiers": {
                "w2": {"planes_dir":
                       "~/missions/DS4_BESTQ/planes_gptq_w2v2",
                       "bytes_per_expert": BYTES["w2"],
                       "kld_anchor": w2_anchor,
                       "lut": json.load(open(
                           f"{BQ}/planes_gptq_w2v2/layer_000.meta.json"
                       ))["lut"]},
                "w3": {"planes_dir":
                       "~/missions/DS4_R6/planes_w3v2_e43",
                       "bytes_per_expert": BYTES["w3"],
                       "kld_anchor": OLD_ANCHOR["w3"],
                       "lut": [-6.5, -3.5, -1.875, -0.875, 0.140625,
                               1.5, 3.5, 6.5]},
                "fp4": {"planes_dir": None,
                        "source": "ckpt e2m1 passthrough",
                        "bytes_per_expert": BYTES["fp4"],
                        "kld_anchor": 0.0},
            },
            "damage_model": sealed["damage_model"] +
                " | R7: w2 tier re-anchored to measured "
                "R4_ds4flash_w2_gptq_v2; rel_e from SOLVE_LEDGER_w2v2",
            "predicted_kld": round(pred, 6),
            "counts": counts,
            "validation": {
                "sealed_88gib_reproduced": True,
                "repro_pred": round(pred88, 6),
                "assignment_mismatches": 0},
            "assignment": {L: assignment[L]
                           for L in sorted(assignment, key=int)},
        }
        p = f"{BQ}/R7_MANIFEST_{tag}.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=1)
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        print(f"WROTE {p} md5={md5}")


if __name__ == "__main__":
    main()
