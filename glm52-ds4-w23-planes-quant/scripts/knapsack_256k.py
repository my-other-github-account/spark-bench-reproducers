#!/usr/bin/env python3
"""t_cf38c8c9 — R6 re-knapsack at 90 GiB (256K robustness budget).

Reconstructs the sealed t_29c4872c damage model from the two GPTQ solve
ledgers, VALIDATES by reproducing the sealed 88 GiB allocation
(counts 5854/5039/115, predicted KLD 0.150647, assignment identical),
then re-solves at budget_bytes = 90 GiB and emits R6_MANIFEST_256K.json.

Damage model (verbatim from R6_MANIFEST.json):
  linear: KLD_pred = sum_e c_tier(e)
  c_tier(e) = KLD_anchor(tier) * mass_e * relRMS_tier(e) / Z_tier
  mass_e    = n_fit + n_val (CALIB split routing mass)
  relRMS    = shipped-arm activation val error fused13+down
  14/22016 rtn_noval entries -> layer-median fill
Allocator: equal-cost greedy over nested upgrade steps
  (w2->w3 gain = c2-c3, w3->fp4 gain = c3; both steps 3,145,728 B).
"""
import json
import sys
from collections import defaultdict

GIB = 1 << 30
BYTES = {"w2": 7_077_888, "w3": 10_223_616, "fp4": 13_369_344}
STEP = 3_145_728
ANCHOR = {"w2": 0.311544, "w3": 0.072742, "fp4": 0.0}
N_WEIGHTS = 277.03e9  # bpw convention denominator (matches sealed rows)

SEALED_BUDGET = 94_489_280_512          # 88 GiB
NEW_BUDGET = 90 * GIB                    # 96,636,764,160
SEALED_COUNTS = {"w2": 5854, "w3": 5039, "fp4": 115}
SEALED_PRED = 0.150647


def load_ledger(path, arm):
    """arm in {'w2','w3v2'} -> {(layer,expert): [mass, errs, noval]}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["layer"], r["expert"])
        mass = r["n_fit"] + r["n_val"]
        errs = {}
        noval = []
        for unit in ("fused13", "down"):
            u = r[arm][unit]
            ship = u["ship"]
            v = u.get(f"val_{ship}")
            if v is None:
                noval.append(unit)
            errs[unit] = v
        out[key] = [mass, errs, noval]
    return out


def fill_noval(led, tag):
    """Layer-median fill for missing shipped-arm val errors."""
    filled = 0
    for unit in ("fused13", "down"):
        med = defaultdict(list)
        for (L, e), (m, errs, nv) in led.items():
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
    """c(e) = anchor * mass_e * relrms_e / Z ; relrms = fused13+down sum."""
    raw = {}
    for key, (m, errs, nv) in led.items():
        raw[key] = m * (errs["fused13"] + errs["down"])
    Z = sum(raw.values())
    return {k: anchor * v / Z for k, v in raw.items()}


def solve(c2, c3, budget):
    keys = sorted(c2.keys())
    nE = len(keys)
    base = nE * BYTES["w2"]
    nsteps = (budget - base) // STEP
    # nested steps: step1 (w2->w3) gain g1=c2-c3, step2 (w3->fp4) gain g2=c3.
    # Lazy greedy honoring prerequisite: heap of available steps.
    import heapq
    heap = []
    for k in keys:
        g1 = c2[k] - c3[k]
        heapq.heappush(heap, (-g1, 0, k))
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
    bytes_used = base + taken * STEP
    pred = sum(c2[k] if tier[k] == "w2" else (c3[k] if tier[k] == "w3" else 0.0)
               for k in keys)
    counts = {"w2": 0, "w3": 0, "fp4": 0}
    for k in keys:
        counts[tier[k]] += 1
    return tier, counts, bytes_used, pred, nsteps


def main():
    led2 = fill_noval(load_ledger("SOLVE_LEDGER.jsonl", "w2"), "w2")
    led3 = fill_noval(load_ledger("SOLVE_LEDGER_w3v2.jsonl", "w3v2"), "w3v2")
    assert set(led2) == set(led3), "ledger key mismatch"
    print(f"experts: {len(led2)}")
    c2 = coeffs(led2, ANCHOR["w2"])
    c3 = coeffs(led3, ANCHOR["w3"])
    # sanity like the sealed run: w2 damage > w3 damage everywhere?
    worse = sum(1 for k in c2 if c2[k] > c3[k])
    print(f"c2>c3 on {worse}/{len(c2)} experts")

    # ---- VALIDATION: reproduce sealed 88 GiB solve
    tier88, counts88, used88, pred88, ns88 = solve(c2, c3, SEALED_BUDGET)
    print(f"[88GiB] steps={ns88} counts={counts88} used={used88/GIB:.2f}GiB "
          f"pred={pred88:.6f} (sealed {SEALED_PRED}, counts {SEALED_COUNTS})")
    sealed = json.load(open("R6_MANIFEST.json"))
    sa = sealed["assignment"]
    mism = 0
    for L, row in sa.items():
        for e, t in row.items():
            if tier88[(int(L), int(e))] != t:
                mism += 1
    print(f"[88GiB] assignment mismatches vs sealed manifest: {mism}/11008")
    ok = (counts88 == SEALED_COUNTS and mism == 0
          and abs(pred88 - SEALED_PRED) < 5e-4)
    print(f"[88GiB] VALIDATION {'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    # ---- NEW SOLVE @ 90 GiB
    tier90, counts90, used90, pred90, ns90 = solve(c2, c3, NEW_BUDGET)
    bpw = used90 * 8 / N_WEIGHTS
    print(f"[90GiB] steps={ns90} counts={counts90} used={used90/GIB:.4f}GiB "
          f"pred={pred90:.6f} bpw={bpw:.4f}")
    # upgrades relative to sealed 88 manifest (for delta extraction)
    ups = []
    for k in sorted(tier90):
        if tier90[k] != tier88[k]:
            ups.append({"layer": k[0], "expert": k[1],
                        "from": tier88[k], "to": tier90[k]})
    print(f"delta vs sealed 88GiB manifest: {len(ups)} expert tier changes")
    from collections import Counter
    print(Counter((u['from'], u['to']) for u in ups))

    assignment = defaultdict(dict)
    for (L, e), t in sorted(tier90.items()):
        assignment[str(L)][str(e)] = t
    out = {
        "task": "t_cf38c8c9",
        "variant": "r6_dynamic_experts_256k",
        "parent_manifest": "R6_MANIFEST_E43.json (assignment == R6_MANIFEST.json, md5 461cddbd219e2ef50c6e22865cf3cae4)",
        "budget_bytes": NEW_BUDGET,
        "bytes_used": used90,
        "bpw": round(bpw, 4),
        "tiers": {
            "w2": {"planes_dir": "~/missions/DS4_GPTQ/planes_gptq_w2",
                   "bytes_per_expert": BYTES["w2"], "kld_anchor": ANCHOR["w2"]},
            "w3": {"planes_dir": "~/missions/DS4_R6/planes_w3v2_e43",
                   "bytes_per_expert": BYTES["w3"], "kld_anchor": ANCHOR["w3"],
                   "lut": [-6.5, -3.5, -1.875, -0.875, 0.140625, 1.5, 3.5, 6.5]},
            "fp4": {"planes_dir": None, "source": "ckpt e2m1 passthrough",
                    "bytes_per_expert": BYTES["fp4"], "kld_anchor": 0.0},
        },
        "damage_model": sealed["damage_model"],
        "predicted_kld": round(pred90, 6),
        "counts": counts90,
        "delta_vs_88gib": ups,
        "validation": {
            "sealed_88gib_reproduced": True,
            "sealed_counts": SEALED_COUNTS,
            "sealed_pred": SEALED_PRED,
            "repro_pred": round(pred88, 6),
            "assignment_mismatches": mism,
        },
        "assignment": {L: assignment[L] for L in sorted(assignment, key=int)},
    }
    with open("R6_MANIFEST_256K.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    import hashlib
    md5 = hashlib.md5(open("R6_MANIFEST_256K.json", "rb").read()).hexdigest()
    print(f"WROTE R6_MANIFEST_256K.json md5={md5}")


if __name__ == "__main__":
    main()
