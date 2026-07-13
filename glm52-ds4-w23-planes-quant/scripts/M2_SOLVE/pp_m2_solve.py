#!/usr/bin/env python3
"""M2-corrected per-projection knapsack solve (96G + 94G).

Derives from leverb_pp_alloc.py with ONE change: per-projection damage
coefficients are multiplied by the M2 shared-ratio anchor-correction
multipliers (fit sealed in ANCHOR_FIT_PATHA.json, validated on 94G-pp
0.4% and r7pp-88G 0.3%):
    a_f13 = 1.3114 (w2) / 1.3059 (w3)   a_down = 0.7491 (w2) / 0.7459 (w3)
Correction preserves each tier's total anchor (renormalized inside pcoeffs
scheme) -- we apply multipliers THEN renormalize the tier sum back to the
anchor so V1 decomposition still holds.

Outputs corrected predictions + manifests PP_M2_MANIFEST_{94G,96G}.json.
Also prints the uncorrected-manifest-under-corrected-model prediction so
the reallocation gain (solve improvement at same bytes) is explicit.
"""
import hashlib
import heapq
import json
from collections import Counter, defaultdict

GIB = 1 << 30
SRC = ("<orchestrator>/.hermes/kanban/boards/glm52-humming-w3/"
       "workspaces/t_cf38c8c9")
OUT = "<orchestrator>/clawd/glm5-humming-w3/W25_TIER_PILOT"

ANCHOR = {"w2": 0.311544, "w3": 0.072742}
N_WEIGHTS = 277.03e9
A = {  # M2 shared-ratio multipliers
    "w2": {"fused13": 1.3114, "down": 0.7491},
    "w3": {"fused13": 1.3059, "down": 0.7459},
}
BUDGETS = [("94g", 100_930_682_880), ("96g", 103_079_215_104)]


def load_ledger(path, arm):
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["layer"], r["expert"])
        mass = r["n_fit"] + r["n_val"]
        errs = {}
        for unit in ("fused13", "down"):
            u = r[arm][unit]
            errs[unit] = u.get(f"val_{u['ship']}")
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
    print(f"[{tag}] layer-median filled {filled}")
    return led


def pcoeffs_corrected(led, anchor, mult):
    """damage with M2 multipliers, renormalized so tier total == anchor."""
    Zc = sum(m * (errs["fused13"] * mult["fused13"]
                  + errs["down"] * mult["down"])
             for m, errs in led.values())
    return {k: {u: anchor * m * errs[u] * mult[u] / Zc
                for u in ("fused13", "down")}
            for k, (m, errs) in led.items()}


def solve_density(items, budget):
    heap = []
    for iid, c1, g1, c2, g2 in items:
        heapq.heappush(heap, (-g1 / c1, 0, iid, c1, g1, c2, g2))
    tier = {it[0]: 0 for it in items}
    spent = 0
    removed = 0.0
    skipped = 0
    while heap:
        d, stage, iid, c1, g1, c2, g2 = heapq.heappop(heap)
        cost = c1 if stage == 0 else c2
        gain = g1 if stage == 0 else g2
        if spent + cost > budget:
            skipped += 1
            if skipped > 4096:
                break
            continue
        spent += cost
        removed += gain
        tier[iid] = stage + 1
        if stage == 0:
            heapq.heappush(heap, (-g2 / c2, 1, iid, c1, g1, c2, g2))
    return tier, spent, removed


PBYTES = {
    "fused13": {"w2": 4_718_592, "w3": 6_815_744, "fp4": 8_912_896},
    "down":    {"w2": 2_359_296, "w3": 3_407_872, "fp4": 4_456_448},
}


def main():
    led2 = fill_noval(load_ledger(f"{SRC}/SOLVE_LEDGER.jsonl", "w2"), "w2")
    led3 = fill_noval(load_ledger(f"{SRC}/SOLVE_LEDGER_w3v2.jsonl", "w3v2"),
                      "w3v2")
    assert set(led2) == set(led3) and len(led2) == 11008
    keys = sorted(led2)
    c2p = pcoeffs_corrected(led2, ANCHOR["w2"], A["w2"])
    c3p = pcoeffs_corrected(led3, ANCHOR["w3"], A["w3"])
    tot2 = sum(v["fused13"] + v["down"] for v in c2p.values())
    print(f"V1 corrected: sum c2={tot2:.6f} (anchor {ANCHOR['w2']})")

    pitems = []
    for k in keys:
        for u in ("fused13", "down"):
            c1 = PBYTES[u]["w3"] - PBYTES[u]["w2"]
            c2_ = PBYTES[u]["fp4"] - PBYTES[u]["w3"]
            pitems.append(((k, u), c1, c2p[k][u] - c3p[k][u],
                           c2_, c3p[k][u]))
    base_p = sum(PBYTES[u]["w2"] for u in ("fused13", "down")) * len(keys)

    for tag, budget in BUDGETS:
        tier_p, spent_p, rem_p = solve_density(pitems, budget - base_p)
        pred_p = tot2 - rem_p
        bytes_p = base_p + spent_p
        tmap = {0: "w2", 1: "w3", 2: "fp4"}
        cu = {u: Counter() for u in ("fused13", "down")}
        for k in keys:
            cu["fused13"][tmap[tier_p[(k, "fused13")]]] += 1
            cu["down"][tmap[tier_p[(k, "down")]]] += 1
        bpw = bytes_p * 8 / N_WEIGHTS
        print(f"[{tag}] M2-CORRECTED SOLVE pred={pred_p:.6f} "
              f"bytes={bytes_p} bpw={bpw:.4f}")
        print(f"      f13 {dict(cu['fused13'])}  down {dict(cu['down'])}")

        # ALSO: evaluate the OLD (uncorrected) manifest under corrected model
        old = json.load(open(f"{OUT}/PP_MANIFEST_{tag.upper()}.json"))
        pred_old = 0.0
        for L, row in old["assignment"].items():
            for e, tt in row.items():
                k = (int(L), int(e))
                for u in ("fused13", "down"):
                    t = tt[u]
                    if t == "w2":
                        pred_old += c2p[k][u]
                    elif t == "w3":
                        pred_old += c3p[k][u]
        print(f"      old-manifest-under-corrected-model: {pred_old:.6f} "
              f"-> reallocation gain {100*(pred_old-pred_p)/pred_old:+.2f}%")

        assignment = defaultdict(dict)
        for (L, e) in keys:
            assignment[str(L)][str(e)] = {
                "fused13": tmap[tier_p[((L, e), "fused13")]],
                "down": tmap[tier_p[((L, e), "down")]]}
        man = {"variant": f"pp_m2_{tag}", "budget_bytes": budget,
               "bytes_used": bytes_p, "bpw": round(bpw, 4),
               "predicted_kld": round(pred_p, 6),
               "correction": "M2_shared_ratio", "multipliers": A,
               "counts_fused13": dict(cu["fused13"]),
               "counts_down": dict(cu["down"]),
               "assignment": {L: assignment[L]
                              for L in sorted(assignment, key=int)}}
        fn = f"{OUT}/PP_M2_MANIFEST_{tag.upper()}.json"
        with open(fn, "w") as f:
            json.dump(man, f)
        md5 = hashlib.md5(open(fn, "rb").read()).hexdigest()
        print(f"      wrote {fn} md5={md5[:10]}")


main()
