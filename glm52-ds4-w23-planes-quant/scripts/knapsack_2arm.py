#!/usr/bin/env python3
"""t_cf38c8c9 SPEC UPDATE — re-knapsack at 94 GiB (primary, bpw~2.915) and
96 GiB (stretch, bpw~2.977) per Banana Bae's single-sequence-256K directive.

Validation chain before solving:
  1. reproduce sealed 88 GiB allocation exactly (counts 5854/5039/115,
     pred KLD 0.150647, 0 assignment mismatches vs R6_MANIFEST.json)
  2. reproduce the already-built 90 GiB solve (counts 5199/5666/143,
     pred 0.136918) vs R6_MANIFEST_256K.json

Emits R6_MANIFEST_94G.json and R6_MANIFEST_96G.json (same schema as the
256K/90G manifest; delta lists relative to the 90G manifest since those
planes are already staged on s7/s8).
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict

GIB = 1 << 30
BYTES = {"w2": 7_077_888, "w3": 10_223_616, "fp4": 13_369_344}
STEP = 3_145_728
ANCHOR = {"w2": 0.311544, "w3": 0.072742, "fp4": 0.0}
N_WEIGHTS = 277.03e9

SEALED_BUDGET = 94_489_280_512
SEALED_COUNTS = {"w2": 5854, "w3": 5039, "fp4": 115}
SEALED_PRED = 0.150647
B90 = 90 * GIB
B90_COUNTS = {"w2": 5199, "w3": 5666, "fp4": 143}
B90_PRED = 0.136918

ARMS = {"94G": 94 * GIB, "96G": 96 * GIB}


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
    print(f"[{tag}] layer-median filled {filled} unit entries")
    return led


def coeffs(led, anchor):
    raw = {k: m * (e["fused13"] + e["down"]) for k, (m, e) in led.items()}
    Z = sum(raw.values())
    return {k: anchor * v / Z for k, v in raw.items()}


def solve(c2, c3, budget):
    import heapq
    keys = sorted(c2.keys())
    base = len(keys) * BYTES["w2"]
    nsteps = (budget - base) // STEP
    heap = [(-(c2[k] - c3[k]), 0, k) for k in keys]
    heapq.heapify(heap)
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
    pred = sum(c2[k] if tier[k] == "w2" else (c3[k] if tier[k] == "w3" else 0.0)
               for k in keys)
    counts = Counter(tier.values())
    return tier, dict(counts), used, pred


def check(tag, tier, counts, pred, want_counts, want_pred, man_path):
    man = json.load(open(man_path))
    mism = sum(1 for L, row in man["assignment"].items()
               for e, t in row.items() if tier[(int(L), int(e))] != t)
    ok = (counts == want_counts and mism == 0
          and abs(pred - want_pred) < 5e-4)
    print(f"[{tag}] counts={counts} pred={pred:.6f} mism={mism} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    c2 = coeffs(fill_noval(load_ledger("SOLVE_LEDGER.jsonl", "w2"), "w2"),
                ANCHOR["w2"])
    c3 = coeffs(fill_noval(load_ledger("SOLVE_LEDGER_w3v2.jsonl", "w3v2"),
                           "w3v2"), ANCHOR["w3"])
    print(f"experts: {len(c2)}")

    t88, cnt88, u88, p88 = solve(c2, c3, SEALED_BUDGET)
    if not check("88G-seal", t88, cnt88, p88, SEALED_COUNTS, SEALED_PRED,
                 "R6_MANIFEST.json"):
        sys.exit(1)
    t90, cnt90, u90, p90 = solve(c2, c3, B90)
    if not check("90G-built", t90, cnt90, p90, B90_COUNTS, B90_PRED,
                 "R6_MANIFEST_256K.json"):
        sys.exit(1)

    sealed = json.load(open("R6_MANIFEST.json"))
    for tag, budget in ARMS.items():
        tier, counts, used, pred = solve(c2, c3, budget)
        bpw = used * 8 / N_WEIGHTS
        ups90 = [{"layer": k[0], "expert": k[1],
                  "from": t90[k], "to": tier[k]}
                 for k in sorted(tier) if tier[k] != t90[k]]
        # monotonicity check vs 90G (pure upgrades expected)
        order = {"w2": 0, "w3": 1, "fp4": 2}
        assert all(order[u["to"]] > order[u["from"]] for u in ups90), \
            f"{tag}: non-monotone tier change vs 90G"
        print(f"[{tag}] counts={counts} used={used/GIB:.4f}GiB "
              f"bpw={bpw:.4f} pred={pred:.6f} delta90={len(ups90)} "
              f"{Counter((u['from'], u['to']) for u in ups90)}")
        assignment = defaultdict(dict)
        for (L, e), t in sorted(tier.items()):
            assignment[str(L)][str(e)] = t
        out = {
            "task": "t_cf38c8c9",
            "variant": f"r6_dynamic_experts_256k_{tag.lower()}",
            "spec": "SPEC UPDATE Jul12: single-seq 256K, batch headroom "
                    "spent on bpw; 94G=primary, 96G=stretch",
            "parent_manifest": "R6_MANIFEST_E43.json (assignment == "
                               "R6_MANIFEST.json, md5 "
                               "461cddbd219e2ef50c6e22865cf3cae4)",
            "budget_bytes": budget,
            "bytes_used": used,
            "bpw": round(bpw, 4),
            "tiers": {
                "w2": {"planes_dir": "~/missions/DS4_GPTQ/planes_gptq_w2",
                       "bytes_per_expert": BYTES["w2"],
                       "kld_anchor": ANCHOR["w2"]},
                "w3": {"planes_dir":
                       "~/missions/DS4_R6/planes_w3v2_e43",
                       "bytes_per_expert": BYTES["w3"],
                       "kld_anchor": ANCHOR["w3"],
                       "lut": [-6.5, -3.5, -1.875, -0.875, 0.140625,
                               1.5, 3.5, 6.5]},
                "fp4": {"planes_dir": None,
                        "source": "ckpt e2m1 passthrough",
                        "bytes_per_expert": BYTES["fp4"], "kld_anchor": 0.0},
            },
            "damage_model": sealed["damage_model"],
            "predicted_kld": round(pred, 6),
            "counts": counts,
            "delta_vs_90gib": ups90,
            "validation": {
                "sealed_88gib_reproduced": True,
                "built_90gib_reproduced": True,
                "sealed_pred": SEALED_PRED,
                "b90_pred": B90_PRED,
            },
            "assignment": {L: assignment[L]
                           for L in sorted(assignment, key=int)},
        }
        fn = f"R6_MANIFEST_{tag}.json"
        with open(fn, "w") as f:
            json.dump(out, f, indent=1, sort_keys=False)
        md5 = hashlib.md5(open(fn, "rb").read()).hexdigest()
        print(f"WROTE {fn} md5={md5}")


if __name__ == "__main__":
    main()
