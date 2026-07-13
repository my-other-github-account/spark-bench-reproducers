#!/usr/bin/env python3
"""FIRST 4-TIER SOLVE: add vqA (measured anchor 0.283803) to {W2, W3v2, FP4},
M2-corrected, per-projection, at 94G/96G. Reports menu-widening gain vs 3-tier."""
import json
from collections import Counter, defaultdict
import heapq

SRC = ("~/.hermes/kanban/boards/glm52-humming-w3/"
       "workspaces/t_cf38c8c9")
ANCHOR = {"w2": 0.311544, "w3": 0.072742, "vqa": 0.283803}
A = {"w2": {"fused13": 1.3114, "down": 0.7491},
     "w3": {"fused13": 1.3059, "down": 0.7459},
     "vqa": {"fused13": 1.3114, "down": 0.7491}}  # shared ratio, same as w2 class
PBYTES = {"fused13": {"w2": 4_718_592, "w3": 6_815_744, "fp4": 8_912_896},
          "down": {"w2": 2_359_296, "w3": 3_407_872, "fp4": 4_456_448}}
BUDGETS = [("94g", 100_930_682_880), ("96g", 103_079_215_104)]
PRIOR_3TIER = {"94g": 0.110466, "96g": 0.099772}


def load_scalar(path, arm):
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


def fill(led):
    for unit in ("fused13", "down"):
        med = defaultdict(list)
        for k, (m, e) in led.items():
            if e[unit] is not None:
                med[k[0]].append(e[unit])
        med = {L: sorted(v)[len(v) // 2] for L, v in med.items()}
        for k, rec in led.items():
            if rec[1][unit] is None:
                rec[1][unit] = med[k[0]]
    return led


led2 = fill(load_scalar(f"{SRC}/SOLVE_LEDGER.jsonl", "w2"))
led3 = fill(load_scalar(f"{SRC}/SOLVE_LEDGER_w3v2.jsonl", "w3v2"))

# vqA ledger: per (layer, expert, proj) rows; dedupe keep-last (full build after pilot)
vqa = {}
for line in open("/tmp/vqa_ledger.jsonl"):
    r = json.loads(line)
    proj = "fused13" if r["proj"] in ("fused13", "13") else "down"
    pv = r.get("pv_" + r["ship"]) or r.get("pv_hg")
    vqa[(r["layer"], r["expert"], proj)] = pv

keys = sorted(led2)
assert len(keys) == 11008
miss = sum(1 for k in keys for u in ("fused13", "down") if (k[0], k[1], u) not in vqa)
print(f"vqA coverage: missing {miss}/22016 units")

# damage coefficients, M2-corrected, renormalized per tier
def coeffs(errfn, anchor, mult):
    Z = sum(led2[k][0] * (errfn(k, "fused13") * mult["fused13"]
                          + errfn(k, "down") * mult["down"]) for k in keys)
    return {(k, u): anchor * led2[k][0] * errfn(k, u) * mult[u] / Z
            for k in keys for u in ("fused13", "down")}


c_w2 = coeffs(lambda k, u: led2[k][1][u], ANCHOR["w2"], A["w2"])
c_w3 = coeffs(lambda k, u: led3[k][1][u], ANCHOR["w3"], A["w3"])
lmed = defaultdict(list)
for (L, e, u), v in vqa.items():
    lmed[L].append(v)
lmed = {L: sorted(v)[len(v) // 2] for L, v in lmed.items()}
c_vq = coeffs(lambda k, u: vqa.get((k[0], k[1], u), lmed[k[0]]),
              ANCHOR["vqa"], A["vqa"])

for tag, budget in BUDGETS:
    # base = best 2.25bpw choice per unit (vqA vs W2, same bytes)
    items = []
    base_damage = 0.0
    base_bytes = 0
    basetier = {}
    for k in keys:
        for u in ("fused13", "down"):
            b = min(c_vq[(k, u)], c_w2[(k, u)])
            basetier[(k, u)] = "vqa" if c_vq[(k, u)] <= c_w2[(k, u)] else "w2"
            base_damage += b
            base_bytes += PBYTES[u]["w2"]
            c1 = PBYTES[u]["w3"] - PBYTES[u]["w2"]
            c2 = PBYTES[u]["fp4"] - PBYTES[u]["w3"]
            items.append(((k, u), c1, b - c_w3[(k, u)], c2, c_w3[(k, u)]))
    heap = []
    for iid, c1, g1, c2, g2 in items:
        heapq.heappush(heap, (-g1 / c1, 0, iid, c1, g1, c2, g2))
    tier = {}
    spent, removed, skipped = 0, 0.0, 0
    while heap:
        d, stage, iid, c1, g1, c2, g2 = heapq.heappop(heap)
        cost = c1 if stage == 0 else c2
        gain = g1 if stage == 0 else g2
        if spent + cost > budget - base_bytes:
            skipped += 1
            if skipped > 4096:
                break
            continue
        spent += cost
        removed += gain
        tier[iid] = stage + 1
        if stage == 0:
            heapq.heappush(heap, (-g2 / c2, 1, iid, c1, g1, c2, g2))
    pred = base_damage - removed
    cnt = Counter()
    for k in keys:
        for u in ("fused13", "down"):
            st = tier.get(((k, u)), 0) if isinstance(tier.get((k, u), 0), int) else 0
            st = tier.get((k, u), 0)
            cnt[basetier[(k, u)] if st == 0 else ("w3" if st == 1 else "fp4")] += 1
    prior = PRIOR_3TIER[tag]
    print(f"[{tag}] 4-TIER (+vqA) pred={pred:.6f} vs 3-tier {prior:.6f} "
          f"-> menu gain {100 * (prior - pred) / prior:+.2f}%")
    print(f"      tiers: {dict(cnt)}  bytes={base_bytes + spent}")
