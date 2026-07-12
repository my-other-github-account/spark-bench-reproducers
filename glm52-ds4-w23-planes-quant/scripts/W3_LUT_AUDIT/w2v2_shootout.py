#!/usr/bin/env python3
"""CPU-only W2 v2 four-arm weight-space shootout for t_bd7728ee."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("W2V2_THREADS", "4")))
ROOT = Path.home() / "missions/W3_LUT_AUDIT"
sys.path.insert(0, str(ROOT))
import w3_lut_shootout as base

LAYERS = [0, 21, 42]
EVAL_E = [9, 50, 100, 150, 200, 254]
FIT_E = [17, 77, 177]
MATS = ["fused13", "down"]
CURRENT = base.W2_LUT
OUT = ROOT / "W2V2_SHOOTOUT.json"


def sym2_from_hist(hist):
    vals, mass = hist
    av = vals.abs()
    uniq, inv = torch.unique(av, return_inverse=True)
    merged = torch.zeros_like(uniq)
    merged.scatter_add_(0, inv, mass)
    pos = base.dp_lloyd(uniq, merged, 2).abs().sort().values
    # Enforce four ordered sign-symmetric nonzero levels.
    eps = torch.finfo(torch.float64).eps
    pos = torch.clamp(pos, min=eps)
    return torch.cat([-pos.flip(0), pos])


def collect_hist(experts, lut=None):
    hist = None
    for L in LAYERS:
        for e in experts:
            for mat in MATS:
                w, sb = base.load_matrix(L, e, mat)
                if lut is None:
                    h = base.u_hist(w, sb)
                else:
                    _, off = base.dequant_scmse(w, sb, lut)
                    s = torch.exp2(sb.double() - 127.0 + off.double())
                    h = base.u_hist(w, sb, s_override=s)
                hist = h if hist is None else base.merge_hist(hist, h)
                del w, sb
            print(f"fit L{L:03d} e{e:03d} {'hist' if lut is None else 'refit'}", flush=True)
    return hist


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "max": float(a.max()), "min": float(a.min()),
            "n": int(a.size)}


def main():
    t0 = time.time()
    h0 = collect_hist(FIT_E)
    asym1 = base.dp_lloyd(h0[0], h0[1], 4)
    sym1 = sym2_from_hist(h0)
    print("asym round1", asym1.tolist(), flush=True)
    print("sym round1", sym1.tolist(), flush=True)
    ha = collect_hist(FIT_E, asym1)
    hs = collect_hist(FIT_E, sym1)
    asym2 = base.dp_lloyd(ha[0], ha[1], 4)
    sym2 = sym2_from_hist(hs)
    print("asym round2", asym2.tolist(), flush=True)
    print("sym round2", sym2.tolist(), flush=True)

    arms = {k: {m: [] for m in MATS} for k in
            ["current_absmax", "current_sse", "dp_asym4_sse", "dp_sym4_sse"]}
    rows = []
    for L in LAYERS:
        for e in EVAL_E:
            for mat in MATS:
                w, sb = base.load_matrix(L, e, mat)
                dqs = {
                    "current_absmax": base.dequant_fixed(w, sb, CURRENT),
                    "current_sse": base.dequant_scmse(w, sb, CURRENT)[0],
                    "dp_asym4_sse": base.dequant_scmse(w, sb, asym2)[0],
                    "dp_sym4_sse": base.dequant_scmse(w, sb, sym2)[0],
                }
                row = {"layer": L, "expert": e, "matrix": mat}
                for name, dq in dqs.items():
                    v = base.relrms(dq, w)
                    arms[name][mat].append(v)
                    row[name] = v
                rows.append(row)
                del w, sb, dqs
            print(f"eval L{L:03d} e{e:03d} elapsed={time.time()-t0:.0f}s", flush=True)

    summary = {}
    for name, bymat in arms.items():
        summary[name] = {m: stats(v) for m, v in bymat.items()}
        both = bymat["fused13"] + bymat["down"]
        summary[name]["all"] = stats(both)
    baseline = summary["current_absmax"]["all"]["mean"]
    for name in summary:
        summary[name]["ratio_to_current"] = summary[name]["all"]["mean"] / baseline
    winner = min(summary, key=lambda x: summary[x]["all"]["mean"])
    result = {
        "task": "t_bd7728ee", "protocol": "3 layers x 6 eval experts x 2 matrices",
        "layers": LAYERS, "eval_experts": EVAL_E, "fit_experts": FIT_E,
        "luts": {"current": CURRENT.tolist(), "dp_asym4_round1": asym1.tolist(),
                 "dp_sym4_round1": sym1.tolist(), "dp_asym4_round2": asym2.tolist(),
                 "dp_sym4_round2": sym2.tolist()},
        "summary": summary, "winner": winner,
        "gate_pass": summary[winner]["ratio_to_current"] < 0.95,
        "rows": rows, "elapsed_s": time.time() - t0,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, OUT)
    print("=== SUMMARY ===")
    for name, s in summary.items():
        print(f"{name:18s} fused13={s['fused13']['mean']:.6f} down={s['down']['mean']:.6f} all={s['all']['mean']:.6f} ratio={s['ratio_to_current']:.4f}")
    print(f"winner={winner} gate_pass={result['gate_pass']} elapsed={result['elapsed_s']:.0f}s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
