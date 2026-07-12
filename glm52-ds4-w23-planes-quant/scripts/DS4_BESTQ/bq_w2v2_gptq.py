#!/usr/bin/env python3
"""t_3d6e422d card-step-2: W2v2-GPTQ solver (per-layer, env DS4_LAYER=N).

Clone of the sealed ds4_gptq_w3v2.py (t_26055bf3) re-aimed at the W2v2-e43
grid, with two deltas gated by the pilots:
  - alpha (PILOT_ALPHA.json): per-tensor fractional scale from the RTN
    build folded into s_col (held static for GPTQ, like the sc bytes).
  - GPTAQ (PILOT_GPTAQ.json): when adopted, the down proj gets a third arm
    (asymmetric-error W* solve, arXiv 2504.02692) and the down val gate
    runs on the TRUE objective ||Av~ Q^T - Av_fp W^T|| for all arms.

Everything else verbatim: blocksize 128, percdamp 0.01, colnorm-desc perm,
STATIC per-column scales, fused13 joint [4096,4096] solve w/ shared H13,
per-projection val-gated ship margin 2%, fragment-major 2-bit pack,
sc bytes byte-identical to the RTN build, all-RTN experts byte-identical.
Output: planes_gptq_w2v2/layer_NNN.* + SOLVE_LEDGER_w2v2.jsonl.
"""
import gc
import json
import os
import signal
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_BESTQ"))
import bq_common as bq  # noqa: E402
import planes_unpack as pu  # noqa: E402

V2 = f"{bq.BQ}/moe_w2_planes_v2e43"
OUTDIR = f"{bq.BQ}/planes_gptq_w2v2"
OUT = f"{bq.BQ}/out"
LEDGER = f"{OUT}/SOLVE_LEDGER_w2v2.jsonl"
LAYER = int(os.environ.get("DS4_LAYER", "0"))
VAL_MARGIN = 0.02
MIN_FIT_ROWS, MIN_VAL_ROWS = 64, 32
LIMIT = bq.swiglu_limit()
LUT = torch.tensor(bq.W2V2_LUT_E43, dtype=torch.float32, device=bq.DEV)

PILOT_A = json.load(open(f"{bq.BQ}/PILOT_ALPHA.json"))
ALPHA_MODE = PILOT_A["decisions"]["w2v2_alpha_mode"]
PILOT_G = json.load(open(f"{bq.BQ}/PILOT_GPTAQ.json"))
GPTAQ = PILOT_G["adopt"]

STOP = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: STOP.set())
signal.signal(signal.SIGINT, lambda *a: STOP.set())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] L{LAYER:03d} {m}", flush=True)


class V2Layer:
    def __init__(self, L):
        p = f"{V2}/layer_{L:03d}"
        self.p13 = np.load(f"{p}.planes13.npy", mmap_mode="r")
        self.p2 = np.load(f"{p}.planes2.npy", mmap_mode="r")
        self.s13 = np.load(f"{p}.sc13.npy", mmap_mode="r")
        self.s2 = np.load(f"{p}.sc2.npy", mmap_mode="r")
        ap = f"{p}.alphas.npy"
        self.alphas = np.load(ap) if os.path.exists(ap) else \
            np.ones((bq.E, 3), dtype=np.float32)

    def expert(self, e):
        r13 = torch.from_numpy(np.asarray(self.p13[e]))
        r2 = torch.from_numpy(np.asarray(self.p2[e]))
        rs13 = torch.from_numpy(np.asarray(self.s13[e]))
        rs2 = torch.from_numpy(np.asarray(self.s2[e]))
        c13 = pu.unpack_fragment_major(r13.to(bq.DEV), bq.N13, bq.K13)
        c2 = pu.unpack_fragment_major(r2.to(bq.DEV), bq.N2, bq.K2)
        sb13 = pu.unpack_scales(rs13, bq.N13, bq.K13 // 32)
        sb2 = pu.unpack_scales(rs2, bq.N2, bq.K2 // 32)
        return c13, sb13, c2, sb2, (r13, r2, rs13, rs2)

    def alpha_rows(self, e):
        a = self.alphas[e]
        a13 = torch.cat([
            torch.full((2048,), float(a[0])),
            torch.full((2048,), float(a[1]))]).to(bq.DEV)
        a2 = torch.full((bq.N2,), float(a[2])).to(bq.DEV)
        return a13, a2


def scol(sb, K, alpha_row):
    s = torch.exp2(sb.to(bq.DEV).float() - 127.0) \
        .repeat_interleave(32, dim=1).clamp_min(1e-38)
    return s * alpha_row.view(-1, 1)


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(f"{OUTDIR}/layer_{LAYER:03d}.meta.json"):
        log("meta exists, skip layer")
        return 0
    sel = bq.calib_selection()
    xf, hitf = bq.load_caps(LAYER, sel["fit_ids"])
    xv, hitv = bq.load_caps(LAYER, sel["val_ids"])
    log(f"caps loaded: fit {tuple(xf.shape)} val {tuple(xv.shape)} "
        f"alpha_mode={ALPHA_MODE} gptaq={GPTAQ}")
    v2 = V2Layer(LAYER)

    planes = {"planes13": np.zeros((bq.E, bq.N13 * bq.K13 // 4),
                                   dtype=np.uint8),
              "planes2": np.zeros((bq.E, bq.N2 * bq.K2 // 4),
                                  dtype=np.uint8),
              "sc13": np.zeros((bq.E, bq.N13 * bq.K13 // 32),
                               dtype=np.uint8),
              "sc2": np.zeros((bq.E, bq.N2 * bq.K2 // 32), dtype=np.uint8)}
    all_rtn = []

    for e in range(bq.E):
        if STOP.is_set():
            log(f"graceful stop at expert {e} (no plane files written)")
            return 1
        te = time.time()
        Xf = xf[hitf[:, e]].float()
        Xv = xv[hitv[:, e]].float()
        n_fit, n_val = int(Xf.shape[0]), int(Xv.shape[0])
        row = dict(unit=f"L{LAYER:03d}_e{e:03d}", layer=LAYER, expert=e,
                   n_fit=n_fit, n_val=n_val)
        W13 = bq.src_dense(LAYER, e, ("w1", "w3"))
        Wd = bq.src_dense(LAYER, e, ("w2",))
        c13_rtn, sb13, c2_rtn, sb2, raw = v2.expert(e)
        a13, a2 = v2.alpha_rows(e)
        s13_col = scol(sb13, bq.K13, a13)
        s2_col = scol(sb2, bq.K2, a2)

        ship = {}
        if n_fit < MIN_FIT_ROWS:
            ship = {"fused13": "rtn_floor", "down": "rtn_floor"}
            c13_ship, c2_ship = c13_rtn, c2_rtn
            row["w2v2"] = {"fallback": "fit_floor"}
        else:
            H13 = Xf.t() @ Xf
            perm13 = bq.weight_perm(W13)
            Hinv13 = bq.gptq_prepare(H13, perm13)
            del H13
            c13_g = bq.gptq_loop(W13, s13_col, Hinv13, perm13, LUT)
            del Hinv13
            dq_g = LUT[c13_g.long()] * s13_col
            dq_r = LUT[c13_rtn.long()] * s13_col
            pv_g = bq.proxy_err(Xv, dq_g, W13) if n_val >= MIN_VAL_ROWS \
                else None
            pv_r = bq.proxy_err(Xv, dq_r, W13) if n_val >= MIN_VAL_ROWS \
                else None
            if pv_g is None:
                dec13 = "rtn_noval"
            elif pv_g <= (1.0 - VAL_MARGIN) * pv_r:
                dec13 = "gptq"
            else:
                dec13 = "rtn"
            ship["fused13"] = dec13
            c13_ship = c13_g if dec13 == "gptq" else c13_rtn
            dq13_ship = dq_g if dec13 == "gptq" else dq_r
            gI, uI = dq13_ship[:2048], dq13_ship[2048:]
            Af = bq.act(Xf, gI, uI, LIMIT)
            Av = bq.act(Xv, gI, uI, LIMIT) if n_val else None
            row["w2v2"] = {"fused13": dict(
                val_gptq=pv_g, val_rtn=pv_r, ship=dec13)}
            del dq_g, dq_r, dq13_ship, gI, uI

            H2 = Af.t() @ Af
            perm2 = bq.weight_perm(Wd)
            Hinv2 = bq.gptq_prepare(H2, perm2)
            c2_g = bq.gptq_loop(Wd, s2_col, Hinv2, perm2, LUT)
            arms = {"gptq": c2_g, "rtn": c2_rtn}
            if GPTAQ:
                Af_fp = bq.act(Xf, W13[:2048], W13[2048:], LIMIT)
                Wstar = bq.damp_solve(
                    H2, (Af.t() @ Af_fp) @ Wd.t().float()).t()
                arms["gptaq"] = bq.gptq_loop(
                    Wstar, s2_col, Hinv2, perm2, LUT)
                del Wstar, Af_fp
            del H2, Hinv2

            if n_val >= MIN_VAL_ROWS:
                if GPTAQ:
                    Av_fp = bq.act(Xv, W13[:2048], W13[2048:], LIMIT)
                    ref = Av_fp @ Wd.t().float()
                    den = (ref.norm() + 1e-30).item()
                    errs = {}
                    for tag, cc in arms.items():
                        dq = LUT[cc.to(bq.DEV).long()] * s2_col
                        errs[tag] = ((Av @ dq.t().float()) - ref
                                     ).norm().item() / den
                    del Av_fp, ref
                else:
                    errs = {}
                    for tag, cc in arms.items():
                        dq = LUT[cc.to(bq.DEV).long()] * s2_col
                        errs[tag] = bq.proxy_err(Av, dq, Wd)
                cal = {t: v for t, v in errs.items() if t != "rtn"}
                best = min(cal, key=cal.get)
                if errs[best] <= (1.0 - VAL_MARGIN) * errs["rtn"]:
                    dec2 = best
                else:
                    dec2 = "rtn"
                row["w2v2"]["down"] = dict(
                    **{f"val_{t}": round(v, 6) for t, v in errs.items()},
                    ship=dec2, objective="true" if GPTAQ else "sym")
            else:
                dec2 = "rtn_noval"
                row["w2v2"]["down"] = dict(ship=dec2)
            ship["down"] = dec2
            c2_ship = arms.get(dec2, c2_rtn)
            del Af, Av, arms, c13_g, c2_g

        if all(v.startswith("rtn") for v in ship.values()):
            all_rtn.append(e)
            planes["planes13"][e] = raw[0].numpy()
            planes["planes2"][e] = raw[1].numpy()
        else:
            p13 = pu.pack_fragment_major(c13_ship.to(bq.DEV))
            p2 = pu.pack_fragment_major(c2_ship.to(bq.DEV))
            assert torch.equal(
                pu.unpack_fragment_major(p13, bq.N13, bq.K13),
                c13_ship.to(bq.DEV)), "rt13"
            assert torch.equal(
                pu.unpack_fragment_major(p2, bq.N2, bq.K2),
                c2_ship.to(bq.DEV)), "rt2"
            planes["planes13"][e] = p13.cpu().numpy()
            planes["planes2"][e] = p2.cpu().numpy()
            del p13, p2
        planes["sc13"][e] = raw[2].numpy()
        planes["sc2"][e] = raw[3].numpy()

        row["ship"] = ship
        row["secs"] = round(time.time() - te, 1)
        bq.jrow(LEDGER, **row)
        del Xf, Xv, W13, Wd, c13_rtn, c2_rtn, s13_col, s2_col, raw
        if e % 8 == 0:
            log(f"e{e}/{bq.E} n_fit={n_fit} n_val={n_val} "
                f"ship={json.dumps(ship)} {row['secs']}s")
            gc.collect()
            torch.cuda.empty_cache()
            with open(f"{OUT}/HEARTBEAT_SOLVE_W2V2", "w") as f:
                json.dump({"layer": LAYER, "expert": e,
                           "ts": time.time()}, f)

    del xf, xv, hitf, hitv
    gc.collect()
    torch.cuda.empty_cache()
    dst = f"{OUTDIR}/layer_{LAYER:03d}"
    for tag, ref in (("sc13", v2.s13), ("sc2", v2.s2)):
        assert np.array_equal(planes[tag], np.asarray(ref)), \
            f"{tag}: scale bytes differ from v2e43 planes"
    for tag, ref in (("planes13", v2.p13), ("planes2", v2.p2)):
        for e in all_rtn:
            assert np.array_equal(planes[tag][e], np.asarray(ref[e])), \
                f"{tag} e{e}: all-RTN expert differs from v2e43 planes"
    for tag in ("planes13", "sc13", "planes2", "sc2"):
        np.save(f"{dst}.{tag}.npy.tmp.npy", planes[tag])
        os.replace(f"{dst}.{tag}.npy.tmp.npy", f"{dst}.{tag}.npy")
    if os.path.exists(f"{V2}/layer_{LAYER:03d}.alphas.npy"):
        np.save(f"{dst}.alphas.npy.tmp.npy", v2.alphas)
        os.replace(f"{dst}.alphas.npy.tmp.npy", f"{dst}.alphas.npy")
    v2meta = json.load(open(f"{V2}/layer_{LAYER:03d}.meta.json"))
    meta = dict(E=bq.E, N13=bq.N13, K13=bq.K13, N2=bq.N2, K2=bq.K2,
                codebook="w2", bpw=2.25, lut=bq.W2V2_LUT_E43,
                lut_provenance=v2meta["lut_provenance"],
                scale_fit=v2meta["scale_fit"] + " (bytes verbatim from "
                          "moe_w2_planes_v2e43; held static for GPTQ)",
                alpha_mode=ALPHA_MODE, gptaq_adopted=GPTAQ,
                variant="gptq_w2v2", task="t_3d6e422d",
                calib="windows_ds4_calib.json "
                      "d09b006997b1843f041bf70c72ab695d",
                n_all_rtn_experts=len(all_rtn))
    tmp = f"{dst}.meta.json.tmp"
    json.dump(meta, open(tmp, "w"))
    os.replace(tmp, f"{dst}.meta.json")

    mins = round((time.time() - t0) / 60, 1)
    ships = []
    for ln in open(LEDGER):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("layer") == LAYER and isinstance(r.get("w2v2"), dict):
            for k in ("fused13", "down"):
                s = r["w2v2"].get(k, {})
                if isinstance(s, dict) and "ship" in s:
                    ships.append(s["ship"])
    summary = dict(unit=f"L{LAYER:03d}", layer=LAYER, minutes=mins,
                   n_gptq=ships.count("gptq"),
                   n_gptaq=ships.count("gptaq"),
                   n_rtn=ships.count("rtn") + ships.count("rtn_noval"),
                   md5_planes13=bq.md5f(f"{dst}.planes13.npy"),
                   md5_planes2=bq.md5f(f"{dst}.planes2.npy"),
                   alpha_mode=ALPHA_MODE, gptaq=GPTAQ)
    bq.jrow(f"{OUT}/SOLVE_DONE_w2v2.jsonl", **summary)
    log(f"DONE in {mins}m gptq={summary['n_gptq']} "
        f"gptaq={summary['n_gptaq']} rtn={summary['n_rtn']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
