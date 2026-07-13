#!/usr/bin/env python3
"""Uniform basic-ternary planes build (P2 ternary anchor). args: START_LAYER END_LAYER(excl)"""
import os, sys, torch
PILOT = os.path.expanduser("~/missions/VQ_W2_PILOT")
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.expanduser("~/missions/DS4_GPTQV2_PILOT"))
import vqw2_pilot as vp
import gptqv2_pilot as gp
gp.M = os.path.expanduser("~/missions/DS4_GPTQV2_PILOT")
DEV = "cuda"
OUT = os.path.expanduser("~/missions/TERN_TIER/planes")
os.makedirs(OUT, exist_ok=True)
LAYERS = list(range(int(sys.argv[1]), int(sys.argv[2])))
log = lambda *a: print(*a, flush=True)

gen = torch.Generator(device=DEV); gen.manual_seed(7)
for L in LAYERS:
    dst = f"{OUT}/tern_layer_{L:03d}.pt"
    if os.path.exists(dst + ".DONE"):
        log(f"L{L:03d} skip"); continue
    bl = gp.WtsBundle(L)
    out = {}
    for proj, get in (("13", bl.fused13), ("2", bl.down)):
        lut = None
        for rnd in range(2):
            us, ws = [], []
            for e in range(0, 256, 16):
                W, sb = get(e)
                u, wts = vp.sample_u(W, sb, lut, gen, 200_000)
                us.append(u); ws.append(wts); del W, sb
            lut = vp.fit_ternary(torch.cat(us), torch.cat(ws)); del us, ws
        codes_l, sc_l = [], []
        for e in range(256):
            W, sb = get(e)
            codes, sc, _ = vp.requant_lut(W, sb, lut, vp.T_OFFSETS)
            codes_l.append(codes.cpu()); sc_l.append(sc.cpu()); del W, sb
        out[f"codes{proj}"] = torch.stack(codes_l)
        out[f"sc{proj}"] = torch.stack(sc_l)
        out[f"lut{proj}"] = torch.tensor(lut)
        log(f"L{L:03d} proj{proj} lut={[round(float(x),4) for x in lut]}")
    torch.save(out, dst); open(dst + ".DONE", "w").write("ok")
    log(f"L{L:03d} saved")
open(os.path.expanduser(f"~/missions/TERN_TIER/TERN_BUILD_{LAYERS[0]}_{LAYERS[-1]}_DONE"), "w").write("ok")
log("HALF DONE")
