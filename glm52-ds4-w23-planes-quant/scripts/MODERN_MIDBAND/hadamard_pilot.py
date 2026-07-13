#!/usr/bin/env python3
"""HADAMARD PILOT A: does incoherence processing improve our W3v2 dp-fit tier?
36 units, 5 pilot layers. Rotate W rows by a random-sign block-Hadamard (block 128),
refit dp-8-level LUT + SSE scales in rotated space, measure relRMS in ORIGINAL space
(rotate back). Pre-registered gate: >=10% mean relRMS improvement -> W3v2-H build.
"""
import json, os, sys, statistics
import torch
M = os.path.expanduser("~/missions/TERN_V2")   # wts staged here on s4
sys.path.insert(0, M)
import gptqv2_pilot as gp
import vqw2_pilot as vp
gp.M = M
DEV = "cuda"
LAYERS = [3, 13, 23, 33, 41]
FIT_E = [7, 63, 119, 175, 231, 254]
torch.manual_seed(3)

def hadamard(n, dev):
    H = torch.ones(1, 1, device=dev)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)

B = 128
Hb = None

def rot(W, sign):
    # per-128-block rotate along K (in_features): W [N, K]
    N, K = W.shape
    Wr = (W.reshape(N, K // B, B) * sign.reshape(1, K // B, B))
    return (Wr @ Hb).reshape(N, K)

def unrot(W, sign):
    N, K = W.shape
    Wr = (W.reshape(N, K // B, B) @ Hb.T)
    return (Wr * sign.reshape(1, K // B, B)).reshape(N, K)

V2LUT = [-6.38, -3.47, -1.87, -0.85, 0.14, 1.47, 3.48, 6.38]

def w3v2_fit(W, sb):
    codes, sc, _ = vp.requant_lut(W, sb, V2LUT, vp.T_OFFSETS)
    sf = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
    return torch.tensor(V2LUT, device=DEV)[codes.long()] * sf

res = {"plain": [], "hadamard": []}
Hb = hadamard(B, "cuda")
for L in LAYERS:
    bl = gp.WtsBundle(L)
    for e in FIT_E:
        for proj, get in (("f13", bl.fused13), ("down", bl.down)):
            W, sb = get(e)
            ref = W.norm()
            dq_p = w3v2_fit(W, sb)
            res["plain"].append(float((W - dq_p).norm() / ref))
            sign = (torch.randint(0, 2, (W.shape[1],), device=DEV) * 2 - 1).float()
            Wr = rot(W, sign)
            # proper rotated-space scales: exponent from Wr block absmax / lut max
            N, K = Wr.shape
            bm = Wr.abs().view(N, K // 32, 32).amax(2).clamp_min(1e-12)
            sbr = (torch.log2(bm / 6.38).ceil() + 127.0).clamp(0, 254).to(torch.uint8)
            dq_r = w3v2_fit(Wr, sbr)
            dq_back = unrot(dq_r, sign)
            res["hadamard"].append(float((W - dq_back).norm() / ref))
            del W
            torch.cuda.empty_cache()
    print(f"L{L:03d} done", flush=True)

out = {k: {"mean": statistics.mean(v), "n": len(v)} for k, v in res.items()}
out["ratio"] = out["hadamard"]["mean"] / out["plain"]["mean"]
json.dump(out, open(f"{M}/out/HADAMARD_PILOT.json", "w"), indent=1)
print(json.dumps(out, indent=1))
