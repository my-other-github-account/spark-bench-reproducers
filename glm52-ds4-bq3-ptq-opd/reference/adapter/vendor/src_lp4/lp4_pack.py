#!/usr/bin/env python3
"""Consolidated per-layer pack for the function-space
repair trainer.

Slices EXACTLY the assigned expert rows per (tier, projection) out of the
staged full-menu plane files (wire bytes untouched) into one
LP4_PACK/layer_NNN.pt per layer, so a training traversal reads ~96G
(shipped budget) instead of 187G and dequants batched.

Subcommands:
  slice   CPU-only row slicing (safe to run while the GPU is busy).
  verify  GPU equality check of the trainer's batched dequant vs the
          SEALED per-expert readers (v3.PlaneSource / VqaPlaneSource /
          TernaryPlaneSource / fp4 path) on sampled experts per layer.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE = os.path.expanduser(os.environ["BQ3_ASSET_ROOT"])
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

MANIFEST = os.path.expanduser(os.environ.get("BQ3_BASE_MANIFEST", os.path.join(BASE, "static", "LP4_MANIFEST.json")))
PACK = os.path.expanduser(os.environ.get("BQ3_PACK_DIR", os.path.join(BASE, "LP4_PACK")))
E, N13, K13, N2, K2 = 256, 4096, 4096, 4096, 2048
TIER_ID = {"w3": 0, "vqa": 1, "ternary": 2, "fp4": 3}


def tier_of(entry, which):
    if isinstance(entry, dict):
        return entry["fused13"] if which == "13" else entry["down"]
    return entry


def load_manifest():
    m = json.load(open(MANIFEST))
    return m


class FP4Ckpt:
    def __init__(self, ckpt_dir):
        self.dir = os.path.expanduser(ckpt_dir)
        self.wm = json.load(open(os.path.join(
            self.dir, "model.safetensors.index.json")))["weight_map"]
        self._h = {}

    def get(self, name):
        from safetensors import safe_open
        sh = self.wm[name]
        if sh not in self._h:
            if len(self._h) > 4:
                self._h.pop(next(iter(self._h)))
            self._h[sh] = safe_open(
                os.path.join(self.dir, sh), framework="pt")
        return self._h[sh].get_tensor(name)


def slice_layer(L, man, fp4c):
    amap = man["assignment"][str(L)]
    a13 = [tier_of(amap[str(e)], "13") for e in range(E)]
    a2 = [tier_of(amap[str(e)], "2") for e in range(E)]
    out = {
        "layer": L,
        "assign13": torch.tensor([TIER_ID[t] for t in a13], dtype=torch.int8),
        "assign2": torch.tensor([TIER_ID[t] for t in a2], dtype=torch.int8),
    }
    row13 = torch.full((E,), -1, dtype=torch.int16)
    row2 = torch.full((E,), -1, dtype=torch.int16)

    ids = {("w3", "13"): [], ("w3", "2"): [], ("vqa", "13"): [],
           ("vqa", "2"): [], ("ternary", "13"): [], ("ternary", "2"): [],
           ("fp4", "13"): [], ("fp4", "2"): []}
    for e in range(E):
        row13[e] = len(ids[(a13[e], "13")])
        ids[(a13[e], "13")].append(e)
        row2[e] = len(ids[(a2[e], "2")])
        ids[(a2[e], "2")].append(e)
    out["row13"], out["row2"] = row13, row2
    for (t, w), lst in ids.items():
        out[f"{t}_ids{w}"] = torch.tensor(lst, dtype=torch.int16)

    tiers = man["tiers"]

    # ---- w3 (npy wire planes, packed 384B blocks + packed scales)
    w3d = os.path.expanduser(tiers["w3"]["planes_dir"])
    meta = json.load(open(f"{w3d}/layer_{L:03d}.meta.json"))
    out["w3_lut"] = torch.tensor(meta["lut"], dtype=torch.float32)
    for w, (pk, sk) in (("13", ("planes13", "sc13")),
                        ("2", ("planes2", "sc2"))):
        sel = ids[("w3", w)]
        pl = np.load(f"{w3d}/layer_{L:03d}.{pk}.npy", mmap_mode="r")
        sc = np.load(f"{w3d}/layer_{L:03d}.{sk}.npy", mmap_mode="r")
        out[f"w3_pl{w}"] = torch.from_numpy(np.ascontiguousarray(pl[sel]))
        out[f"w3_sc{w}"] = torch.from_numpy(np.ascontiguousarray(sc[sel]))

    # ---- vqa (.pt codes/sc subset rows; codebooks FULL [256,4] both projs)
    vq = torch.load(f"{os.path.expanduser(tiers['vqa']['planes_dir'])}"
                    f"/vqa_layer_{L:03d}.pt", map_location="cpu")
    out["cb13"] = vq["cb13"].clone()
    out["cb2"] = vq["cb2"].clone()
    for w in ("13", "2"):
        sel = torch.tensor(ids[("vqa", w)], dtype=torch.long)
        out[f"vqa_codes{w}"] = vq[f"codes{w}"][sel].clone()
        out[f"vqa_sc{w}"] = vq[f"sc{w}"][sel].clone()

    # ---- ternary (already-subset .pt with expert_ids; remap rows)
    tpath = (f"{os.path.expanduser(tiers['ternary']['planes_dir'])}"
             f"/tern_layer_{L:03d}.pt")
    if os.path.exists(tpath) and (ids[("ternary", "13")]
                                  or ids[("ternary", "2")]):
        td = torch.load(tpath, map_location="cpu")
        for w in ("13", "2"):
            sel = ids[("ternary", w)]
            out[f"tern_lut{w}"] = td[f"lut{w}"].clone()
            if not sel:
                continue
            file_ids = td[f"expert_ids{w}"].tolist()
            rows = []
            for e in sel:
                assert e in file_ids, (L, w, e, file_ids)
                rows.append(file_ids.index(e))
            rows = torch.tensor(rows, dtype=torch.long)
            out[f"tern_codes{w}"] = td[f"codes{w}"][rows].clone()
            out[f"tern_sc{w}"] = td[f"sc{w}"][rows].clone()
    else:
        assert not ids[("ternary", "13")] and not ids[("ternary", "2")], \
            (L, "ternary assigned but no plane file")

    # ---- fp4 (ckpt mxfp4 bytes verbatim, w1 rows then w3 rows for 13)
    for w, names, kb in (("13", ("w1", "w3"), K13), ("2", ("w2",), K2)):
        sel = ids[("fp4", w)]
        if not sel:
            out[f"fp4_wb{w}"] = torch.zeros(0, dtype=torch.uint8)
            out[f"fp4_sb{w}"] = torch.zeros(0, dtype=torch.uint8)
            continue
        wbs, sbs = [], []
        for e in sel:
            wparts, sparts = [], []
            for nm in names:
                pre = f"layers.{L}.ffn.experts.{e}.{nm}"
                wparts.append(fp4c.get(pre + ".weight").view(torch.uint8))
                sparts.append(fp4c.get(pre + ".scale").view(torch.uint8))
            wbs.append(torch.cat(wparts, 0))
            sbs.append(torch.cat(sparts, 0))
        out[f"fp4_wb{w}"] = torch.stack(wbs)
        out[f"fp4_sb{w}"] = torch.stack(sbs)
    return out


def cmd_slice(args):
    man = load_manifest()
    fp4c = FP4Ckpt(man["ckpt_dir"])
    os.makedirs(PACK, exist_ok=True)
    todo = [int(x) for x in args.layers.split(",")] if args.layers \
        else list(range(43))
    for L in todo:
        dst = f"{PACK}/layer_{L:03d}.pt"
        if os.path.exists(dst) and not args.force:
            print(f"L{L:02d} exists, skip", flush=True)
            continue
        d = slice_layer(L, man, fp4c)
        torch.save(d, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        n = {t: len(d[f"{t}_ids13"]) for t in ("w3", "vqa", "ternary", "fp4")}
        print(f"L{L:02d} sliced {n} "
              f"({os.path.getsize(dst)/1e9:.2f} GB)", flush=True)


# ------------------------------------------------------- batched dequant
def unpack_w3_batched(plane, N, K):
    """[B, N*K*3//8] u8 -> [B, N, K] u8 codes. Batched pu.unpack_w3_plane."""
    B = plane.shape[0]
    nb, kb = N // 16, K // 64
    dev = plane.device
    p = plane.view(B, nb, kb, 384).to(torch.int64)
    low = p[..., :256].reshape(B, nb, kb, 32, 8)
    lo = torch.stack([(low >> (2 * i)) & 3 for i in range(4)], dim=-1)
    hib = p[..., 256:].reshape(B, nb, kb, 32, 4)
    hw = (hib[..., 0] | (hib[..., 1] << 8) | (hib[..., 2] << 16)
          | (hib[..., 3] << 24))
    shift = (torch.arange(8, device=dev).view(1, 1, 1, 1, 8, 1) * 4
             + torch.arange(4, device=dev).view(1, 1, 1, 1, 1, 4))
    hi = (hw.view(B, nb, kb, 32, 1, 1) >> shift) & 1
    codes = (lo | (hi << 2)).view(B, nb, kb, 8, 4, 2, 2, 2, 4)
    # sealed pu.unpack_w3_plane permute (0,4,2,1,5,6,3,7) shifted by the
    # leading batch dim -> (0,) + (j+1 for j in (0,4,2,1,5,6,3,7))
    codes = codes.permute(0, 1, 5, 3, 2, 6, 7, 4, 8).contiguous()
    return codes.view(B, N, K).to(torch.uint8)


def unpack_scales_batched(flat, N, KS):
    """[B, N*KS] u8 -> [B, N, KS] u8. Batched pu.unpack_scales."""
    B = flat.shape[0]
    return (flat.view(B, N // 16, KS, 16).permute(0, 1, 3, 2)
            .contiguous().view(B, N, KS))


def scol(sc_u8):
    return torch.exp2(sc_u8.float() - 127.0).repeat_interleave(32, dim=-1)


def deq_w3_batched(pl, scb, lut, N, K):
    codes = unpack_w3_batched(pl, N, K)
    sb = unpack_scales_batched(scb, N, K // 32)
    return (lut.float().to(pl.device)[codes.long()]
            * scol(sb)).to(torch.bfloat16)


def deq_vqa_batched(codes, sc, cb):
    w = cb.float().to(codes.device)[codes.long()]
    return (w.reshape(codes.shape[0], codes.shape[1], -1)
            * scol(sc)).to(torch.bfloat16)


def deq_tern_batched(codes, sc, lut):
    rep = codes.shape[-1] // sc.shape[-1]
    s = torch.exp2(sc.float() - 127.0).repeat_interleave(rep, dim=-1)
    return (lut.float().to(codes.device)[codes.long()]
            * s).to(torch.bfloat16)


def cmd_verify(args):
    import t8192_ds4_build_v3 as v3
    from bq_sources_revc import BestqManifestSource
    dev = "cuda"
    ms = BestqManifestSource(MANIFEST)
    todo = [int(x) for x in args.layers.split(",")] if args.layers \
        else list(range(43))
    rng = np.random.default_rng(0)
    for L in todo:
        d = torch.load(f"{PACK}/layer_{L:03d}.pt", map_location="cpu",
                       mmap=True)
        exp, _ = ms.layer(L)
        n_ok = 0
        for w, (N, K) in (("13", (N13, K13)), ("2", (N2, K2))):
            for t in ("w3", "vqa", "ternary", "fp4"):
                ids = d[f"{t}_ids{w}"]
                if len(ids) == 0:
                    continue
                pick = sorted(set(
                    rng.choice(len(ids), min(args.n, len(ids)),
                               replace=False).tolist()))
                rows = torch.tensor(pick)
                if t == "w3":
                    mine = deq_w3_batched(
                        d[f"w3_pl{w}"][rows].to(dev),
                        d[f"w3_sc{w}"][rows].to(dev), d["w3_lut"], N, K)
                elif t == "vqa":
                    mine = deq_vqa_batched(
                        d[f"vqa_codes{w}"][rows].to(dev),
                        d[f"vqa_sc{w}"][rows].to(dev), d[f"cb{w}"])
                elif t == "ternary":
                    mine = deq_tern_batched(
                        d[f"tern_codes{w}"][rows].to(dev),
                        d[f"tern_sc{w}"][rows].to(dev), d[f"tern_lut{w}"])
                else:
                    mine = v3.deq_fp4_block32(
                        d[f"fp4_wb{w}"][rows].to(dev),
                        d[f"fp4_sb{w}"][rows].to(dev), "e2m1")
                for j, ri in enumerate(pick):
                    e = int(ids[ri])
                    ref = exp(e, w)
                    if ref.dtype != torch.bfloat16:
                        ref = ref.to(torch.bfloat16)
                    assert torch.equal(mine[j], ref.to(dev)), \
                        f"L{L} {t}/{w} e{e} MISMATCH"
                    n_ok += 1
                del mine
        print(f"L{L:02d} verify PASS ({n_ok} experts exact)", flush=True)
        del d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("slice", "verify"))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=6, help="experts/tier to verify")
    a = ap.parse_args()
    (cmd_slice if a.cmd == "slice" else cmd_verify)(a)
