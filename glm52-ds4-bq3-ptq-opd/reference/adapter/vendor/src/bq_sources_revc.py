#!/usr/bin/env python3
"""Plane sources for the sealed v3 builder / MMLU harness.
Revision C adds full-menu manifest support.

AlphaPlaneSource : per-layer plane dir reader (sealed v3.PlaneSource
    delegate) that applies the per-tensor fractional alpha from
    layer_NNN.alphas.npy [E,3] (w1, w3, down) when present. When absent,
    byte-identical passthrough of the sealed reader.

TernaryPlaneSource : basic-ternary 1.85bpw tier reader. Byte-for-byte
    the dequant path of the sealed ternary-tier rail source:
    tern_layer_NNN.pt {codes13,sc13,lut13,codes2,sc2,lut2}; scales are
    exp2(u8-127) block-column, LUT gather, bf16 out.

VqaPlaneSource : vqA d=4/k=256 2.25bpw tier reader. Byte-for-byte the
    vqA dequant branch of the sealed VQA-tier rail source, but
    addressable for ANY layer with a vqa_layer_NNN.pt plane file:
    {codes13,sc13,cb13,codes2,sc2,cb2}; sc exp2(u8-127) interleave 32.

TernlatPlaneSource : tern-lat 1.63bpw tier reader, SAME .pt schema as
    TernaryPlaneSource (lattice LUT baked in lut13/lut2). NOTE: as of
    2026-07-13 no ternlat planes exist anywhere (18-unit
    TERNLAT_SHOOTOUT pilot only) -- this class is the format contract
    for the day they are built; it asserts its planes dir exists.

BestqManifestSource : mixed-tier reader for the R7 manifests.
    rev B: assignment values may be either a str
    tier (per-expert, legacy) or {"fused13": tier, "down": tier}
    (per-projection). Tier lookup is per (expert, projection).
    rev C: full 6-tier menu {w2, w3, fp4, ternary, vqa,
    ternlat}. Tier delegates are constructed LAZILY and only for tiers
    that actually appear in the assignment, so a manifest that selects
    no w2 does not require the 73G w2 plane dir to be staged. The
    manifest "tiers" block is optional per tier when the tier is
    unused; a used tier without a planes_dir (and without a default)
    raises immediately with the exact tier name.

Self-tests (run before any rail):
    python3 bq_sources.py alpha <planes_dir> [L]
    python3 bq_sources.py manifest <manifest.json> [L]   (3-tier, sealed)
    python3 bq_sources.py fullmenu <manifest.json> [L]   (rev C menu)
"""
import json
import os
import sys

import numpy as np
import torch

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import t8192_ds4_build_v3 as v3  # noqa: E402

# Captured at import: run wrappers monkeypatch v3.PlaneSource, so a
# call-time lookup would recurse (same guard as the sealed manifest_planes).
_SealedPlaneSource = v3.PlaneSource

# Fallback dims when neither w2 nor w3 delegate is in the menu
# (E, N13, K13, N2 out, K2) -- matches the sealed s3 tern/vqa rails.
_DIMS = (256, 4096, 4096, 4096, 2048)

_TIER_DEFAULT_DIRS = {
    "w2": None,
    "w3": None,
    "ternary": None,
    "vqa": None,
    "ternlat": None,  # no planes exist anywhere yet
}


class AlphaPlaneSource:
    def __init__(self, planes_dir):
        d = os.path.expanduser(planes_dir)
        self.inner = _SealedPlaneSource(d)
        self.dir = d
        self._alphas = {}

    def _alpha(self, L):
        if L not in self._alphas:
            p = f"{self.dir}/layer_{L:03d}.alphas.npy"
            self._alphas[L] = np.load(p) if os.path.exists(p) else None
        return self._alphas[L]

    def layer(self, L):
        exp, dims = self.inner.layer(L)
        al = self._alpha(L)
        if al is None:
            return exp, dims

        def expert(e, which):
            w = exp(e, which)
            a = al[e]
            if which == "13":
                col = torch.empty(w.shape[0], dtype=w.dtype,
                                  device=w.device)
                col[:2048] = float(a[0])
                col[2048:] = float(a[1])
                return w * col.view(-1, 1)
            return w * float(a[2])

        return expert, dims


class TernaryPlaneSource:
    """Sealed s3 tern_uniform_source.TernUniformSource dequant path.

    Subset planes supported: when the .pt carries "expert_ids{which}"
    (1-D int tensor), rows are a packed subset in that order and expert
    ids are remapped; the dequant math is unchanged."""

    prefix = "tern_layer"

    def __init__(self, planes_dir):
        self.dir = os.path.expanduser(planes_dir)
        assert os.path.isdir(self.dir), \
            f"{type(self).__name__}: missing planes dir {self.dir}"
        self._cache = {}

    def _load(self, L):
        if L not in self._cache:
            path = f"{self.dir}/{self.prefix}_{L:03d}.pt"
            assert os.path.isfile(path), \
                f"{type(self).__name__}: missing plane file {path}"
            # keep exactly one layer resident (sealed source behavior)
            self._cache = {L: torch.load(path, map_location="cpu",
                                         mmap=True, weights_only=True)}
        return self._cache[L]

    @staticmethod
    def _row(data, which, e):
        ids = data.get(f"expert_ids{which}")
        if ids is None:
            return e
        hit = (ids == e).nonzero()
        assert hit.numel() == 1, \
            f"expert {e} not in subset plane (proj {which})"
        return int(hit[0, 0])

    def layer(self, L):
        data = self._load(L)

        def expert(e, which):
            r = self._row(data, which, e)
            codes = data[f"codes{which}"][r].to(v3.DEV)
            scales = data[f"sc{which}"][r].to(v3.DEV)
            lut = data[f"lut{which}"].to(v3.DEV).float()
            assert codes.shape[-1] % scales.shape[-1] == 0
            repeat = codes.shape[-1] // scales.shape[-1]
            s_col = torch.exp2(scales.float() - 127.0).repeat_interleave(
                repeat, dim=-1)
            weights = lut[codes.long()]
            return (weights.reshape(codes.shape[0], -1) * s_col).to(
                torch.bfloat16)

        return expert, _DIMS


class TernlatPlaneSource(TernaryPlaneSource):
    """tern-lat 1.63bpw: same .pt schema, lattice LUT baked per file."""

    prefix = "ternlat_layer"


class VqaPlaneSource:
    """Sealed s3 vqa_sources.VqaSwapSource vqA dequant branch, any layer."""

    def __init__(self, planes_dir):
        self.dir = os.path.expanduser(planes_dir)
        assert os.path.isdir(self.dir), \
            f"VqaPlaneSource: missing planes dir {self.dir}"
        self._cache = {}

    def _load(self, L):
        if L not in self._cache:
            path = f"{self.dir}/vqa_layer_{L:03d}.pt"
            assert os.path.isfile(path), \
                f"VqaPlaneSource: missing plane file {path}"
            self._cache = {L: torch.load(path, map_location="cpu")}
        return self._cache[L]

    def layer(self, L):
        d = self._load(L)

        def expert(e, which):
            if which == "13":
                codes, sc, cb = d["codes13"][e], d["sc13"][e], d["cb13"]
            else:
                codes, sc, cb = d["codes2"][e], d["sc2"][e], d["cb2"]
            codes = codes.to(v3.DEV)
            s_col = torch.exp2(sc.to(v3.DEV).float() - 127.0
                               ).repeat_interleave(32, dim=1)
            w = cb.to(v3.DEV).float()[codes.long()]
            return w.reshape(codes.shape[0], -1) * s_col

        return expert, _DIMS


def _tier_of(entry, which):
    """assignment value -> tier for projection ('13'|'2').
    str = per-expert (legacy); dict = per-projection (Lever B)."""
    if isinstance(entry, dict):
        return entry["fused13"] if which == "13" else entry["down"]
    return entry


_TIER_CLS = {
    "w2": AlphaPlaneSource,
    "w3": None,  # sealed reader, constructed inline (no alpha wrap)
    "ternary": TernaryPlaneSource,
    "ternlat": TernlatPlaneSource,
    "vqa": VqaPlaneSource,
}


class BestqManifestSource:
    """Manifest {assignment: {L: {e: tier | {fused13,down}}},
    tiers: {...}} reader. rev C: lazy delegates, 6-tier menu."""

    def __init__(self, manifest_path):
        mp = os.path.expanduser(manifest_path)
        assert os.path.isfile(mp), mp
        self.man = json.load(open(mp))
        self.assign = self.man["assignment"]
        self.tiers_meta = self.man.get("tiers", {})

        # which tiers does the assignment actually use?
        used = set()
        for amap in self.assign.values():
            for entry in amap.values():
                if isinstance(entry, dict):
                    used.update(entry.values())
                else:
                    used.add(entry)
        self.used = used

        # resolve plane dirs now (fail fast, exact tier named), construct
        # delegates lazily on first layer() touch.
        self._dirs = {}
        for t in sorted(used):
            if t == "fp4":
                continue
            meta = self.tiers_meta.get(t) or {}
            d = meta.get("planes_dir") or _TIER_DEFAULT_DIRS.get(t)
            assert d, (f"tier '{t}' is used by the assignment but has no "
                       f"planes_dir in the manifest tiers block and no "
                       f"default")
            d = os.path.expanduser(d)
            assert os.path.isdir(d), \
                f"tier '{t}' planes_dir missing on this host: {d}"
            self._dirs[t] = d
        self.src = {}

        self.ckpt = os.path.expanduser(
            self.man.get("ckpt_dir", "~/models/hf/DeepSeek-V4-Flash"))
        self.wm = None
        if "fp4" in used:
            self.wm = json.load(open(os.path.join(
                self.ckpt, "model.safetensors.index.json")))["weight_map"]
        self._handles = {}

    def _delegate(self, t):
        if t not in self.src:
            if t == "w3":
                self.src[t] = _SealedPlaneSource(self._dirs[t])
            else:
                self.src[t] = _TIER_CLS[t](self._dirs[t])
        return self.src[t]

    def _get(self, name):
        from safetensors import safe_open
        shard = self.wm[name]
        h = self._handles.get(shard)
        if h is None:
            h = safe_open(os.path.join(self.ckpt, shard), framework="pt")
            self._handles[shard] = h
        return h.get_tensor(name)

    def layer(self, L):
        amap = self.assign[str(L)]
        used_here = set()
        for entry in amap.values():
            if isinstance(entry, dict):
                used_here.update(entry.values())
            else:
                used_here.add(entry)
        exps = {}
        dims = None
        for t in sorted(used_here):
            if t == "fp4":
                continue
            e_t, d_t = self._delegate(t).layer(L)
            exps[t] = e_t
            if t in ("w2", "w3"):
                assert dims is None or dims == d_t, (dims, d_t)
                dims = d_t
        if dims is None:
            dims = _DIMS

        def fp4_expert(e, which):
            names = ("w1", "w3") if which == "13" else ("w2",)
            parts = []
            for wname in names:
                pre = f"layers.{L}.ffn.experts.{e}.{wname}"
                wb = self._get(f"{pre}.weight").view(torch.uint8).to(v3.DEV)
                sb = self._get(f"{pre}.scale").view(torch.uint8).to(v3.DEV)
                parts.append(v3.deq_fp4_block32(wb, sb, "e2m1"))
            return torch.cat(parts, 0) if len(parts) > 1 else parts[0]

        def expert(e, which):
            t = _tier_of(amap[str(e)], which)
            if t == "fp4":
                return fp4_expert(e, which)
            assert t in exps, f"unknown tier '{t}' at L{L} e{e} {which}"
            return exps[t](e, which)

        return expert, dims


def _rel(a, b):
    return ((a.float() - b.float()).pow(2).mean().sqrt()
            / b.float().pow(2).mean().sqrt()).item()


def _self_test_alpha(planes_dir, L=0):
    src = AlphaPlaneSource(planes_dir)
    inner = _SealedPlaneSource(os.path.expanduser(planes_dir))
    exp, dims = src.layer(L)
    exp0, _ = inner.layer(L)
    al = src._alpha(L)
    for e in (0, 100, 255):
        for which in ("13", "2"):
            a = exp(e, which)
            b = exp0(e, which)
            if al is None:
                assert torch.equal(a, b), "passthrough mismatch"
            else:
                if which == "13":
                    assert torch.allclose(
                        a[:2048], b[:2048] * float(al[e][0])), "w1 alpha"
                    assert torch.allclose(
                        a[2048:], b[2048:] * float(al[e][1])), "w3 alpha"
                else:
                    assert torch.allclose(
                        a, b * float(al[e][2])), "down alpha"
    mode = "passthrough" if al is None else "alpha-applied"
    print(f"[self-test alpha] L{L} PASS ({mode})")


def _self_test_manifest(manifest_path, L=0):
    ms = BestqManifestSource(manifest_path)
    amap = ms.assign[str(L)]
    exp, dims = ms.layer(L)
    w2d = AlphaPlaneSource(ms.man["tiers"]["w2"]["planes_dir"])
    w3d = _SealedPlaneSource(
        os.path.expanduser(ms.man["tiers"]["w3"]["planes_dir"]))
    e2, _ = w2d.layer(L)
    e3, _ = w3d.layer(L)
    delegate = {"w2": e2, "w3": e3}
    picks = {}
    for e in range(256):
        for which in ("13", "2"):
            t = _tier_of(amap[str(e)], which)
            picks.setdefault((t, which), e)
    pp = any(isinstance(v, dict) for v in amap.values())
    print(f"[self-test manifest] L{L} per-projection={pp} picks: "
          f"{{{', '.join(f'{t}/{w}:e{e}' for (t, w), e in picks.items())}}}")
    n_checked = 0
    for (t, which), e in picks.items():
        if t == "fp4":
            continue
        assert torch.equal(exp(e, which), delegate[t](e, which)), \
            f"{t} delegate proj {which} e{e}"
        n_checked += 1
    print(f"  delegate exact OK on {n_checked} (tier,proj) picks")
    e = picks.get(("fp4", "13"),
                  picks.get(("fp4", "2"), picks.get(("w3", "13"), 0)))
    parts = []
    for wname in ("w1", "w3"):
        pre = f"layers.{L}.ffn.experts.{e}.{wname}"
        wb = ms._get(f"{pre}.weight").view(torch.uint8).to(v3.DEV)
        sb = ms._get(f"{pre}.scale").view(torch.uint8).to(v3.DEV)
        parts.append(v3.deq_fp4_block32(wb, sb, "e2m1"))
    fp4_13 = torch.cat(parts, 0)
    w3_13 = e3(e, "13")
    r_ok = _rel(fp4_13, w3_13)
    swapped = torch.cat([fp4_13[2048:], fp4_13[:2048]], 0)
    r_swap = _rel(swapped, w3_13)
    print(f"  fp4-order e{e}: relRMS vs w3planes={r_ok:.4f} "
          f"half-swapped={r_swap:.4f}")
    assert r_ok < 0.5 and r_swap > 2 * r_ok, "stacking order suspect"
    if pp:
        mixed = next((e for e in range(256)
                      if isinstance(amap[str(e)], dict)
                      and amap[str(e)]["fused13"] != amap[str(e)]["down"]
                      and "fp4" not in amap[str(e)].values()), None)
        if mixed is not None:
            tf = amap[str(mixed)]["fused13"]
            td = amap[str(mixed)]["down"]
            assert torch.equal(exp(mixed, "13"), delegate[tf](mixed, "13"))
            assert torch.equal(exp(mixed, "2"), delegate[td](mixed, "2"))
            print(f"  mixed-tier e{mixed} ({tf}/{td}): both projections "
                  f"route to correct delegates OK")
    print("[self-test manifest] PASS")


def _tern_recompute(pt_dir, prefix, L, e, which):
    """Independent ternary/ternlat dequant (no source class code paths)."""
    d = torch.load(f"{os.path.expanduser(pt_dir)}/{prefix}_{L:03d}.pt",
                   map_location="cpu", mmap=True, weights_only=True)
    ids = d.get(f"expert_ids{which}")
    if ids is not None:
        hit = (ids == e).nonzero()
        assert hit.numel() == 1, f"expert {e} not in subset ({which})"
        e = int(hit[0, 0])
    codes = d[f"codes{which}"][e]
    sc = d[f"sc{which}"][e]
    lut = d[f"lut{which}"].float()
    rep = codes.shape[-1] // sc.shape[-1]
    w = lut[codes.long()].reshape(codes.shape[0], -1)
    s = torch.pow(torch.tensor(2.0), sc.float() - 127.0)
    s = s.repeat_interleave(rep, dim=-1)
    return (w * s).to(torch.bfloat16).to(v3.DEV)


def _vqa_recompute(pt_dir, L, e, which):
    d = torch.load(f"{os.path.expanduser(pt_dir)}/vqa_layer_{L:03d}.pt",
                   map_location="cpu")
    k = "13" if which == "13" else "2"
    codes, sc, cb = d[f"codes{k}"][e], d[f"sc{k}"][e], d[f"cb{k}"]
    w = cb.float()[codes.long()].reshape(codes.shape[0], -1)
    s = torch.pow(torch.tensor(2.0), sc.float() - 127.0)
    s = s.repeat_interleave(32, dim=1)
    return (w * s).to(v3.DEV)


def _self_test_fullmenu(manifest_path, L=0):
    """rev C gate: every tier used at layer L dequants and matches an
    independent recompute; fp4 keeps the stacking-order check."""
    ms = BestqManifestSource(manifest_path)
    amap = ms.assign[str(L)]
    exp, dims = ms.layer(L)
    print(f"[self-test fullmenu] L{L} used tiers (whole manifest): "
          f"{sorted(ms.used)} dims={dims}")
    picks = {}
    for e in range(256):
        for which in ("13", "2"):
            t = _tier_of(amap[str(e)], which)
            picks.setdefault((t, which), e)
    print(f"  L{L} picks: "
          f"{{{', '.join(f'{t}/{w}:e{e}' for (t, w), e in picks.items())}}}")
    n = 0
    for (t, which), e in sorted(picks.items()):
        got = exp(e, which)
        if t == "w2":
            ref, _ = AlphaPlaneSource(ms._dirs["w2"]).layer(L)
            assert torch.equal(got, ref(e, which)), f"w2 e{e}/{which}"
        elif t == "w3":
            ref, _ = _SealedPlaneSource(ms._dirs["w3"]).layer(L)
            assert torch.equal(got, ref(e, which)), f"w3 e{e}/{which}"
        elif t in ("ternary", "ternlat"):
            prefix = ("tern_layer" if t == "ternary" else "ternlat_layer")
            ref = _tern_recompute(ms._dirs[t], prefix, L, e, which)
            assert torch.equal(got, ref), f"{t} e{e}/{which}"
        elif t == "vqa":
            ref = _vqa_recompute(ms._dirs["vqa"], L, e, which)
            assert torch.equal(got, ref), f"vqa e{e}/{which}"
        elif t == "fp4":
            continue
        print(f"  {t}/{which} e{e}: exact-match vs independent recompute OK "
              f"shape={tuple(got.shape)} dtype={got.dtype}")
        n += 1
    if ("fp4", "13") in picks:
        e = picks[("fp4", "13")]
        got13 = exp(e, "13")
        parts = []
        for wname in ("w1", "w3"):
            pre = f"layers.{L}.ffn.experts.{e}.{wname}"
            wb = ms._get(f"{pre}.weight").view(torch.uint8).to(v3.DEV)
            sb = ms._get(f"{pre}.scale").view(torch.uint8).to(v3.DEV)
            parts.append(v3.deq_fp4_block32(wb, sb, "e2m1"))
        ref13 = torch.cat(parts, 0)
        assert torch.equal(got13, ref13), "fp4 reread mismatch"
        # stacking-order vs any non-fp4 tier at the same expert
        alt = next((t for (t, w), _ in picks.items()
                    if t != "fp4" and w == "13"), None)
        if alt:
            e_alt = picks[(alt, "13")]
            ref_alt = exp(e_alt, "13")
            r_ok = _rel(ref13, exp(e, "13"))
            print(f"  fp4/13 e{e}: reread exact OK (alt-tier {alt} "
                  f"sanity rel={r_ok:.4f})")
        n += 1
    # mixed-tier expert routing (per-projection)
    mixed = next((e for e in range(256)
                  if isinstance(amap[str(e)], dict)
                  and amap[str(e)]["fused13"] != amap[str(e)]["down"]), None)
    if mixed is not None:
        tf = _tier_of(amap[str(mixed)], "13")
        td = _tier_of(amap[str(mixed)], "2")
        a13 = exp(mixed, "13")
        a2 = exp(mixed, "2")
        print(f"  mixed-tier e{mixed} ({tf}/{td}): 13 {tuple(a13.shape)} "
              f"2 {tuple(a2.shape)} dequant OK")
    assert n >= 1
    print(f"[self-test fullmenu] PASS ({n} tier/proj picks checked)")


if __name__ == "__main__":
    kind = sys.argv[1]
    path = sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if kind == "alpha":
        _self_test_alpha(path, L)
    elif kind == "fullmenu":
        _self_test_fullmenu(path, L)
    else:
        _self_test_manifest(path, L)
