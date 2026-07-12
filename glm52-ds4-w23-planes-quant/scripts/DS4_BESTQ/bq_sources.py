#!/usr/bin/env python3
"""t_3d6e422d plane sources for the sealed v3 builder / MMLU harness.

AlphaPlaneSource : per-layer plane dir reader (sealed v3.PlaneSource
    delegate) that applies the per-tensor fractional alpha from
    layer_NNN.alphas.npy [E,3] (w1, w3, down) when present. When absent,
    byte-identical passthrough of the sealed reader.

BestqManifestSource : per-expert mixed-tier reader for the R7 manifests.
    Clone of the sealed ~/missions/DS4_R6/manifest_planes.py
    (t_29c4872c) with the w2 tier read through AlphaPlaneSource.

Self-tests (run before any rail):
    python3 bq_sources.py alpha <planes_dir> [L]
    python3 bq_sources.py manifest <manifest.json> [L]
"""
import json
import os
import sys

import numpy as np
import torch

_TEACH = os.path.expanduser("~/missions/DS4_TEACHER")
if _TEACH not in sys.path:
    sys.path.insert(0, _TEACH)

import t8192_ds4_build_v3 as v3  # noqa: E402

# Captured at import: run wrappers monkeypatch v3.PlaneSource, so a
# call-time lookup would recurse (same guard as the sealed manifest_planes).
_SealedPlaneSource = v3.PlaneSource


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


class BestqManifestSource:
    """Manifest {assignment: {L: {e: w2|w3|fp4}}, tiers: {...}} reader."""

    def __init__(self, manifest_path):
        mp = os.path.expanduser(manifest_path)
        assert os.path.isfile(mp), mp
        self.man = json.load(open(mp))
        self.assign = self.man["assignment"]
        tiers = self.man["tiers"]
        self.src = {
            "w2": AlphaPlaneSource(tiers["w2"]["planes_dir"]),
            "w3": _SealedPlaneSource(
                os.path.expanduser(tiers["w3"]["planes_dir"])),
        }
        self.ckpt = os.path.expanduser(
            self.man.get("ckpt_dir", "~/models/hf/DeepSeek-V4-Flash"))
        self.wm = json.load(open(os.path.join(
            self.ckpt, "model.safetensors.index.json")))["weight_map"]
        self._handles = {}

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
        exp_w2, dims = self.src["w2"].layer(L)
        exp_w3, dims3 = self.src["w3"].layer(L)
        assert dims == dims3, (dims, dims3)

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
            t = amap[str(e)]
            if t == "w2":
                return exp_w2(e, which)
            if t == "w3":
                return exp_w3(e, which)
            assert t == "fp4", t
            return fp4_expert(e, which)

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
    picks = {t: None for t in ("w2", "w3", "fp4")}
    for e in range(256):
        t = amap[str(e)]
        if picks[t] is None:
            picks[t] = e
    print(f"[self-test manifest] L{L} picks: {picks}")
    if picks["w2"] is not None:
        e = picks["w2"]
        assert torch.equal(exp(e, "13"), e2(e, "13")), "w2 delegate"
        assert torch.equal(exp(e, "2"), e2(e, "2"))
        print(f"  w2 tier e{e}: delegate exact OK")
    if picks["w3"] is not None:
        e = picks["w3"]
        assert torch.equal(exp(e, "13"), e3(e, "13")), "w3 delegate"
        assert torch.equal(exp(e, "2"), e3(e, "2"))
        print(f"  w3 tier e{e}: delegate exact OK")
    e = picks["fp4"] if picks["fp4"] is not None else picks["w3"]
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
    print("[self-test manifest] PASS")


if __name__ == "__main__":
    kind = sys.argv[1]
    path = sys.argv[2]
    L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if kind == "alpha":
        _self_test_alpha(path, L)
    else:
        _self_test_manifest(path, L)
