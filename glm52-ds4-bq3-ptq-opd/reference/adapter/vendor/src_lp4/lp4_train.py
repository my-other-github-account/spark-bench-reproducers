#!/usr/bin/env python3
"""Function-space repair trainer (pilot layers 3,13,23,33,41).

Trains ONLY vqA codebooks (fp32 master -> fp16 STE, exact wire round-trip)
and ternary LUTs (fp32, wire-exact) against banked BF16 teacher logits
(top-8192 support), objective = scorer-exact KL(P_ref,S || Q_S), renorm both
sides, on 1024-token truncations (pos_cutoff=1024 causal-prefill identity).

Student forward = HF DeepseekV4 eager graph (same class/venv as the sealed
builder), attn/norm/router/shared/hc materialized ONCE for all 43 layers
(fp8->bf16, per sealed build_layer_sd mapping), routed experts replaced by
TrainableExperts modules that dequant the EXACT wire bytes (LP4_PACK slices,
verified torch.equal vs sealed readers by lp4_pack.py verify).

Gate numbers are NEVER produced here: export writes wire-format planes and
the sealed pipeline (lp4_rail_run.py + kld_score.py) scores them.

Subcommands:
  equiv      init forward on probe windows vs sealed rail q8192 rows (<2e-2)
  gradcheck  finite-difference vs autograd on one cb entry + one lut entry
  roundtrip  export at init and assert tensor-level equality vs staged planes
  measure    time one training step (no state written)
  train      the pilot run (resumable, quick-val, best-checkpoint)
  export     write trained planes + repointed manifest from a state file
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

VENDOR_ROOT = Path(__file__).resolve().parents[1]
BASE = os.path.expanduser(os.environ["BQ3_ASSET_ROOT"])
sys.path.insert(0, str(VENDOR_ROOT / "src"))
sys.path.insert(0, str(VENDOR_ROOT / "src_lp4"))

import t8192_ds4_build_v3 as v3            # noqa: E402  sealed helpers
import lp4_pack as pk                      # noqa: E402  verified batched deq

DEV = "cuda"
E, N13, K13, N2, K2 = 256, 4096, 4096, 4096, 2048
PILOT = (3, 13, 23, 33, 41)
CKPT = os.path.expanduser(os.environ["BQ3_SOURCE_CHECKPOINT"])
PACK = os.path.expanduser(os.environ.get("BQ3_PACK_DIR", os.path.join(BASE, "LP4_PACK")))
TEACH = os.path.expanduser(os.environ.get("BQ3_STATIC_TEACHER_DIR", os.path.join(BASE, "teacher_calib")))
CORPUS = os.path.expanduser(os.environ.get("BQ3_STATIC_CORPUS", os.path.join(BASE, "static", "windows_ds4_calib.json")))
SEL = os.path.expanduser(os.environ.get("BQ3_CALIB_SELECTION", os.path.join(BASE, "static", "CALIB_SELECTION.json")))
MAN = os.path.expanduser(os.environ.get("BQ3_BASE_MANIFEST", os.path.join(BASE, "static", "LP4_MANIFEST.json")))
T_TRAIN = 1024
DEQ_CHUNK = 4       # unified-mem guard: int64 unpack transients scale with
                    # chunk (w3 lo/hi, vqa gather); 4 keeps peaks ~2GB


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


MEMLOG = os.environ.get("LP4_MEMLOG", "") == "1"


def memlog(tag):
    if not MEMLOG:
        return
    rss = anon = 0
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmRSS"):
                rss = int(ln.split()[1]) >> 20
            elif ln.startswith("RssAnon"):
                anon = int(ln.split()[1]) >> 20
    log(f"MEM {tag} alloc={torch.cuda.memory_allocated()>>30}G "
        f"res={torch.cuda.memory_reserved()>>30}G rss={rss}G anon={anon}G")


# GB10 unified mem: mmap'd plane pages stay mapped-resident after use; with
# a 99G pack they starve NVRM sysmem allocs (cudaMalloc fails at MemFree~1G
# because the kernel won't reclaim mapped page cache for it -> the measure
# OOMs at backward). Evict each layer's plane pages right after use.
import ctypes                                                    # noqa: E402
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
MADV_DONTNEED = 4
_PAGE = 4096


def evict_tensor(t):
    st = t.untyped_storage()
    ptr, nb = st.data_ptr(), st.nbytes()
    a = (ptr + _PAGE - 1) & ~(_PAGE - 1)
    b = (ptr + nb) & ~(_PAGE - 1)
    if b > a:
        _LIBC.madvise(ctypes.c_void_p(a), ctypes.c_size_t(b - a),
                      MADV_DONTNEED)


def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------- STE dequant ops
class VqaDeqFn(torch.autograd.Function):
    """w = fp16_STE(cb32)[codes] * exp2(sc-127) -> bf16. Exact sealed math
    at init (cb32 = wire fp16 .float() -> fp16 cast is identity). Backward
    routes grads straight to the fp32 master (STE through fp16 + bf16)."""

    @staticmethod
    def forward(ctx, cb32, codes, sc):
        cbq = cb32.detach().to(torch.float16).float()
        w = cbq[codes.long()]                       # [N, K/4, 4] fp32
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)
        out = (w.reshape(codes.shape[0], -1) * s).to(torch.bfloat16)
        ctx.save_for_backward(codes, sc)
        return out

    @staticmethod
    def backward(ctx, g):
        codes, sc = ctx.saved_tensors
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)
        gf = (g.float() * s).reshape(codes.shape[0], codes.shape[1], 4)
        grad_cb = torch.zeros(256, 4, device=g.device, dtype=torch.float32)
        grad_cb.index_add_(0, codes.reshape(-1).long(), gf.reshape(-1, 4))
        return grad_cb, None, None


class TernDeqFn(torch.autograd.Function):
    """w = lut32[codes] * exp2(sc-127) -> bf16 (lut is fp32 on the wire;
    directly differentiable, no quantizer between master and wire)."""

    @staticmethod
    def forward(ctx, lut32, codes, sc):
        rep = codes.shape[-1] // sc.shape[-1]
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(rep, dim=-1)
        out = (lut32.detach()[codes.long()] * s).to(torch.bfloat16)
        ctx.save_for_backward(codes, sc)
        return out

    @staticmethod
    def backward(ctx, g):
        codes, sc = ctx.saved_tensors
        rep = codes.shape[-1] // sc.shape[-1]
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(rep, dim=-1)
        gf = g.float() * s
        grad_lut = torch.zeros(3, device=g.device, dtype=torch.float32)
        grad_lut.index_add_(0, codes.reshape(-1).long(), gf.reshape(-1))
        return grad_lut, None, None


# ------------------------------------------------------- experts module
class TrainableExperts(nn.Module):
    """Drop-in for DeepseekV4Experts.forward(flat, top_k_index, top_k_weights)
    dequanting the LP4_PACK wire bytes. Non-pilot: everything frozen, dense
    batched dequant per forward. Pilot: vqa/ternary rows go through the STE
    autograd ops; w3/fp4 rows stay frozen dense (serve-anchor contract)."""

    def __init__(self, L, pilot):
        super().__init__()
        self.L, self.pilot = L, pilot
        d = torch.load(f"{PACK}/layer_{L:03d}.pt", map_location="cpu",
                       mmap=True, weights_only=False)
        self.d = d                                   # plain attr: not in sd
        self.limit = 10.0
        self.act = F.silu
        # small index tensors -> GPU once
        self.a13 = d["assign13"].to(DEV)
        self.a2 = d["assign2"].to(DEV)
        self.row13 = d["row13"].to(DEV)
        self.row2 = d["row2"].to(DEV)
        self.idsl = {}
        for t in ("w3", "vqa", "ternary", "fp4"):
            for w in ("13", "2"):
                self.idsl[(t, w)] = d[f"{t}_ids{w}"].to(DEV).long()
        if pilot:
            self.cb13 = nn.Parameter(d["cb13"].float().to(DEV))
            self.cb2 = nn.Parameter(d["cb2"].float().to(DEV))
            if "tern_lut13" in d:
                self.lut13 = nn.Parameter(
                    d["tern_lut13"].clone().float().to(DEV))
                self.lut2 = nn.Parameter(
                    d["tern_lut2"].clone().float().to(DEV))
            # ---- ARM C: per-unit trainable W3v2 LUTs (init = shared V2 LUT)
            nw13 = len(d["w3_ids13"])
            nw2 = len(d["w3_ids2"])
            if nw13:
                self.w3lut13 = nn.Parameter(
                    d["w3_lut"].clone().float().to(DEV)
                    .unsqueeze(0).repeat(nw13, 1))
            if nw2:
                self.w3lut2 = nn.Parameter(
                    d["w3_lut"].clone().float().to(DEV)
                    .unsqueeze(0).repeat(nw2, 1))

    # ---- frozen dense dequant of selected tiers into SHARED [E,*] buffers
    # (allocated ONCE; per-layer alloc/free + empty_cache churned the GB10
    # unified-mem driver ~2.3GB anon leak per cycle -> OOM. Aliasing is safe:
    # non-reentrant checkpoint recomputes layer L and backprops it BEFORE
    # the L-1 recompute overwrites the buffers.)
    _GU = None
    _DN = None

    @staticmethod
    def _deq_bufs():
        if TrainableExperts._GU is None:
            TrainableExperts._GU = torch.empty(
                E, N13, K13, dtype=torch.bfloat16, device=DEV)
            TrainableExperts._DN = torch.empty(
                E, N2, K2, dtype=torch.bfloat16, device=DEV)
        return TrainableExperts._GU, TrainableExperts._DN

    def _dense(self, tiers13, tiers2):
        d = self.d
        gu, dn = self._deq_bufs()
        with torch.no_grad():
            for w, dst, N, K, tiers in (("13", gu, N13, K13, tiers13),
                                        ("2", dn, N2, K2, tiers2)):
                for t in tiers:
                    ids = self.idsl[(t, w)]
                    n = len(ids)
                    if n == 0:
                        continue
                    for c0 in range(0, n, DEQ_CHUNK):
                        cs = slice(c0, min(c0 + DEQ_CHUNK, n))
                        if t == "w3":
                            wq = pk.deq_w3_batched(
                                d[f"w3_pl{w}"][cs].to(DEV),
                                d[f"w3_sc{w}"][cs].to(DEV),
                                d["w3_lut"], N, K)
                        elif t == "vqa":
                            wq = pk.deq_vqa_batched(
                                d[f"vqa_codes{w}"][cs].to(DEV),
                                d[f"vqa_sc{w}"][cs].to(DEV), d[f"cb{w}"])
                        elif t == "ternary":
                            wq = pk.deq_tern_batched(
                                d[f"tern_codes{w}"][cs].to(DEV),
                                d[f"tern_sc{w}"][cs].to(DEV),
                                d[f"tern_lut{w}"])
                        else:
                            wq = v3.deq_fp4_block32(
                                d[f"fp4_wb{w}"][cs].to(DEV),
                                d[f"fp4_sb{w}"][cs].to(DEV), "e2m1")
                        dst[ids[cs]] = wq
                        del wq
        return gu, dn

    def _w_pilot(self, e, which):
        """Per-expert weight for a pilot layer: STE path for vqa/ternary."""
        t = ("w3", "vqa", "ternary", "fp4")[
            int((self.a13 if which == "13" else self.a2)[e])]
        r = int((self.row13 if which == "13" else self.row2)[e])
        d = self.d
        if t == "vqa":
            cb = self.cb13 if which == "13" else self.cb2
            return VqaDeqFn.apply(
                cb, d[f"vqa_codes{which}"][r].to(DEV),
                d[f"vqa_sc{which}"][r].to(DEV))
        if t == "ternary":
            lut = self.lut13 if which == "13" else self.lut2
            return TernDeqFn.apply(
                lut, d[f"tern_codes{which}"][r].to(DEV),
                d[f"tern_sc{which}"][r].to(DEV))
        if t == "w3":
            lutp = self.w3lut13 if which == "13" else self.w3lut2
            N, K = (N13, K13) if which == "13" else (N2, K2)
            codes = pk.unpack_w3_batched(
                self.d[f"w3_pl{which}"][r:r+1].to(DEV), N, K)[0]
            sb = pk.unpack_scales_batched(
                self.d[f"w3_sc{which}"][r:r+1].to(DEV), N, K // 32)[0]
            s = pk.scol(sb.unsqueeze(0))[0]
            return (lutp[r][codes.long()] * s).to(torch.bfloat16)
        return None  # frozen tier -> dense row

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        if self.pilot:
            gu, dn = self._dense(("w3", "fp4"), ("w3", "fp4"))
        else:
            gu, dn = self._dense(("w3", "vqa", "ternary", "fp4"),
                                 ("w3", "vqa", "ternary", "fp4"))
        for e_ in hit:
            e = int(e_[0])
            top_k_pos, token_idx = torch.where(mask[e])
            xt = hidden_states[token_idx]
            W13 = self._w_pilot(e, "13") if self.pilot else None
            if W13 is None:
                W13 = gu[e]
            cur = F.linear(xt, W13)
            gate, up = cur.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            cur = self.act(gate) * up
            W2 = self._w_pilot(e, "2") if self.pilot else None
            if W2 is None:
                W2 = dn[e]
            cur = F.linear(cur, W2) * top_k_weights[token_idx, top_k_pos,
                                                    None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
            del W13, W2, cur, xt
        self._evict_planes()
        memlog(f"L{self.L:02d} expfwd grad={torch.is_grad_enabled()}")
        return final

    def _evict_planes(self):
        import os as _os3
        if _os3.environ.get("FD_NOEVICT", "0") == "1":
            return
        """madvise(DONTNEED) this layer's big mmap'd plane tensors so
        resident file pages stay bounded (~1 layer) across the traversal."""
        for k, v in self.d.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cpu" \
                    and v.untyped_storage().nbytes() > (16 << 20):
                evict_tensor(v)


# ------------------------------------------------------- model assembly
def build_nonexpert_sd(L, wm, get_tensor):
    """Sealed v3.build_layer_sd mapping MINUS routed experts (those live in
    TrainableExperts). Copied from the sealed builder; expert keys are
    intentionally unread (consumed-marked, same as planes mode)."""
    pre = f"layers.{L}."
    keys = [k for k in wm if k.startswith(pre)]
    sd = {}
    consumed = set()

    def T(name):
        consumed.add(pre + name)
        return get_tensor(pre + name)

    def has(name):
        return (pre + name) in wm

    def fp8(name):
        return v3.deq_fp8_block(T(name + ".weight"), T(name + ".scale"))

    f32 = lambda name: T(name).to(DEV).to(torch.float32)   # noqa: E731
    bf = lambda name: T(name).to(DEV).to(torch.bfloat16)   # noqa: E731

    sd["self_attn.q_a_proj.weight"] = fp8("attn.wq_a")
    sd["self_attn.q_b_proj.weight"] = fp8("attn.wq_b")
    sd["self_attn.kv_proj.weight"] = fp8("attn.wkv")
    sd["self_attn.o_a_proj.weight"] = fp8("attn.wo_a")
    sd["self_attn.o_b_proj.weight"] = fp8("attn.wo_b")
    sd["self_attn.sinks"] = f32("attn.attn_sink")
    sd["self_attn.q_a_norm.weight"] = bf("attn.q_norm.weight")
    sd["self_attn.kv_norm.weight"] = bf("attn.kv_norm.weight")
    sd["input_layernorm.weight"] = bf("attn_norm.weight")
    sd["post_attention_layernorm.weight"] = bf("ffn_norm.weight")
    if has("attn.compressor.wkv.weight"):
        sd["self_attn.compressor.position_bias"] = f32("attn.compressor.ape")
        sd["self_attn.compressor.kv_norm.weight"] = bf(
            "attn.compressor.norm.weight")
        sd["self_attn.compressor.kv_proj.weight"] = bf(
            "attn.compressor.wkv.weight")
        sd["self_attn.compressor.gate_proj.weight"] = bf(
            "attn.compressor.wgate.weight")
    if has("attn.indexer.wq_b.weight"):
        idx = "self_attn.compressor.indexer."
        sd[idx + "position_bias"] = f32("attn.indexer.compressor.ape")
        sd[idx + "kv_norm.weight"] = bf("attn.indexer.compressor.norm.weight")
        sd[idx + "kv_proj.weight"] = bf("attn.indexer.compressor.wkv.weight")
        sd[idx + "gate_proj.weight"] = bf(
            "attn.indexer.compressor.wgate.weight")
        sd[idx + "q_b_proj.weight"] = fp8("attn.indexer.wq_b")
        sd[idx + "scorer.weights_proj.weight"] = bf(
            "attn.indexer.weights_proj.weight")
    sd["mlp.gate.weight"] = bf("ffn.gate.weight")
    if has("ffn.gate.tid2eid"):
        sd["mlp.gate.tid2eid"] = T("ffn.gate.tid2eid").to(DEV)
    if has("ffn.gate.bias"):
        sd["mlp.gate.e_score_correction_bias"] = f32("ffn.gate.bias")
    sd["attn_hc.fn"] = f32("hc_attn_fn")
    sd["attn_hc.base"] = f32("hc_attn_base")
    sd["attn_hc.scale"] = f32("hc_attn_scale")
    sd["ffn_hc.fn"] = f32("hc_ffn_fn")
    sd["ffn_hc.base"] = f32("hc_ffn_base")
    sd["ffn_hc.scale"] = f32("hc_ffn_scale")
    sd["mlp.shared_experts.gate_proj.weight"] = fp8("ffn.shared_experts.w1")
    sd["mlp.shared_experts.up_proj.weight"] = fp8("ffn.shared_experts.w3")
    sd["mlp.shared_experts.down_proj.weight"] = fp8("ffn.shared_experts.w2")
    for k in keys:
        if ".ffn.experts." in k:
            consumed.add(k)
    missed = set(keys) - consumed
    if missed:
        raise RuntimeError(f"layer {L}: unconsumed: {sorted(missed)[:8]}")
    return sd


class Student:
    def __init__(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4RotaryEmbedding)
        from safetensors import safe_open
        import transformers
        log(f"transformers {transformers.__version__} torch "
            f"{torch.__version__}")
        self.config = AutoConfig.from_pretrained(CKPT)
        self.wm = json.load(open(os.path.join(
            CKPT, "model.safetensors.index.json")))["weight_map"]
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(
                self.config, attn_implementation="eager")
        self.model.eval()
        handles = {}

        def get_tensor(name):
            sh = self.wm[name]
            if sh not in handles:
                while len(handles) >= 3:
                    handles.pop(next(iter(handles)))
                handles[sh] = safe_open(os.path.join(CKPT, sh),
                                        framework="pt")
            return handles[sh].get_tensor(name)

        self.get_tensor = get_tensor
        m = self.model
        log("materializing embed/head/norm/hc_head")
        m.model.embed_tokens.weight = nn.Parameter(
            get_tensor("embed.weight").to(DEV).to(torch.bfloat16),
            requires_grad=False)
        m.lm_head.weight = nn.Parameter(
            get_tensor("head.weight").to(DEV).to(torch.bfloat16),
            requires_grad=False)
        m.model.norm.weight = nn.Parameter(
            get_tensor("norm.weight").to(DEV).to(torch.bfloat16),
            requires_grad=False)
        m.model.hc_head.hc_fn = nn.Parameter(
            get_tensor("hc_head_fn").to(DEV).float(), requires_grad=False)
        m.model.hc_head.hc_base = nn.Parameter(
            get_tensor("hc_head_base").to(DEV).float(), requires_grad=False)
        m.model.hc_head.hc_scale = nn.Parameter(
            get_tensor("hc_head_scale").to(DEV).float(), requires_grad=False)
        m.model.rotary_emb = DeepseekV4RotaryEmbedding(self.config).to(DEV)

        self.experts = {}
        t0 = time.time()
        for L in range(self.config.num_hidden_layers):
            lay = m.model.layers[L]
            te = TrainableExperts(L, pilot=L in PILOT)
            lay.mlp.experts = te
            self.experts[L] = te
            sd = build_nonexpert_sd(L, self.wm, get_tensor)
            v3.materialize_layer(m, L, sd, self.config)
            del sd
        train_ids = {id(q) for te in self.experts.values()
                     for q in te.parameters()}
        for p in m.parameters():
            if id(p) not in train_ids:
                p.requires_grad_(False)
        log(f"materialized 43 layers in {time.time()-t0:.0f}s "
            f"(gpu_res {torch.cuda.memory_reserved()>>30}G)")

    def trainable(self):
        cbs, luts = [], []
        for L in PILOT:
            te = self.experts[L]
            cbs += [te.cb13, te.cb2]
            if hasattr(te, "lut13"):
                luts += [te.lut13, te.lut2]
            if hasattr(te, "w3lut13"):
                luts += [te.w3lut13]
            if hasattr(te, "w3lut2"):
                luts += [te.w3lut2]
        return cbs, luts

    def forward_batch(self, ids, requires_grad):
        """ids [B, T_TRAIN] long on DEV -> hidden after final norm/hc_head
        [B, T, hidden] bf16. Layers 0..min(PILOT)-1 run no-grad (no trainable
        params below); the rest are gradient-checkpointed when training."""
        from transformers.masking_utils import (
            create_sliding_window_causal_mask)
        from transformers.cache_utils import DynamicCache
        m, config = self.model, self.config
        pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
        embeds = m.model.embed_tokens(ids)
        pe = {
            "main": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                       layer_type="main"),
            "compress": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                           layer_type="compress"),
        }
        mask = create_sliding_window_causal_mask(
            config=config, inputs_embeds=embeds, attention_mask=None,
            past_key_values=DynamicCache(config=config), position_ids=pos)
        hidden = embeds.unsqueeze(2).expand(
            -1, -1, config.hc_mult, -1).contiguous()
        del embeds

        def run_layer(L, h):
            return m.model.layers[L](
                h, position_embeddings=pe, position_ids=pos,
                attention_mask=mask, input_ids=ids,
                past_key_values=DynamicCache(config=config))

        first_pilot = min(PILOT)
        with torch.no_grad():
            for L in range(first_pilot):
                hidden = run_layer(L, hidden)
                torch.cuda.empty_cache()
        for L in range(first_pilot, config.num_hidden_layers):
            if requires_grad:
                hidden = checkpoint(run_layer, L, hidden,
                                    use_reentrant=False)
            else:
                with torch.no_grad():
                    hidden = run_layer(L, hidden)
            torch.cuda.empty_cache()
        return m.model.norm(m.model.hc_head(hidden))


# ------------------------------------------------------- data + loss
def load_corpus():
    return json.load(open(CORPUS))


def window_ids(corpus, k):
    t = corpus[k]["token_ids"][:T_TRAIN]
    ids = torch.full((T_TRAIN,), 1, dtype=torch.long)
    ids[:len(t)] = torch.tensor(t, dtype=torch.long)
    return ids, min(corpus[k]["real_len"], T_TRAIN)


def teacher_rows(k):
    r = torch.load(os.path.join(TEACH, f"t8192_win{k}.pt"),
                   map_location="cpu", weights_only=False)
    idx = r["idx"].long()[:T_TRAIN].to(DEV)
    lp = r["logprob"].float()[:T_TRAIN].to(DEV)
    lp_n = lp - lp.logsumexp(-1, keepdim=True)
    return idx, lp_n, lp_n.exp()


def kl_loss(h_j, ids_j, T_j, student, idx, lp_n, p_n):
    logits = student.model.lm_head(h_j[:T_j].to(torch.bfloat16)).float()
    lq = torch.log_softmax(logits, dim=-1)
    q = lq.gather(1, idx[:T_j])
    lq_n = q - q.logsumexp(-1, keepdim=True)
    return (p_n[:T_j] * (lp_n[:T_j] - lq_n)).sum(-1).mean()


def batch_loss(student, corpus, wins, requires_grad):
    ids = torch.stack([window_ids(corpus, k)[0] for k in wins]).to(DEV)
    rls = [window_ids(corpus, k)[1] for k in wins]
    h = student.forward_batch(ids, requires_grad)
    loss = 0.0
    for j, k in enumerate(wins):
        idx, lp_n, p_n = teacher_rows(k)
        loss = loss + kl_loss(h[j], ids[j], rls[j], student, idx, lp_n, p_n)
        del idx, lp_n, p_n
    return loss / len(wins)


# ------------------------------------------------------- subcommands
def cmd_equiv(a):
    """Init training-graph forward vs sealed rail rows on probe windows."""
    student = Student()
    corpus = load_corpus()
    probes = [int(x) for x in a.windows.split(",")]
    worst = 0.0
    for k in probes:
        ref = torch.load(os.path.join(TEACH, f"t8192_win{k}.pt"),
                         map_location=DEV, weights_only=False)
        sealed = torch.load(os.path.join(BASE, a.rail_dir,
                                         f"q8192_win{k}.pt"),
                            map_location=DEV, weights_only=False)
        if a.tlen > T_TRAIN:            # diagnostic: sealed-rail length
            t = corpus[k]["token_ids"][:a.tlen]
            ids = torch.full((a.tlen,), 1, dtype=torch.long)
            ids[:len(t)] = torch.tensor(t, dtype=torch.long)
            rl = min(corpus[k]["real_len"], T_TRAIN)
        else:
            ids, rl = window_ids(corpus, k)
        with torch.no_grad():
            h = student.forward_batch(ids.unsqueeze(0).to(DEV), False)
            logits = student.model.lm_head(
                h[0, :rl].to(torch.bfloat16)).float()
            lq = torch.log_softmax(logits, dim=-1)
            mine = lq.gather(1, ref["idx"].long()[:rl])
        T = min(rl, sealed["q_lp_at_ref"].shape[0])
        d = (mine[:T] - sealed["q_lp_at_ref"].float()[:T]).abs()
        for b0 in range(0, T, 256):
            db = d[b0:b0 + 256]
            log(f"  pos[{b0}:{b0+db.shape[0]}] max={db.max().item():.5f} "
                f"mean={db.mean().item():.6f}")
        am_mine = lq.argmax(-1)[:T]
        am_agree = (am_mine == sealed["q_argmax"].long()[:T]).float().mean()
        log(f"win{k}: max|dq_lp|={d.max().item():.5f} "
            f"mean={d.mean().item():.6f} argmax_agree={am_agree:.4f}")
        worst = max(worst, d.max().item())
        del h, logits, lq, mine, ref, sealed
        torch.cuda.empty_cache()
    ok = worst < 2e-2
    log(f"EQUIV {'PASS' if ok else 'FAIL'} (worst {worst:.5f} vs 2e-2)")
    return 0 if ok else 1


def cmd_gradcheck(a):
    """Op-level FD vs autograd for the two STE dequant ops.

    The module-level FD is swamped by bf16 output rounding (probe signal
    ~1e-7 vs mean-loss noise ~5e-6), so we check the ops directly:
    loss = (op(...).float() * G).sum() has dloss/dout = G EXACTLY, so the
    autograd grad is pure fp32 backward math and must match the finite
    difference of the fp32 REFERENCE function (same math, no bf16 cast).
    For the vqa op the forward snaps cb to fp16 (STE), so the FD
    denominator is the realized fp16 step fp16(v+h)-fp16(v-h)."""
    torch.manual_seed(0)
    L = PILOT[0]
    d = torch.load(f"{PACK}/layer_{L:03d}.pt", map_location="cpu",
                   mmap=True, weights_only=False)
    results = []

    # ---- vqa cb13
    r = 0
    codes = d["vqa_codes13"][r].to(DEV)
    sc = d["vqa_sc13"][r].to(DEV)
    cb = nn.Parameter(d["cb13"].float().to(DEV))
    G = torch.randn(N13, K13, device=DEV)
    s_col = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)

    def ref_loss(cb_val):
        cbq = cb_val.detach().to(torch.float16).float()
        w = cbq[codes.long()].reshape(codes.shape[0], -1) * s_col
        return (w * G).sum().item()

    loss = (VqaDeqFn.apply(cb, codes, sc).float() * G).sum()
    loss.backward()
    ci, cj = int(codes[0, 0]), 0
    ag = cb.grad[ci, cj].item()
    v = cb.data[ci, cj].item()
    h = max(abs(v) / 16.0, 1e-2)
    with torch.no_grad():
        cb.data[ci, cj] = v + h
        lp = ref_loss(cb.data)
        vp = float(torch.tensor(v + h).to(torch.float16))
        cb.data[ci, cj] = v - h
        lm = ref_loss(cb.data)
        vm = float(torch.tensor(v - h).to(torch.float16))
        cb.data[ci, cj] = v
    fd = (lp - lm) / (vp - vm)
    rel = abs(fd - ag) / max(abs(fd), abs(ag), 1e-8)
    log(f"vqa cb13[{ci},{cj}]: autograd={ag:.6e} fd={fd:.6e} rel={rel:.4f}")
    results.append(rel)

    # ---- ternary lut13 (if this layer has ternary experts)
    if "tern_codes13" in d and len(d["tern_codes13"]):
        codes_t = d["tern_codes13"][0].to(DEV)
        sc_t = d["tern_sc13"][0].to(DEV)
        lut = nn.Parameter(d["tern_lut13"].clone().float().to(DEV))
        Gt = torch.randn(codes_t.shape[0], codes_t.shape[1], device=DEV)
        rep = codes_t.shape[-1] // sc_t.shape[-1]
        s_t = torch.exp2(sc_t.float() - 127.0).repeat_interleave(rep, dim=-1)

        def ref_loss_t(lut_val):
            return ((lut_val.detach()[codes_t.long()] * s_t) * Gt
                    ).sum().item()

        loss = (TernDeqFn.apply(lut, codes_t, sc_t).float() * Gt).sum()
        loss.backward()
        i = 0
        ag = lut.grad[i].item()
        v = lut.data[i].item()
        h = max(abs(v) / 16.0, 1e-3)
        with torch.no_grad():
            lut.data[i] = v + h
            lp = ref_loss_t(lut.data)
            lut.data[i] = v - h
            lm = ref_loss_t(lut.data)
            lut.data[i] = v
        fd = (lp - lm) / (2 * h)
        rel = abs(fd - ag) / max(abs(fd), abs(ag), 1e-8)
        log(f"tern lut13[{i}]: autograd={ag:.6e} fd={fd:.6e} rel={rel:.4f}")
        results.append(rel)

    ok = all(r < 0.02 for r in results)
    log(f"GRADCHECK {'PASS' if ok else 'FAIL'} (rel tol 0.02)")
    return 0 if ok else 1


def export_planes(state, out_tag, check_init=False):
    """Write trained vqa/ternary planes (pilot layers) + symlinks for the
    rest + a repointed manifest. Wire formats byte-preserved except the
    trained leaves (cb fp16, lut fp32)."""
    man = json.load(open(MAN))
    vqa_src = os.path.expanduser(man["tiers"]["vqa"]["planes_dir"])
    tern_src = os.path.expanduser(man["tiers"]["ternary"]["planes_dir"])
    vqa_dst = os.path.join(BASE, f"planes_vqa_{out_tag}")
    tern_dst = os.path.join(BASE, f"planes_ternary_{out_tag}")
    os.makedirs(vqa_dst, exist_ok=True)
    os.makedirs(tern_dst, exist_ok=True)
    n_link = n_write = 0
    for src, dst, pref in ((vqa_src, vqa_dst, "vqa_layer"),
                           (tern_src, tern_dst, "tern_layer")):
        for f in sorted(os.listdir(src)):
            if not (f.startswith(pref) and f.endswith(".pt")):
                continue
            sp, dp = os.path.join(src, f), os.path.join(dst, f)
            L = int(f.split("_")[-1].split(".")[0])
            if L not in PILOT:
                if not os.path.exists(dp):
                    os.symlink(sp, dp)
                n_link += 1
                continue
            d = torch.load(sp, map_location="cpu", weights_only=False)
            key = f"L{L}"
            if pref == "vqa_layer":
                d["cb13"] = state[key]["cb13"].to(torch.float16)
                d["cb2"] = state[key]["cb2"].to(torch.float16)
            else:
                if key in state and "lut13" in state[key]:
                    d["lut13"] = state[key]["lut13"].float()
                    d["lut2"] = state[key]["lut2"].float()
            torch.save(d, dp + ".tmp")
            os.replace(dp + ".tmp", dp)
            n_write += 1
            if check_init:
                orig = torch.load(sp, map_location="cpu",
                                  weights_only=False)
                new = torch.load(dp, map_location="cpu", weights_only=False)
                for k in orig:
                    if isinstance(orig[k], torch.Tensor):
                        assert torch.equal(orig[k], new[k]), (f, k)
                    else:
                        assert orig[k] == new[k], (f, k)
                log(f"  roundtrip {f}: all keys equal")
    man["tiers"]["vqa"]["planes_dir"] = vqa_dst
    man["tiers"]["ternary"]["planes_dir"] = tern_dst
    man["lp4_export"] = {"tag": out_tag, "pilot_layers": list(PILOT),
                         "trained": ["vqa cb13/cb2 fp16",
                                     "ternary lut13/lut2 fp32"]}
    mp = os.path.join(BASE, "static",
                      f"LP4_MANIFEST_S7LOCAL_{out_tag.upper()}.json")
    json.dump(man, open(mp, "w"), indent=1)
    log(f"export: {n_write} written, {n_link} symlinked, manifest {mp}")
    return mp


def state_from_experts(experts):
    st = {}
    for L in PILOT:
        te = experts[L]
        st[f"L{L}"] = {"cb13": te.cb13.data.cpu(), "cb2": te.cb2.data.cpu()}
        if hasattr(te, "lut13"):
            st[f"L{L}"]["lut13"] = te.lut13.data.cpu()
            st[f"L{L}"]["lut2"] = te.lut2.data.cpu()
    return st


def cmd_roundtrip(a):
    st = {}
    for L in PILOT:
        d = torch.load(f"{PACK}/layer_{L:03d}.pt", map_location="cpu",
                       mmap=True, weights_only=False)
        st[f"L{L}"] = {"cb13": d["cb13"].float(), "cb2": d["cb2"].float()}
        if "tern_lut13" in d:
            st[f"L{L}"]["lut13"] = d["tern_lut13"].float()
            st[f"L{L}"]["lut2"] = d["tern_lut2"].float()
    export_planes(st, "inittest", check_init=True)
    log("ROUNDTRIP PASS")
    return 0


def qval(student, corpus, wins, mb):
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(wins), mb):
            tot += batch_loss(student, corpus, wins[i:i + mb],
                              False).item() * len(wins[i:i + mb])
    return tot / len(wins)




# ------------------------------------------------------- fast convergence rig
def cmd_fastdiag(a):
    """Fast-iteration convergence diagnostic: small fixed batch,
    per-step probe KL, per-group grad norms, param displacement, grad cosine.
    Steps are mb=1 x FD_WINS windows; probe = FD_PROBE fixed windows every step.
    """
    import itertools
    out = os.path.expanduser(a.out or "/tmp/bq3-fastdiag")
    os.makedirs(out, exist_ok=True)
    tlog = os.path.join(out, "fastdiag.jsonl")
    student = Student()
    cbs, luts = student.trainable()
    groups = {"cb": cbs, "lut": luts}
    n_par = sum(p_.numel() for g in groups.values() for p_ in g)
    log(f"fastdiag trainable: {n_par} params in {len(cbs)}cb+{len(luts)}lut")
    sel = json.load(open(SEL))
    fit, val = sel["fit_ids"], sel["val_ids"]
    corpus = load_corpus()
    FIT_W = fit[: int(os.environ.get("FD_WINS", "2"))]
    PROBE_W = val[: int(os.environ.get("FD_PROBE", "2"))]
    theta0 = {k: [p_.detach().clone() for p_ in g] for k, g in groups.items()}
    opt = torch.optim.Adam([p_ for g in groups.values() for p_ in g], lr=a.lr)
    prev_flat = None
    log("fd: corpus loaded FIT_W=%s PROBE_W=%s" % (FIT_W, PROBE_W))
    log("fd: starting probe0 qval...")
    p0 = qval(student, corpus, PROBE_W, 1)
    log(f"probe0 KL = {p0:.6f} (probe={len(PROBE_W)}w fit={len(FIT_W)}w)")
    jrow(tlog, kind="probe", step=0, probe=round(p0, 6))
    for step in range(1, a.max_steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(student, corpus, FIT_W, True)
        loss.backward()
        gns = {}
        flat = []
        for k, g in groups.items():
            gs = [p_.grad for p_ in g if p_.grad is not None]
            gns[k] = float(torch.sqrt(sum((x * x).sum() for x in gs))) if gs else 0.0
            flat += [x.reshape(-1) for x in gs]
        fl = torch.cat(flat) if flat else torch.zeros(1, device=DEV)
        cos = float((fl @ prev_flat) / (fl.norm() * prev_flat.norm() + 1e-12)) if (prev_flat is not None and fl.numel() == prev_flat.numel()) else 0.0
        prev_flat = fl.detach().clone()
        opt.step()
        disp = {k: float(torch.sqrt(sum(((a_ - b_) ** 2).sum()
                for a_, b_ in zip([p_.detach() for p_ in g], theta0[k]))))
                for k, g in groups.items()}
        pk = qval(student, corpus, PROBE_W, 1)
        dt = time.time() - t0
        log(f"fd step {step} loss {float(loss):.6f} probe {pk:.6f} "
            f"d_probe {pk - p0:+.6f} gn_cb {gns.get('cb', 0):.4f} gn_lut {gns.get('lut', 0):.4f} "
            f"cos {cos:+.3f} disp_cb {disp.get('cb', 0):.4f} disp_lut {disp.get('lut', 0):.4f} {dt:.0f}s")
        jrow(tlog, kind="fd", step=step, loss=round(float(loss), 6),
             probe=round(pk, 6), cos=round(cos, 4),
             **{f"gn_{k}": round(v, 5) for k, v in gns.items()},
             **{f"disp_{k}": round(v, 5) for k, v in disp.items()})

def cmd_train(a):
    sel = json.load(open(SEL))
    fit, val = sel["fit_ids"], sel["val_ids"]
    qwins = val[:8]
    corpus = load_corpus()
    out = os.path.join(BASE, a.out)
    os.makedirs(out, exist_ok=True)
    tlog = os.path.join(out, "TRAIN_LOG.jsonl")
    hb = os.path.join(out, "HEARTBEAT")
    student = Student()
    cbs, luts = student.trainable()
    n_par = sum(p.numel() for p in cbs + luts)
    log(f"trainable: {len(cbs)} cb + {len(luts)} lut tensors, {n_par} params")
    opt = torch.optim.Adam(
        [{"params": cbs, "lr": a.lr},
         {"params": luts, "lr": a.lr * a.lut_lr_scale}])
    step0, best = 0, (float("inf"), -1)
    last_p = os.path.join(out, "state_last.pt")
    if os.path.exists(last_p) and not a.fresh:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        step0, best = ck["step"], tuple(ck["best"])
        opt.load_state_dict(ck["opt"])
        for L in PILOT:
            te = student.experts[L]
            te.cb13.data.copy_(ck["state"][f"L{L}"]["cb13"])
            te.cb2.data.copy_(ck["state"][f"L{L}"]["cb2"])
            if hasattr(te, "lut13"):
                te.lut13.data.copy_(ck["state"][f"L{L}"]["lut13"])
                te.lut2.data.copy_(ck["state"][f"L{L}"]["lut2"])
        torch.set_rng_state(ck["rng"])
        log(f"resumed at step {step0} (best qval {best[0]:.6f} @ {best[1]})")
    else:
        torch.manual_seed(a.seed)

    def save(path, step):
        torch.save({"step": step, "best": list(best),
                    "state": state_from_experts(student.experts),
                    "opt": opt.state_dict(),
                    "rng": torch.get_rng_state()}, path + ".tmp")
        os.replace(path + ".tmp", path)

    if step0 == 0:
        v0 = qval(student, corpus, qwins, a.mb_val)
        best = (v0, 0)
        jrow(tlog, kind="qval", step=0, qval=round(v0, 6), best=True)
        log(f"init quick-val KL = {v0:.6f}")
        save(os.path.join(out, "state_best.pt"), 0)

    t_run = time.time()
    steps_per_epoch = math.ceil(len(fit) / a.mb)
    max_steps = a.max_steps
    step = step0
    while step < max_steps:
        if (time.time() - t_run) / 3600 > a.max_hours:
            log(f"wall guard {a.max_hours}h hit at step {step}")
            break
        ep = step // steps_per_epoch
        g = torch.Generator().manual_seed(a.seed + ep)
        perm = torch.randperm(len(fit), generator=g).tolist()
        i0 = (step % steps_per_epoch) * a.mb
        wins = [fit[perm[i % len(fit)]] for i in range(i0, i0 + a.mb)]
        lr = a.lr * (0.1 + 0.9 * 0.5 *
                     (1 + math.cos(math.pi * step / max_steps)))
        opt.param_groups[0]["lr"] = lr
        opt.param_groups[1]["lr"] = lr * a.lut_lr_scale
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(student, corpus, wins, True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(cbs + luts, 1.0)
        li = loss.item()
        if math.isnan(li) or math.isinf(li):
            log(f"step {step}: NaN/inf loss, skipping update")
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        dt = time.time() - t0
        step += 1
        jrow(tlog, kind="step", step=step, loss=round(li, 6),
             lr=round(lr, 6), grad_norm=round(float(gn), 4),
             wins=wins, dt=round(dt, 1))
        with open(hb, "w") as f:
            json.dump({"step": step, "ts": time.time()}, f)
        log(f"step {step}/{max_steps} loss {li:.6f} gn {float(gn):.3f} "
            f"lr {lr:.5f} {dt:.0f}s")
        if step % a.save_every == 0:
            save(last_p, step)
        if step % a.qval_every == 0 or step == max_steps:
            v = qval(student, corpus, qwins, a.mb_val)
            is_best = v < best[0]
            if is_best:
                best = (v, step)
                save(os.path.join(out, "state_best.pt"), step)
            jrow(tlog, kind="qval", step=step, qval=round(v, 6),
                 best=is_best)
            log(f"quick-val KL = {v:.6f} (best {best[0]:.6f} @ {best[1]})")
    save(last_p, step)
    log(f"train done at step {step}; best qval {best[0]:.6f} @ "
        f"step {best[1]}")
    return 0


def cmd_measure(a):
    corpus = load_corpus()
    sel = json.load(open(SEL))
    student = Student()
    cbs, luts = student.trainable()
    wins = sel["fit_ids"][:a.mb]
    for trial in range(2):
        t0 = time.time()
        loss = batch_loss(student, corpus, wins, True)
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        g = sum(float(p.grad.abs().sum()) for p in cbs + luts
                if p.grad is not None)
        log(f"trial{trial}: fwd {t1-t0:.0f}s bwd {t2-t1:.0f}s "
            f"loss {loss.item():.6f} sum|g| {g:.4f} "
            f"gpu_res {torch.cuda.memory_reserved()>>30}G")
        for p in cbs + luts:
            p.grad = None
    return 0


def cmd_export(a):
    ck = torch.load(os.path.join(BASE, a.out, a.state),
                    map_location="cpu", weights_only=False)
    log(f"exporting state from step {ck['step']} (best {ck['best']})")
    export_planes(ck["state"], a.tag)
    return 0


def cmd_ldiff(a):
    """Per-layer bisect: sealed-built layer vs my trainable layer.

    Runs two hidden streams on one window (sealed-graph stream A drives
    both), building each sealed layer transiently via v3.build_layer_sd
    (planes mode, BestqManifestSource) exactly as the rail does.
      op_max[L]     = max|myLayer(hA) - sealedLayer(hA)|   (matched input)
    Reports the first layer whose operator output diverges."""
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import (
        create_sliding_window_causal_mask)
    from transformers.cache_utils import DynamicCache
    from bq_sources_revc import BestqManifestSource
    student = Student()
    m, config = student.model, student.config
    src = BestqManifestSource(MAN)
    with torch.device("meta"):
        model2 = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager")
    model2.eval()

    corpus = load_corpus()
    k = int(a.windows.split(",")[0])
    t = corpus[k]["token_ids"][:a.tlen]
    ids = torch.full((1, a.tlen), 1, dtype=torch.long)
    ids[0, :len(t)] = torch.tensor(t, dtype=torch.long)
    ids = ids.to(DEV)
    pos = torch.arange(a.tlen, device=DEV).unsqueeze(0)

    with torch.no_grad():
        embeds = m.model.embed_tokens(ids)
        pe = {
            "main": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                       layer_type="main"),
            "compress": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                           layer_type="compress"),
        }
        cacheA = DynamicCache(config=config)
        mask = create_sliding_window_causal_mask(
            config=config, inputs_embeds=embeds, attention_mask=None,
            past_key_values=cacheA, position_ids=pos)
        hA = embeds.unsqueeze(2).expand(
            -1, -1, config.hc_mult, -1).contiguous()
        del embeds

        for L in range(config.num_hidden_layers):
            sd = v3.build_layer_sd(L, student.wm, student.get_tensor,
                                   "planes", planes=src)
            layA = v3.materialize_layer(model2, L, sd, config)
            del sd
            hA_next = layA(hA, position_embeddings=pe, position_ids=pos,
                           attention_mask=mask, input_ids=ids,
                           past_key_values=cacheA)
            v3.dematerialize_layer(model2, L)
            torch.cuda.empty_cache()
            hB_next = m.model.layers[L](
                hA, position_embeddings=pe, position_ids=pos,
                attention_mask=mask, input_ids=ids,
                past_key_values=DynamicCache(config=config))
            torch.cuda.empty_cache()
            dop = (hB_next.float() - hA_next.float()).abs()
            log(f"L{L:02d} op_max={dop.max().item():.6f} "
                f"op_mean={dop.mean().item():.7f} "
                f"h_scale={hA_next.float().abs().mean().item():.4f}")
            del hB_next, dop
            hA = hA_next
    return 0


def _cmd_export_orig(a):
    ck = torch.load(os.path.join(BASE, a.out, a.state),
                    map_location="cpu", weights_only=False)
    log(f"exporting state from step {ck['step']} (best {ck['best']})")
    export_planes(ck["state"], a.tag)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("equiv", "gradcheck", "roundtrip", "fastdiag",
                                    "measure", "train", "export", "ldiff"))
    ap.add_argument("--windows", default="0,1")
    ap.add_argument("--tlen", type=int, default=T_TRAIN)
    ap.add_argument("--rail-dir", default="rail_base")
    ap.add_argument("--out", default="train_v1")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--mb-val", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lut-lr-scale", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=64)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--qval-every", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--state", default="state_best.pt")
    ap.add_argument("--tag", default="trained")
    a = ap.parse_args()
    sys.exit({"equiv": cmd_equiv, "gradcheck": cmd_gradcheck,
              "roundtrip": cmd_roundtrip, "measure": cmd_measure,
              "train": cmd_train, "export": cmd_export, "fastdiag": cmd_fastdiag,
              "ldiff": cmd_ldiff}[a.cmd](a))
