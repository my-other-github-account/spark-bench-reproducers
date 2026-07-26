#!/usr/bin/env python3
"""DS4-Flash teacher rail + W2/planes candidate rail (PUBLIC_TASK, v3 PUBLIC_TASK).

v3 adds --mode planes: routed experts dequant from the SHIPPED plane bytes
(moe_w2_planes / moe_w3_planes wire format on disk, auto-detected by plane
byte-width, LUT from meta.json when present). This is the exact interface
calibrated (GPTQ) planes arrive in, so R4/R5 rows are a --planes-dir swap.
Requires planes_unpack.py alongside.

Teacher mode (--mode bf16):
  out/t8192_win<k>.pt  {"idx": int32 [T,8192] desc by teacher lp,
                        "logprob": fp16 [T,8192] full-softmax lp}
  Teacher = source ckpt dequantized to bf16 (fp8 e4m3 block-128 dense,
  fp4 e2m1 block-32 e8m0 routed experts), HF deepseek_v4 eager graph,
  layer-streamed (one materialized layer at a time), shards streamed
  over QSFP from compute-node-6.

Candidate mode (--mode w2 --ref-dir <teacher_dir>):
  out/q8192_win<k>.pt {"q_lp_at_ref": fp16 [T,8192] full-softmax lp
                       gathered at ref idx, "q_argmax": int32 [T]}
  Identical forward except ROUTED experts dequant through the shipped
  W2 sign-sym codebook {-4,-1,1,4} (vllm-moet moe_w2_planes nibble->code
  LUT, same e8m0 scales -> numerically identical to serve-side planes).
  Attn / shared_experts / gate / dense stay on the teacher bf16 path,
  matching the G2-sealed serve (planes cover routed experts only).

Corpus: windows_ds4_eval.json (512 win). Downstream scoring:
kld_score.py --pos-cutoff 1024 (sealed convention).

Resume-safe: skips windows in DONE.jsonl; atomic save (tmp+rename) + md5.

dtype rules (verified against modeling_deepseek_v4.py):
  bf16: all F.linear weights (attn projs, compressor/indexer projs, gate
        weight, shared+routed experts, embed, lm_head) and all RMSNorm
        weights (norm returns weight*x -> must stay bf16 for next linear).
  fp32: sinks (torch.cat promotes), position_bias/ape (add promotes),
        hc_* (module .float()s internally), e_score_correction_bias
        (added to fp32 scores), tid2eid int64.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"
SUP = 8192
ATTN_IMPLEMENTATIONS = ("eager", "sdpa")

# e2m1 nibble -> value (mag [0,.5,1,1.5,2,3,4,6], bit3 = sign)
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_VAL = torch.tensor(E2M1_MAG + [-m for m in E2M1_MAG])
# W2 snap: nibble -> code [2,2,2,2,2,3,3,3, 1,1,1,1,1,0,0,0]; levels [-4,-1,1,4]
W2_VAL = torch.tensor([1., 1., 1., 1., 1., 4., 4., 4.,
                       -1., -1., -1., -1., -1., -4., -4., -4.])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def default_attention_implementation():
    """Resolve the rail attention backend without changing sealed defaults."""
    value = os.environ.get("TWOBIN_ATTN_IMPL", "eager")
    if value not in ATTN_IMPLEMENTATIONS:
        raise ValueError(
            f"TWOBIN_ATTN_IMPL must be one of {ATTN_IMPLEMENTATIONS}, got {value!r}"
        )
    return value


def validate_attention_implementation(value):
    """Fail closed on the fast backend after its paired rail cross-check failed."""
    if value == "sdpa" and os.environ.get("TWOBIN_ALLOW_UNSAFE_SDPA") != "1":
        raise RuntimeError(
            "SDPA is research-only: paired 3-window KLD shifted by 81.58%; "
            "set TWOBIN_ALLOW_UNSAFE_SDPA=1 only for an explicitly labeled experiment"
        )
    return value


def deepseek_v4_sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """SDPA equivalent of V4 eager attention, including learned sink logits."""
    groups = int(module.num_key_value_groups)
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    batch, heads, query_len, head_dim = query.shape
    key_len = key.shape[-2]
    if attention_mask is None:
        query_positions = torch.arange(query_len, device=query.device)[:, None]
        key_positions = torch.arange(key_len, device=query.device)[None, :]
        offset = key_len - query_len
        allowed = key_positions <= query_positions + offset
        attention_mask = torch.zeros(
            (1, 1, query_len, key_len), dtype=query.dtype, device=query.device
        ).masked_fill(~allowed.reshape(1, 1, query_len, key_len), float("-inf"))
    elif attention_mask.dtype == torch.bool:
        attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill(
            ~attention_mask, float("-inf")
        )
    else:
        attention_mask = attention_mask.to(dtype=query.dtype, device=query.device)

    attention_mask = attention_mask.expand(batch, heads, query_len, key_len)
    sink_bias = module.sinks.to(dtype=query.dtype, device=query.device).reshape(1, heads, 1, 1)
    sink_bias = sink_bias.expand(batch, heads, query_len, 1)
    attention_mask = torch.cat((attention_mask, sink_bias), dim=-1)

    zero_key = key.new_zeros((batch, heads, 1, head_dim))
    zero_value = value.new_zeros((batch, heads, 1, head_dim))
    key = torch.cat((key, zero_key), dim=-2)
    value = torch.cat((value, zero_value), dim=-2)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous(), None


def install_deepseek_v4_sdpa():
    """Register sink-aware SDPA for this isolated rail process."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as deepseek_v4

    ALL_ATTENTION_FUNCTIONS.register("sdpa", deepseek_v4_sdpa_attention_forward)
    deepseek_v4.DeepseekV4PreTrainedModel._supports_sdpa = True
    deepseek_v4.DeepseekV4ForCausalLM._supports_sdpa = True


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_exclusive_json(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    dfd = os.open(os.path.dirname(path), os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)


def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- shards
class ShardCache:
    """Streams shards from remote over QSFP; keeps <= keep on disk."""

    def __init__(self, remote, cache_dir, keep=3):
        self.remote = remote
        self.dir = cache_dir
        self.keep = keep
        os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.have = {}      # shard -> path (complete)
        self.fetching = {}  # shard -> threading.Event
        self.order = []

    def _fetch(self, shard):
        dst = os.path.join(self.dir, shard)
        if os.path.exists(dst + ".ok"):
            return dst
        free = shutil.disk_usage(self.dir).free
        if free < 8 << 30:
            raise RuntimeError(f"disk guard: only {free>>30}G free")
        cmd = ["rsync", "--inplace", f"{self.remote}/{shard}", dst]
        for attempt in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                open(dst + ".ok", "w").close()
                return dst
            log(f"rsync {shard} failed (try {attempt}): {r.stderr[-200:]}")
            time.sleep(5)
        raise RuntimeError(f"rsync failed for {shard}")

    def prefetch(self, shard):
        with self.lock:
            if shard in self.have or shard in self.fetching:
                return
            ev = threading.Event()
            self.fetching[shard] = ev

        def run():
            try:
                p = self._fetch(shard)
                with self.lock:
                    self.have[shard] = p
                    self.order.append(shard)
            except Exception as e:
                log(f"prefetch {shard}: {e}")
            finally:
                ev.set()
        threading.Thread(target=run, daemon=True).start()

    def _evict_pagecache(self, path):
        try:
            fd = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass

    def get(self, shard):
        for _ in range(3):
            with self.lock:
                if shard in self.have:
                    break
                ev = self.fetching.get(shard)
            if ev is None:
                self.prefetch(shard)
                with self.lock:
                    ev = self.fetching.get(shard)
            if ev is not None:
                ev.wait()
            with self.lock:
                self.fetching.pop(shard, None)
        with self.lock:
            if shard not in self.have:
                raise RuntimeError(f"could not fetch {shard}")
            while len(self.order) > self.keep:
                old = next((s for s in self.order if s != shard), None)
                if old is None:
                    break
                self.order.remove(old)
                p = self.have.pop(old, None)
                if p:
                    self._evict_pagecache(p)
                    for f in (p, p + ".ok"):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
            return self.have[shard]


class LocalSource:
    """Reads shards straight from a local checkpoint dir (host with ckpt)."""

    def __init__(self, ckpt_dir):
        self.dir = ckpt_dir

    def prefetch(self, shard):
        pass

    def get(self, shard):
        return os.path.join(self.dir, shard)

    def evict(self, shard):
        try:
            fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------- dequant
@triton.jit
def _fp8_block128_to_bf16_kernel(
    weight_ptr, scale_ptr, output_ptr,
    n_cols: tl.constexpr, scale_cols: tl.constexpr, total,
    BLOCK: tl.constexpr,
):
    """Exact elementwise equivalent of the sealed FP8 block-128 dequant."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    row = offsets // n_cols
    col = offsets - row * n_cols
    exponent = tl.load(
        scale_ptr + (row // 128) * scale_cols + col // 128,
        mask=mask,
        other=127,
    ).to(tl.float32) - 127.0
    value = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, value * tl.exp2(exponent), mask=mask)


def e8m0(t):
    return torch.exp2(t.view(torch.uint8).to(torch.float32) - 127.0)


def deq_fp8_block(w, s, block=128):
    """fp8 e4m3 [N,K] + e8m0 scale [ceil(N/128),ceil(K/128)] -> bf16 [N,K]."""
    if block != 128:
        raise ValueError(f"fused fp8 kernel requires block=128, got {block}")
    w = w.to(DEV).contiguous()
    s = s.to(DEV).view(torch.uint8).contiguous()
    N, K = w.shape
    out = torch.empty((N, K), dtype=torch.bfloat16, device=DEV)
    total = N * K
    launch_block = 4096
    _fp8_block128_to_bf16_kernel[(triton.cdiv(total, launch_block),)](
        w, s, out, K, s.shape[1], total,
        BLOCK=launch_block, num_warps=8,
    )
    return out


_BYTE_LUT = {}


def byte_lut(kind):
    # byte -> (val_lo_nibble, val_hi_nibble) table [256,2]
    if kind not in _BYTE_LUT:
        vals = E2M1_VAL if kind == "e2m1" else W2_VAL
        b = torch.arange(256)
        t = torch.stack([vals[(b & 0xF)], vals[(b >> 4)]], -1)
        _BYTE_LUT[kind] = t.to(DEV)
    return _BYTE_LUT[kind]


def deq_fp4_block32(wb, sb, kind):
    """packed nibbles [.., N, K/2] u8 + e8m0 [.., N, K/32] -> bf16 [.., N, K].

    Nibble order: low nibble = even k (matches mxfp4 packing / vllm-moet).
    """
    lut = byte_lut(kind)
    vals = lut[wb.long()].flatten(-2)          # [.., N, K]
    sc = e8m0(sb).repeat_interleave(32, -1)    # [.., N, K]
    return (vals * sc).to(torch.bfloat16)


# ---------------------------------------------------------------- planes
class PlaneSource:
    """Shipped plane bytes (moe_w2/w3_planes wire format) for routed experts.

    Auto-detects codebook width per layer from planes13 byte size:
      N13*K13/4   bytes/expert -> 2-bit (W2 fragment-major)
      N13*K13*3/8 bytes/expert -> 3-bit (W3 384B-block)
    LUT: meta.json "lut" when present, else the codebook default.
    """

    def __init__(self, planes_dir):
        import numpy as np
        import planes_unpack as pu
        self.np, self.pu = np, pu
        self.dir = os.path.expanduser(planes_dir)
        assert os.path.isdir(self.dir), self.dir

    def layer(self, L):
        np, pu = self.np, self.pu
        meta = json.load(open(os.path.join(
            self.dir, f"layer_{L:03d}.meta.json")))
        m = {k: np.load(os.path.join(self.dir, f"layer_{L:03d}.{k}.npy"),
                        mmap_mode="r")
             for k in ("planes13", "planes2", "sc13", "sc2")}
        E, N13, K13 = meta["E"], meta["N13"], meta["K13"]
        N2, K2 = meta["N2"], meta["K2"]
        bpe13 = m["planes13"].shape[1]
        if bpe13 == N13 * K13 // 4:
            unpack, levels = pu.unpack_fragment_major, pu.W2_LEVELS
        elif bpe13 == N13 * K13 * 3 // 8:
            unpack, levels = pu.unpack_w3_plane, pu.W3_LEVELS
        else:
            raise RuntimeError(f"layer {L}: unknown plane width {bpe13}")
        if "lut" in meta:
            levels = torch.tensor(meta["lut"], dtype=torch.float32)
        assert len(m["planes13"]) == E

        def expert(e, which):
            np_ = self.np
            if which == "13":
                pl, sc, N, K = m["planes13"], m["sc13"], N13, K13
            else:
                pl, sc, N, K = m["planes2"], m["sc2"], N2, K2
            plane = torch.from_numpy(np_.asarray(pl[e]).copy()).to(DEV)
            scb = torch.from_numpy(np_.asarray(sc[e]).copy()).to(DEV)
            codes = unpack(plane, N, K)
            sb = pu.unpack_scales(scb, N, K // 32)
            return pu.plane_dequant(codes, sb, levels).to(torch.bfloat16)

        return expert, (E, N13, K13, N2, K2)


# ---------------------------------------------------------------- loading
def fill_plane_experts(planes, layer, gate_up, down):
    """Use a source bulk-fill path when available, preserving scalar fallback."""
    bulk_fill = getattr(planes, "fill_layer", None)
    if callable(bulk_fill):
        bulk_fill(layer, gate_up, down)
        return
    expert, dims = planes.layer(layer)
    assert dims == (256, 4096, 4096, 4096, 2048), dims
    for expert_id in range(256):
        gate_up[expert_id] = expert(expert_id, "13")
        down[expert_id] = expert(expert_id, "2")


def build_layer_sd(L, wm, get_tensor, mode, planes=None):
    """Native ckpt keys for layer L -> HF-named tensor dict (on DEV)."""
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
        return deq_fp8_block(T(name + ".weight"), T(name + ".scale"))

    f32 = lambda name: T(name).to(DEV).to(torch.float32)
    bf = lambda name: T(name).to(DEV).to(torch.bfloat16)

    # attention core (fp8 -> bf16)
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

    # compressor (CSA/HCA layers)
    if has("attn.compressor.wkv.weight"):
        sd["self_attn.compressor.position_bias"] = f32("attn.compressor.ape")
        sd["self_attn.compressor.kv_norm.weight"] = bf("attn.compressor.norm.weight")
        sd["self_attn.compressor.kv_proj.weight"] = bf("attn.compressor.wkv.weight")
        sd["self_attn.compressor.gate_proj.weight"] = bf("attn.compressor.wgate.weight")
    # indexer (CSA layers)
    if has("attn.indexer.wq_b.weight"):
        idx = "self_attn.compressor.indexer."
        sd[idx + "position_bias"] = f32("attn.indexer.compressor.ape")
        sd[idx + "kv_norm.weight"] = bf("attn.indexer.compressor.norm.weight")
        sd[idx + "kv_proj.weight"] = bf("attn.indexer.compressor.wkv.weight")
        sd[idx + "gate_proj.weight"] = bf("attn.indexer.compressor.wgate.weight")
        sd[idx + "q_b_proj.weight"] = fp8("attn.indexer.wq_b")
        sd[idx + "scorer.weights_proj.weight"] = bf("attn.indexer.weights_proj.weight")

    # router
    sd["mlp.gate.weight"] = bf("ffn.gate.weight")
    if has("ffn.gate.tid2eid"):
        sd["mlp.gate.tid2eid"] = T("ffn.gate.tid2eid").to(DEV)
    if has("ffn.gate.bias"):
        sd["mlp.gate.e_score_correction_bias"] = f32("ffn.gate.bias")

    # hyper-connections (module floats internally; keep fp32)
    sd["attn_hc.fn"] = f32("hc_attn_fn")
    sd["attn_hc.base"] = f32("hc_attn_base")
    sd["attn_hc.scale"] = f32("hc_attn_scale")
    sd["ffn_hc.fn"] = f32("hc_ffn_fn")
    sd["ffn_hc.base"] = f32("hc_ffn_base")
    sd["ffn_hc.scale"] = f32("hc_ffn_scale")

    # shared experts (fp8, teacher path in BOTH modes)
    sd["mlp.shared_experts.gate_proj.weight"] = fp8("ffn.shared_experts.w1")
    sd["mlp.shared_experts.up_proj.weight"] = fp8("ffn.shared_experts.w3")
    sd["mlp.shared_experts.down_proj.weight"] = fp8("ffn.shared_experts.w2")

    # routed experts (fp4 block-32; W2 snap in cand mode; shipped bytes
    # in planes mode)
    E = 256
    gu = torch.empty(E, 4096, 4096, dtype=torch.bfloat16, device=DEV)
    dn = torch.empty(E, 4096, 2048, dtype=torch.bfloat16, device=DEV)
    if mode == "planes":
        fill_plane_experts(planes, L, gu, dn)
        # ckpt expert keys are intentionally unread; mark consumed for gate
        for k in keys:
            if ".ffn.experts." in k:
                consumed.add(k)
    else:
        kind = "w2" if mode == "w2" else "e2m1"
        CH = 8
        for e0 in range(0, E, CH):
            es = range(e0, min(e0 + CH, E))
            for wname, dst, rows in (("w1", gu, slice(0, 2048)),
                                     ("w3", gu, slice(2048, 4096)),
                                     ("w2", dn, slice(0, 4096))):
                wb = torch.stack([T(f"ffn.experts.{e}.{wname}.weight").view(torch.uint8)
                                  for e in es]).to(DEV)
                sb = torch.stack([T(f"ffn.experts.{e}.{wname}.scale").view(torch.uint8)
                                  for e in es]).to(DEV)
                dst[e0:e0 + len(es), rows] = deq_fp4_block32(wb, sb, kind)
                del wb, sb
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn

    missed = set(keys) - consumed
    if missed:
        raise RuntimeError(f"layer {L}: unconsumed ckpt keys: {sorted(missed)[:8]}")
    return sd


def materialize_layer(model, L, sd, config):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4RotaryEmbedding)
    lay = model.model.layers[L]
    missing, unexpected = lay.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"layer {L} unexpected: {unexpected[:8]}")
    # rebuild rotary submodules (buffers were meta; deterministic from config)
    for name, mod in list(lay.named_modules()):
        if isinstance(mod, DeepseekV4RotaryEmbedding):
            parent = lay.get_submodule(name.rsplit(".", 1)[0]) if "." in name else lay
            setattr(parent, name.rsplit(".", 1)[-1],
                    DeepseekV4RotaryEmbedding(config).to(DEV))
    bad = [n for n, p in lay.named_parameters() if p.is_meta]
    bad += [n for n, b in lay.named_buffers() if b.is_meta]
    if bad:
        raise RuntimeError(f"layer {L} still meta: {bad[:8]}")
    return lay


def dematerialize_layer(model, L):
    lay = model.model.layers[L]
    for mod in lay.modules():
        for n, p in list(mod._parameters.items()):
            if p is not None:
                mod._parameters[n] = torch.nn.Parameter(
                    torch.empty(p.shape, device="meta", dtype=p.dtype),
                    requires_grad=False)
        for n, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[n] = torch.empty(
                    b.shape, device="meta", dtype=b.dtype)
    # References above are gone; keep allocator blocks for same-stream reuse.
    # The existing chunk-end empty_cache() remains the bounded-memory flush.


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bf16", "w2", "planes"), default="bf16")
    ap.add_argument("--planes-dir", default=None,
                    help="planes mode: dir of layer_NNN.{meta.json,planes13,"
                         "planes2,sc13,sc2} in the shipped wire format "
                         "(W2 or W3 auto-detected; GPTQ dirs drop in here)")
    ap.add_argument("--meta-dir", default=os.path.expanduser(
        "$HOME/run-bundles/DS4_TEACHER/ckpt_cache"))
    ap.add_argument(
        "--remote",
        default=os.environ.get("DS4_MODEL_REMOTE"),
        help="rsync source for the model; defaults to $DS4_MODEL_REMOTE",
    )
    ap.add_argument("--corpus", default=os.path.expanduser(
        "$HOME/run-bundles/DS4_TEACHER/static/windows_ds4_eval.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir", default=None, help="teacher dir (w2 mode)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument(
        "--attn-implementation",
        choices=ATTN_IMPLEMENTATIONS,
        default=default_attention_implementation(),
        help="attention backend; defaults to $TWOBIN_ATTN_IMPL or sealed eager",
    )
    ap.add_argument("--limit-layers", type=int, default=0, help="debug only")
    ap.add_argument("--cand-pos-limit", type=int, default=0,
                    help="w2 mode: store only first P positions per window "
                         "(disk relief; exactly equivalent for kld_score "
                         "--pos-cutoff <= P since scorer takes "
                         "T=min(ref_T,cand_T,cutoff)). 0 = full window.")
    ap.add_argument("--windows", default=None, help="csv of window ids (debug)")
    ap.add_argument("--local-dir", default=None,
                    help="local checkpoint dir (skip QSFP streaming)")
    ap.add_argument("--shard-buf", default=os.path.expanduser(
        "$HOME/run-bundles/DS4_TEACHER/shard_buf"))
    ap.add_argument("--keep-shards", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    validate_attention_implementation(a.attn_implementation)

    assert a.mode == "bf16" or a.ref_dir, "--ref-dir required in cand modes"
    assert a.mode != "planes" or a.planes_dir, "--planes-dir required"
    planes = PlaneSource(a.planes_dir) if a.mode == "planes" else None
    loader_progress_path = os.environ.get("FULL512_LOADER_PROGRESS_PATH")
    loader_sha256_expected = os.environ.get("FULL512_LOADER_SHA256")
    loader_input_sha256 = os.environ.get("FULL512_LOADER_INPUT_SHA256")
    loader_sentinel_path = os.environ.get("FULL512_LOADER_SENTINEL_PATH")
    loader_task_id = os.environ.get("FULL512_TASK_ID")
    os.makedirs(a.out, exist_ok=True)
    done_path = os.path.join(a.out, "DONE.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            try:
                done.add(json.loads(line)["win"])
            except Exception:
                pass

    corpus = json.load(open(a.corpus))
    if a.windows:
        todo = [int(x) for x in a.windows.split(",") if int(x) not in done]
    else:
        todo = [k for k in range(a.start, min(a.start + a.count, len(corpus)))
                if k not in done]
    if not todo:
        log("nothing to do")
        return 0
    log(f"mode={a.mode} todo={len(todo)} windows out={a.out}")

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4RotaryEmbedding)
    from transformers.cache_utils import DynamicCache
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(a.meta_dir)
    wm = json.load(open(os.path.join(a.meta_dir,
                   "model.safetensors.index.json")))["weight_map"]

    log(
        f"transformers {transformers.__version__} torch {torch.__version__} "
        f"attn={a.attn_implementation} mb={a.mb}"
    )
    if a.attn_implementation == "sdpa":
        install_deepseek_v4_sdpa()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation=a.attn_implementation)
    model.eval()

    if a.local_dir:
        cache = LocalSource(os.path.expanduser(a.local_dir))
    else:
        cache = ShardCache(a.remote, a.shard_buf, keep=a.keep_shards)
    handles = {}

    def get_tensor(name):
        shard = wm[name]
        path = cache.get(shard)
        if path not in handles:
            while len(handles) >= 3:
                handles.pop(next(iter(handles)))
            handles[path] = safe_open(path, framework="pt")
        return handles[path].get_tensor(name)

    # static parts
    log("materializing embed/head/norm/hc_head")
    for s in (wm["embed.weight"], wm["head.weight"]):
        cache.prefetch(s)
    model.model.embed_tokens.weight = torch.nn.Parameter(
        get_tensor("embed.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.lm_head.weight = torch.nn.Parameter(
        get_tensor("head.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.model.norm.weight = torch.nn.Parameter(
        get_tensor("norm.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.model.hc_head.hc_fn = torch.nn.Parameter(
        get_tensor("hc_head_fn").to(DEV).to(torch.float32), requires_grad=False)
    model.model.hc_head.hc_base = torch.nn.Parameter(
        get_tensor("hc_head_base").to(DEV).to(torch.float32), requires_grad=False)
    model.model.hc_head.hc_scale = torch.nn.Parameter(
        get_tensor("hc_head_scale").to(DEV).to(torch.float32), requires_grad=False)
    model.model.rotary_emb = DeepseekV4RotaryEmbedding(config).to(DEV)

    NL = a.limit_layers or config.num_hidden_layers
    layer_shards = {L: sorted({wm[k] for k in wm if k.startswith(f"layers.{L}.")})
                    for L in range(NL)}

    hb = os.path.join(a.out, "HEARTBEAT")
    t_run = time.time()
    for c0 in range(0, len(todo), a.chunk):
        wins = todo[c0:c0 + a.chunk]
        t_chunk = time.time()
        log(f"chunk {c0//a.chunk}: windows {wins[0]}..{wins[-1]} (n={len(wins)})")
        ids = torch.full((len(wins), 2048), 1, dtype=torch.long)
        rlens = []
        for i, k in enumerate(wins):
            t = corpus[k]["token_ids"]
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            rlens.append(corpus[k]["real_len"])
        ids = ids.to(DEV)
        pos = torch.arange(2048, device=DEV).unsqueeze(0)

        mbs = [slice(i, min(i + a.mb, len(wins)))
               for i in range(0, len(wins), a.mb)]
        stream_count = int(os.environ.get("TWOBIN_STREAMS", "1"))
        if stream_count < 1:
            raise ValueError("TWOBIN_STREAMS must be positive")
        streams = [torch.cuda.Stream() for _ in range(min(stream_count, len(mbs)))]
        log(f"forward streams={len(streams)} microbatches={len(mbs)} mb={a.mb}")
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids)
            pe = {
                "main": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="main"),
                "compress": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="compress"),
            }
            stateless_cache = os.environ.get("TWOBIN_STATELESS_CACHE", "0") == "1"
            reuse_mask = os.environ.get("TWOBIN_REUSE_MASK", "0") == "1"
            caches = [None for _ in mbs] if stateless_cache else [
                DynamicCache(config=config) for _ in mbs
            ]
            masks, hidden = [], []
            for mi, s in enumerate(mbs):
                if reuse_mask and masks:
                    masks.append(masks[0])
                else:
                    masks.append(create_sliding_window_causal_mask(
                        config=config, inputs_embeds=embeds[s],
                        attention_mask=None, past_key_values=caches[mi],
                        position_ids=pos))
                hidden.append(embeds[s].unsqueeze(2).expand(
                    -1, -1, config.hc_mult, -1).contiguous())
            log(f"stateless_cache={int(stateless_cache)} reuse_mask={int(reuse_mask)}")
            del embeds

            overlap_layers = os.environ.get("TWOBIN_LAYER_OVERLAP", "0") == "1"
            prep_executor = ThreadPoolExecutor(max_workers=1) if overlap_layers else None
            prep_stream = torch.cuda.Stream() if overlap_layers else None

            def prepare_layer(layer, stream=None):
                """Materialize one state dict, optionally on an overlap CUDA stream."""
                started = time.time()
                for shard in layer_shards[layer]:
                    cache.prefetch(shard)
                if layer + 1 < NL:
                    for shard in layer_shards[layer + 1]:
                        cache.prefetch(shard)
                if stream is None:
                    state = build_layer_sd(layer, wm, get_tensor, a.mode, planes)
                    return state, None, time.time() - started
                with torch.cuda.stream(stream):
                    state = build_layer_sd(layer, wm, get_tensor, a.mode, planes)
                    ready = torch.cuda.Event()
                    ready.record(stream)
                return state, ready, time.time() - started

            prepared = prepare_layer(0)
            future = None
            try:
                for L in range(NL):
                    wait_started = time.time()
                    if future is not None:
                        sd, ready, prep_wall = future.result()
                        ready.synchronize()
                    elif L == 0:
                        sd, ready, prep_wall = prepared
                    else:
                        sd, ready, prep_wall = prepare_layer(L)
                    load_wait = time.time() - wait_started
                    lay = materialize_layer(model, L, sd, config)
                    default_stream = torch.cuda.current_stream()
                    for tensor in sd.values():
                        if tensor.is_cuda:
                            tensor.record_stream(default_stream)
                    del sd
                    if prep_executor is not None and L + 1 < NL:
                        future = prep_executor.submit(prepare_layer, L + 1, prep_stream)
                    else:
                        future = None
                    t1 = time.time()
                    if len(streams) == 1:
                        for mi, s in enumerate(mbs):
                            hidden[mi] = lay(
                                hidden[mi], position_embeddings=pe, position_ids=pos,
                                attention_mask=masks[mi], input_ids=ids[s],
                                past_key_values=caches[mi])
                    else:
                        default_stream = torch.cuda.current_stream()
                        for stream in streams:
                            stream.wait_stream(default_stream)
                        for mi, s in enumerate(mbs):
                            with torch.cuda.stream(streams[mi % len(streams)]):
                                hidden[mi] = lay(
                                    hidden[mi], position_embeddings=pe, position_ids=pos,
                                    attention_mask=masks[mi], input_ids=ids[s],
                                    past_key_values=caches[mi])
                        torch.cuda.synchronize()
                    dematerialize_layer(model, L)
                    t2 = time.time()
                    if overlap_layers:
                        load_text = f"wait {load_wait:5.1f}s prep {prep_wall:5.1f}s"
                    else:
                        load_text = f"load {prep_wall:5.1f}s"
                    log(f"  L{L:02d} {load_text} fwd {t2-t1:5.1f}s "
                        f"gpu_res {torch.cuda.memory_reserved()>>30}G")
                    with open(hb, "w") as f:
                        json.dump({"chunk": c0 // a.chunk, "layer": L,
                                   "ts": time.time()}, f)
            finally:
                if prep_executor is not None:
                    prep_executor.shutdown(wait=True, cancel_futures=True)

            loader_progress_sha256 = None
            loader_chunk_receipt_sha256 = None
            loader_sentinel_sha256 = None
            if loader_progress_path:
                if not all((loader_sha256_expected, loader_input_sha256,
                            loader_sentinel_path, loader_task_id)):
                    raise RuntimeError("binding loader receipt environment incomplete")
                if not os.path.isfile(loader_progress_path) or not os.path.isfile(loader_sentinel_path):
                    raise RuntimeError("binding loader progress/sentinel receipt missing")
                loader_progress_sha256 = sha256(loader_progress_path)
                loader_sentinel_sha256 = sha256(loader_sentinel_path)
                with open(loader_progress_path) as handle:
                    progress = json.load(handle)
                chunk_index = wins[0] // 64
                if (
                    wins != list(range(chunk_index * 64, (chunk_index + 1) * 64))
                    or chunk_index not in range(1, 8)
                    or progress.get("completed_layers") != list(range(43)) * (chunk_index + 1)
                    or progress.get("mmap_completed_layers") != list(range(43)) * chunk_index
                    or progress.get("completed_chunks") != chunk_index + 1
                    or progress.get("current_chunk_layers") != []
                    or progress.get("local_stage_retired") is not True
                    or progress.get("mmap_loader_mode") != "torch-mmap"
                    or progress.get("mmap_loader_sha256") != loader_sha256_expected
                    or progress.get("mmap_input_identity_sha256") != loader_input_sha256
                ):
                    raise RuntimeError(f"binding loader chunk progress drift chunk={chunk_index}")
                chunk_receipt_path = os.path.join(
                    os.path.dirname(loader_progress_path),
                    f"ARM4_MMAP_CHUNK_{wins[0]:03d}_{wins[-1]:03d}.json",
                )
                chunk_receipt = {
                    "schema": "genesis-arm4-mmap-chunk-v1",
                    "status": "PASS_ON_PATH",
                    "task_id": loader_task_id,
                    "mode": "torch-mmap",
                    "loader_sha256": loader_sha256_expected,
                    "input_identity_sha256": loader_input_sha256,
                    "sentinel_sha256": loader_sentinel_sha256,
                    "window_ids": wins,
                    "stream_completed_layers": progress["completed_layers"],
                    "mmap_completed_layers": progress["mmap_completed_layers"],
                    "loader_progress_sha256": loader_progress_sha256,
                    "created_unix": time.time(),
                }
                atomic_exclusive_json(chunk_receipt_path, chunk_receipt)
                loader_chunk_receipt_sha256 = sha256(chunk_receipt_path)

            # readout
            for mi, s in enumerate(mbs):
                h = model.model.norm(model.model.hc_head(hidden[mi]))
                for j in range(h.shape[0]):
                    k = wins[s.start + j]
                    rl = rlens[s.start + j]
                    P = min(rl, a.cand_pos_limit) if a.cand_pos_limit else rl
                    prefix_logits = os.environ.get("TWOBIN_PREFIX_LOGITS", "0") == "1"
                    if prefix_logits and (a.mode != "planes" or not a.cand_pos_limit):
                        raise RuntimeError("prefix logits require candidate planes mode and a cutoff")
                    logit_rows = P if prefix_logits else rl
                    logits = model.lm_head(h[j, :logit_rows].to(torch.bfloat16)).float()
                    lp = torch.log_softmax(logits, dim=-1)
                    npos = min(1024, rl - 1)
                    tgt = ids[s.start + j, 1:npos + 1]
                    nll = -lp[:npos].gather(1, tgt.unsqueeze(1)).mean().item()
                    if a.mode == "bf16":
                        lp_s, idx = torch.sort(lp, dim=-1, descending=True)
                        obj = {"idx": idx[:, :SUP].to(torch.int32).cpu(),
                               "logprob": lp_s[:, :SUP].to(torch.float16).cpu()}
                        fname = f"t8192_win{k}.pt"
                        del lp_s, idx
                    else:
                        teacher_cache_dir = os.environ.get("TWOBIN_TEACHER_CACHE")
                        ref = torch.load(
                            os.path.join(a.ref_dir, f"t8192_win{k}.pt"),
                            map_location="cpu" if teacher_cache_dir else DEV,
                            mmap=bool(teacher_cache_dir),
                            weights_only=bool(teacher_cache_dir),
                        )
                        ridx = ref["idx"].long()[:P].to(DEV)
                        q_lp_at_ref = lp[:P].gather(1, ridx).to(
                            torch.float16
                        ).cpu()
                        if os.environ.get("TWOBIN_KLD_STREAM_OUT", "0") == "1":
                            # Preserve the sealed scorer's exact fp16-bank -> fp32
                            # support-renormalized arithmetic while avoiding the
                            # ~8 GiB q8192 bank.  One float32 KL vector per window
                            # is sufficient for exact per-class tails/quantiles.
                            ref_lp = ref["logprob"][:P, :SUP].to(
                                torch.float16
                            ).cpu().float()
                            cand_lp = q_lp_at_ref.float()
                            teacher_cache_row = None
                            if teacher_cache_dir:
                                teacher_cache_row = torch.load(
                                    os.path.join(teacher_cache_dir, f"teacher_win{k}.pt"),
                                    map_location="cpu", mmap=True, weights_only=True,
                                )
                                if (teacher_cache_row.get("win") != k
                                        or teacher_cache_row.get("support") != SUP
                                        or teacher_cache_row.get("cutoff") != 1024):
                                    raise RuntimeError(f"teacher cache row contract drift win={k}")
                                logz = teacher_cache_row["logz"][:P]
                                teacher_p = teacher_cache_row["teacher_p"][:P]
                                lp_n = ref_lp - logz
                            else:
                                lp_n = ref_lp - torch.logsumexp(
                                    ref_lp, dim=-1, keepdim=True
                                )
                                teacher_p = torch.exp(lp_n)
                            lq_n = cand_lp - torch.logsumexp(
                                cand_lp, dim=-1, keepdim=True
                            )
                            kl = torch.sum(
                                teacher_p * (lp_n - lq_n), dim=-1
                            ).contiguous()
                            obj = {
                                "kld": kl,
                                "win": int(k),
                                "support": int(SUP),
                                "cutoff": int(P),
                                "direction": "KL(teacher||candidate)",
                                "support_policy": (
                                    "teacher top-k; fp16 banks promoted to fp32; "
                                    "teacher and candidate renormalized on support"
                                ),
                            }
                            fname = f"kld_win{k}.pt"
                            del ref_lp, cand_lp, lp_n, lq_n, kl, teacher_p
                            if teacher_cache_row is not None:
                                del teacher_cache_row, logz
                        else:
                            obj = {
                                "q_lp_at_ref": q_lp_at_ref,
                                "q_argmax": lp[:P].argmax(-1).to(
                                    torch.int32
                                ).cpu(),
                            }
                            fname = f"q8192_win{k}.pt"
                        del ref, ridx, q_lp_at_ref
                    if loader_progress_sha256 is not None:
                        obj.update({
                            "loader_mode": "torch-mmap",
                            "loader_sha256": loader_sha256_expected,
                            "loader_sentinel_sha256": loader_sentinel_sha256,
                            "input_identity_sha256": loader_input_sha256,
                            "loader_progress_sha256": loader_progress_sha256,
                            "loader_chunk_receipt_sha256": loader_chunk_receipt_sha256,
                        })
                    out_p = os.path.join(a.out, fname)
                    torch.save(obj, out_p + ".tmp")
                    os.replace(out_p + ".tmp", out_p)
                    jrow(done_path, win=k, file=fname, md5=md5(out_p),
                         real_len=rl, npos=npos, nll1024=round(nll, 5),
                         mode=a.mode, tag=a.tag,
                         loader_progress_sha256=loader_progress_sha256,
                         loader_sentinel_sha256=loader_sentinel_sha256,
                         loader_chunk_receipt_sha256=loader_chunk_receipt_sha256)
                    del logits, lp, obj
                del h
            del hidden, caches, masks
            torch.cuda.empty_cache()
        log(f"chunk done in {(time.time()-t_chunk)/60:.1f} min "
            f"({len(wins)} windows)")

    log(f"ALL DONE {len(todo)} windows in {(time.time()-t_run)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())