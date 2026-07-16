#!/usr/bin/env python3
"""DS4-Flash teacher rail + W2 candidate rail (t_394f19e7).

Teacher mode (--mode bf16):
  out/t8192_win<k>.pt  {"idx": int32 [T,8192] desc by teacher lp,
                        "logprob": fp16 [T,8192] full-softmax lp}
  Teacher = source ckpt dequantized to bf16 (fp8 e4m3 block-128 dense,
  fp4 e2m1 block-32 e8m0 routed experts), HF deepseek_v4 eager graph,
  layer-streamed (one materialized layer at a time), shards streamed
  over QSFP from spark-6.

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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import torch

DEV = "cuda"
SUP = 8192

# e2m1 nibble -> value (mag [0,.5,1,1.5,2,3,4,6], bit3 = sign)
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_VAL = torch.tensor(E2M1_MAG + [-m for m in E2M1_MAG])
# W2 snap: nibble -> code [2,2,2,2,2,3,3,3, 1,1,1,1,1,0,0,0]; levels [-4,-1,1,4]
W2_VAL = torch.tensor([1., 1., 1., 1., 1., 4., 4., 4.,
                       -1., -1., -1., -1., -1., -4., -4., -4.])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
def e8m0(t):
    return torch.exp2(t.view(torch.uint8).to(torch.float32) - 127.0)


def deq_fp8_block(w, s, block=128):
    """fp8 e4m3 [N,K] + e8m0 scale [ceil(N/128),ceil(K/128)] -> bf16 [N,K]."""
    w = w.to(DEV)
    sc = e8m0(s.to(DEV))
    N, K = w.shape
    sc = sc.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (w.to(torch.float32) * sc).to(torch.bfloat16)


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


# ---------------------------------------------------------------- loading
def build_layer_sd(L, wm, get_tensor, mode):
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

    # routed experts (fp4 block-32; W2 snap in cand mode)
    kind = "w2" if mode == "w2" else "e2m1"
    E = 256
    gu = torch.empty(E, 4096, 4096, dtype=torch.bfloat16, device=DEV)
    dn = torch.empty(E, 4096, 2048, dtype=torch.bfloat16, device=DEV)
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
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bf16", "w2"), default="bf16")
    ap.add_argument("--meta-dir", default=os.path.expanduser(
        "~/missions/DS4_TEACHER/ckpt_cache"))
    ap.add_argument(
        "--remote",
        default=os.environ.get("DS4_MODEL_REMOTE"),
        help="rsync source for the model; defaults to $DS4_MODEL_REMOTE",
    )
    ap.add_argument("--corpus", default=os.path.expanduser(
        "~/missions/DS4_TEACHER/static/windows_ds4_eval.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-dir", default=None, help="teacher dir (w2 mode)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--mb", type=int, default=4)
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
        "~/missions/DS4_TEACHER/shard_buf"))
    ap.add_argument("--keep-shards", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    assert a.mode == "bf16" or a.ref_dir, "--ref-dir required in w2 mode"
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

    log(f"transformers {transformers.__version__} torch {torch.__version__}")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation="eager")
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
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids)
            pe = {
                "main": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="main"),
                "compress": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="compress"),
            }
            caches = [DynamicCache(config=config) for _ in mbs]
            masks, hidden = [], []
            for mi, s in enumerate(mbs):
                masks.append(create_sliding_window_causal_mask(
                    config=config, inputs_embeds=embeds[s],
                    attention_mask=None, past_key_values=caches[mi],
                    position_ids=pos))
                hidden.append(embeds[s].unsqueeze(2).expand(
                    -1, -1, config.hc_mult, -1).contiguous())
            del embeds

            for L in range(NL):
                t0 = time.time()
                for sh in layer_shards[L]:
                    cache.prefetch(sh)
                if L + 1 < NL:
                    for sh in layer_shards[L + 1]:
                        cache.prefetch(sh)
                sd = build_layer_sd(L, wm, get_tensor, a.mode)
                t1 = time.time()
                lay = materialize_layer(model, L, sd, config)
                del sd
                for mi, s in enumerate(mbs):
                    hidden[mi] = lay(
                        hidden[mi], position_embeddings=pe, position_ids=pos,
                        attention_mask=masks[mi], input_ids=ids[s],
                        past_key_values=caches[mi])
                dematerialize_layer(model, L)
                t2 = time.time()
                log(f"  L{L:02d} load {t1-t0:5.1f}s fwd {t2-t1:5.1f}s "
                    f"gpu_res {torch.cuda.memory_reserved()>>30}G")
                with open(hb, "w") as f:
                    json.dump({"chunk": c0 // a.chunk, "layer": L,
                               "ts": time.time()}, f)

            # readout
            for mi, s in enumerate(mbs):
                h = model.model.norm(model.model.hc_head(hidden[mi]))
                for j in range(h.shape[0]):
                    k = wins[s.start + j]
                    rl = rlens[s.start + j]
                    logits = model.lm_head(h[j, :rl].to(torch.bfloat16)).float()
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
                        ref = torch.load(os.path.join(
                            a.ref_dir, f"t8192_win{k}.pt"), map_location=DEV)
                        P = min(rl, a.cand_pos_limit) if a.cand_pos_limit else rl
                        ridx = ref["idx"].long()[:P]
                        obj = {"q_lp_at_ref": lp[:P].gather(1, ridx).to(
                                   torch.float16).cpu(),
                               "q_argmax": lp[:P].argmax(-1).to(torch.int32).cpu()}
                        fname = f"q8192_win{k}.pt"
                        del ref, ridx
                    out_p = os.path.join(a.out, fname)
                    torch.save(obj, out_p + ".tmp")
                    os.replace(out_p + ".tmp", out_p)
                    jrow(done_path, win=k, file=fname, md5=md5(out_p),
                         real_len=rl, npos=npos, nll1024=round(nll, 5),
                         mode=a.mode, tag=a.tag)
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
