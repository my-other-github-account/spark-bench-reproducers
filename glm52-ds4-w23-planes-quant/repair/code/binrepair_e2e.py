#!/usr/bin/env python3
"""BINREPAIR pilot (t_2956f863): end-to-end KL repair of the k4096-menu
IQ3_BIN artifact with TRAINABLE vq3b codebooks.

Student = DeepseekV4 eager graph (sealed lp4_train.Student assembly) with
routed experts dequanted per the IQ3_BIN target manifest:
  base LP4_PACK wire bytes  +  IQ3_BIN delta pack (w3->vqa, vqa->ternary)
  +  vq3u k4096/d4 planes (w3->vq3b, fp4->vq3b; 14,211 unit-projections).

Trainable: per pilot layer, the vq3b codebooks cb13/cb2 (each [4096,4]
fp16 wire; fp32 master -> fp16 STE, exact wire round-trip at init).
Codes / scales / every other tier stay frozen wire bytes.

Loss: scorer-exact support-renormalized KL(P_ref,S || Q_S) on teacher
top-8192 rows, 1024-token truncations (same metric the IQ3_BIN ledger
reports per window).

Checkpoints are keyed to a codes hash (md5 over codes13/sc13/codes2/sc2 of
each pilot layer's plane) + the target-assignment md5; resume refuses on
mismatch.

Env:
  BR_MANIFEST    target manifest json (required)
  BR_DELTA_DIR   IQ3_BIN delta pack dir (required, needs DELTA_PACK.COMPLETE)
  BR_VQ3B_DIR    dir with vq3u_layer_NNN.pt for ALL 43 layers (required)
  BR_TRAINABLE   pilot layers, default "23,33,41"
  BR_TRAIN / BR_PROBE   comma window lists (eval corpus indices)
  BR_STEPS=48 BR_LR=1e-2 BR_BATCH=2 BR_PROBE_EVERY=8
  BR_OUTDIR BR_TAG BR_TEACH BR_CORPUS
  BR_REF_KLD     optional json {win: ledger kl_mean} soft cross-check
  BR_GRADCHECK=1 op-level FD check before training
  BR_CACHE_ONLY=0 BR_CACHE_BATCH=4
  BR_MAX_HOURS=12 BR_EARLY_STOP=3
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.expanduser("~/missions/LP4_REPAIR/src_lp4"))
sys.path.insert(0, os.path.expanduser("~/missions/LP4_REPAIR/src"))
import lp4_train as T                     # noqa: E402  sealed trainer parts
import lp4_pack as pk                     # noqa: E402  verified batched deq
import t8192_ds4_build_v3 as v3           # noqa: E402  fp4 dequant

DEV = T.DEV
E, N13, K13, N2, K2 = T.E, T.N13, T.K13, T.N2, T.K2
DEQ_CHUNK = 4

MANIFEST = Path(os.path.expanduser(os.environ["BR_MANIFEST"]))
DELTA_DIR = Path(os.path.expanduser(os.environ["BR_DELTA_DIR"]))
VQ3B_DIR = Path(os.path.expanduser(os.environ["BR_VQ3B_DIR"]))
TRAINABLE = [int(x) for x in
             os.environ.get("BR_TRAINABLE", "23,33,41").split(",")]
TRAIN_WINS = [int(x) for x in os.environ["BR_TRAIN"].split(",")]
PROBE_WINS = [int(x) for x in os.environ["BR_PROBE"].split(",")]
STEPS = int(os.environ.get("BR_STEPS", "48"))
LR = float(os.environ.get("BR_LR", "1e-2"))
BATCH = int(os.environ.get("BR_BATCH", "2"))
PROBE_EVERY = int(os.environ.get("BR_PROBE_EVERY", "8"))
OUTDIR = Path(os.path.expanduser(os.environ.get(
    "BR_OUTDIR", "~/missions/BINREPAIR_t_2956f863/out")))
TAG = os.environ.get("BR_TAG", "pilot1")
REF_KLD_PATH = os.environ.get("BR_REF_KLD", "")
GRADCHECK = os.environ.get("BR_GRADCHECK", "1") == "1"
CACHE_ONLY = os.environ.get("BR_CACHE_ONLY", "0") == "1"
CACHE_BATCH = int(os.environ.get("BR_CACHE_BATCH", "4"))
MAX_HOURS = float(os.environ.get("BR_MAX_HOURS", "12"))
EARLY_STOP = int(os.environ.get("BR_EARLY_STOP", "3"))

if os.environ.get("BR_TEACH"):
    T.TEACH = os.path.expanduser(os.environ["BR_TEACH"])
if os.environ.get("BR_CORPUS"):
    T.CORPUS = os.path.expanduser(os.environ["BR_CORPUS"])

FIRST_TRAIN = min(TRAINABLE)
PREFIX = OUTDIR / f"BINREPAIR_{TAG}"
JLOG = Path(str(PREFIX) + ".jsonl")
STATUS = Path(str(PREFIX) + ".status.json")
LATEST = Path(str(PREFIX) + ".latest.pt")
BEST = Path(str(PREFIX) + ".best.pt")
FINAL = Path(str(PREFIX) + ".final.json")
ACTCACHE_DIR = OUTDIR / f"BR_ACTCACHE_L{FIRST_TRAIN:03d}"

MAN = json.loads(MANIFEST.read_text())
AMD5 = hashlib.md5(json.dumps(
    MAN["assignment"], sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
BASE_MAN = json.loads(Path(os.path.expanduser(
    "~/missions/LP4_REPAIR/static/LP4_MANIFEST_S7LOCAL.json")).read_text())
CACHE_ID = f"binrepair|{AMD5[:12]}|L{FIRST_TRAIN}"
assert (DELTA_DIR / "DELTA_PACK.COMPLETE").is_file(), DELTA_DIR


def tier_of(entry, proj):
    return entry["fused13" if proj == "13" else "down"] \
        if isinstance(entry, dict) else entry


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def emit(**row):
    row.setdefault("ts", round(time.time(), 3))
    with JLOG.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[{time.strftime('%H:%M:%S')}] "
          f"{json.dumps(row, sort_keys=True)[:240]}", flush=True)


def status(**fields):
    current = {}
    if STATUS.exists():
        try:
            current = json.loads(STATUS.read_text())
        except Exception:
            pass
    current.update(fields)
    current["updated_ts"] = time.time()
    current["host"] = os.uname().nodename
    atomic_json(STATUS, current)


# ============================================== trainable vq3b dequant op
class Vq3bDeqFn(torch.autograd.Function):
    """w = fp16_STE(cb32)[codes] * exp2(sc-127) -> bf16. Identical math to
    the sealed vq3b reader at init (cb32 = wire fp16 .float(); the fp16
    cast is then an identity). Backward routes grads to the fp32 master
    (STE through fp16 + bf16), exactly the VqaDeqFn pattern with a
    4096-entry codebook."""

    @staticmethod
    def forward(ctx, cb32, codes, sc):
        cbq = cb32.detach().to(torch.float16).float()
        w = cbq[codes.long()]                          # [N, K/4, 4] fp32
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)
        out = (w.reshape(codes.shape[0], -1) * s).to(torch.bfloat16)
        ctx.save_for_backward(codes, sc)
        return out

    @staticmethod
    def backward(ctx, g):
        codes, sc = ctx.saved_tensors
        s = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)
        gf = (g.float() * s).reshape(codes.shape[0], codes.shape[1], 4)
        grad_cb = torch.zeros(4096, 4, device=g.device, dtype=torch.float32)
        grad_cb.index_add_(0, codes.reshape(-1).long(), gf.reshape(-1, 4))
        return grad_cb, None, None


# ============================================== k4096-assignment experts
class K4096Experts(nn.Module):
    """Drop-in for lp4_train.TrainableExperts, dequanting the IQ3_BIN
    target assignment: base pack + delta pack + vq3b planes. Pilot layers
    route vq3b rows through the STE op (trainable cb13/cb2); everything
    else is frozen dense dequant into shared [E,*] buffers."""

    _GU = None
    _DN = None

    def __init__(self, L, pilot):
        super().__init__()
        self.L, self.pilot = L, pilot
        self.base = torch.load(f"{T.PACK}/layer_{L:03d}.pt",
                               map_location="cpu", mmap=True,
                               weights_only=False)
        self.delta = torch.load(DELTA_DIR / f"layer_{L:03d}.pt",
                                map_location="cpu", mmap=True,
                                weights_only=False)
        tmap = MAN["assignment"][str(L)]
        bmap = BASE_MAN["assignment"][str(L)]
        self.limit = 10.0
        self.act = F.silu
        self.tiers = {}
        self.rows = {}
        self.groups = {}          # (proj, tier, src) -> (ids, rows)
        needs_vq3b = False
        for proj in ("13", "2"):
            tiers, rows, srcs = [], [], []
            for e in range(E):
                tt = tier_of(tmap[str(e)], proj)
                bt = tier_of(bmap[str(e)], proj)
                use_delta = tt != bt
                if tt == "vq3b":
                    needs_vq3b = True
                    r, src = e, "plane"
                elif use_delta:
                    assert tt in ("vqa", "ternary"), (
                        L, e, proj, tt, "unexpected delta tier")
                    ids = self.delta[f"{tt}_ids{proj}"]
                    hit = (ids == e).nonzero()
                    assert hit.numel() == 1, (L, e, proj, tt)
                    r, src = int(hit[0, 0]), "delta"
                else:
                    r, src = int(self.base[f"row{proj}"][e]), "base"
                tiers.append(tt)
                rows.append(r)
                srcs.append(src)
                self.groups.setdefault((proj, tt, src), []).append((e, r))
            self.tiers[proj] = tiers
            self.rows[proj] = rows
        self.vq3b = None
        if needs_vq3b:
            self.vq3b = torch.load(VQ3B_DIR / f"vq3u_layer_{L:03d}.pt",
                                   map_location="cpu", mmap=True,
                                   weights_only=True)
            assert tuple(self.vq3b["cb13"].shape) == (4096, 4)
        if pilot:
            assert needs_vq3b, f"pilot layer {L} has no vq3b units"
            self.cb13 = nn.Parameter(self.vq3b["cb13"].float().to(DEV))
            self.cb2 = nn.Parameter(self.vq3b["cb2"].float().to(DEV))

    @staticmethod
    def _deq_bufs():
        if K4096Experts._GU is None:
            K4096Experts._GU = torch.empty(
                E, N13, K13, dtype=torch.bfloat16, device=DEV)
            K4096Experts._DN = torch.empty(
                E, N2, K2, dtype=torch.bfloat16, device=DEV)
        return K4096Experts._GU, K4096Experts._DN

    def _deq_group(self, proj, tier, src, ers, dst, N, K):
        """Dequant one (tier, source) group of experts into dst[E,N,K]."""
        base, delta = self.base, self.delta
        for c0 in range(0, len(ers), DEQ_CHUNK):
            chunk = ers[c0:c0 + DEQ_CHUNK]
            ids = torch.tensor([e for e, _ in chunk], dtype=torch.long)
            rws = torch.tensor([r for _, r in chunk], dtype=torch.long)
            if tier == "w3":
                wq = pk.deq_w3_batched(
                    base[f"w3_pl{proj}"][rws].to(DEV),
                    base[f"w3_sc{proj}"][rws].to(DEV),
                    base["w3_lut"], N, K)
            elif tier == "vqa":
                s = delta if src == "delta" else base
                wq = pk.deq_vqa_batched(
                    s[f"vqa_codes{proj}"][rws].to(DEV),
                    s[f"vqa_sc{proj}"][rws].to(DEV), base[f"cb{proj}"])
            elif tier == "ternary":
                if src == "delta":
                    wq = pk.deq_tern_batched(
                        delta[f"ternary_codes{proj}"][rws].to(DEV),
                        delta[f"ternary_sc{proj}"][rws].to(DEV),
                        delta[f"ternary_lut{proj}"])
                else:
                    wq = pk.deq_tern_batched(
                        base[f"tern_codes{proj}"][rws].to(DEV),
                        base[f"tern_sc{proj}"][rws].to(DEV),
                        base[f"tern_lut{proj}"])
            elif tier == "fp4":
                wq = v3.deq_fp4_block32(
                    base[f"fp4_wb{proj}"][rws].to(DEV),
                    base[f"fp4_sb{proj}"][rws].to(DEV), "e2m1")
            elif tier == "vq3b":
                cb = self.vq3b[f"cb{proj}"].to(DEV).float()
                codes = self.vq3b[f"codes{proj}"][rws].to(DEV).long()
                sc = self.vq3b[f"sc{proj}"][rws].to(DEV)
                s = torch.exp2(sc.float() - 127.0).repeat_interleave(
                    32, dim=-1)
                wq = (cb[codes].reshape(codes.shape[0], N, -1)
                      * s).to(torch.bfloat16)
            else:
                raise KeyError(tier)
            dst[ids.to(DEV)] = wq.to(torch.bfloat16)
            del wq

    def _dense(self, skip_vq3b):
        gu, dn = self._deq_bufs()
        with torch.no_grad():
            for proj, dst, N, K in (("13", gu, N13, K13),
                                    ("2", dn, N2, K2)):
                for (p, tier, src), ers in self.groups.items():
                    if p != proj:
                        continue
                    if skip_vq3b and tier == "vq3b":
                        continue
                    self._deq_group(proj, tier, src, ers, dst, N, K)
        return gu, dn

    def _w_pilot(self, e, which):
        if self.tiers[which][e] != "vq3b":
            return None
        cb = self.cb13 if which == "13" else self.cb2
        return Vq3bDeqFn.apply(
            cb, self.vq3b[f"codes{which}"][e].to(DEV),
            self.vq3b[f"sc{which}"][e].to(DEV))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=E).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        gu, dn = self._dense(skip_vq3b=self.pilot)
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
        return final

    def _evict_planes(self):
        # ALWAYS evict (multi-layer suffix working set ~115G mapped pages
        # would starve NVRM cudaMalloc on GB10 unified mem; FASTCFG multi
        # arms ran with eviction on at 95-102G peak).
        for d in (self.base, self.delta, self.vq3b):
            if d is None:
                continue
            for v in d.values():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu" \
                        and v.untyped_storage().nbytes() > (16 << 20):
                    T.evict_tensor(v)


# ============================================== identity hashes
def codes_hash():
    out = {}
    for L in TRAINABLE:
        h = hashlib.md5()
        d = torch.load(VQ3B_DIR / f"vq3u_layer_{L:03d}.pt",
                       map_location="cpu", mmap=True, weights_only=True)
        for k in ("codes13", "sc13", "codes2", "sc2", "cb13", "cb2"):
            h.update(d[k].numpy().tobytes())
        out[str(L)] = h.hexdigest()
        del d
    return out


# ============================================== activation cache
class ActCache:
    def __init__(self, student):
        self.student = student
        self.mem = {}
        ACTCACHE_DIR.mkdir(parents=True, exist_ok=True)

    def path(self, k):
        return ACTCACHE_DIR / f"win{k}.pt"

    def _load(self, k):
        payload = torch.load(self.path(k), map_location=DEV,
                             weights_only=False)
        expected = {"win": k, "upto": FIRST_TRAIN, "cache_id": CACHE_ID}
        bad = {key: (payload.get(key), val) for key, val in expected.items()
               if payload.get(key) != val}
        if bad:
            raise RuntimeError(f"actcache identity mismatch {k}: {bad}")
        self.mem[k] = payload["h"]
        emit(event="actcache_load", win=k)

    def build_many(self, corpus, wins):
        missing = []
        for k in wins:
            if k in self.mem:
                continue
            if self.path(k).exists():
                self._load(k)
            else:
                missing.append(k)
        if not missing:
            return
        ids = torch.stack([T.window_ids(corpus, k)[0]
                           for k in missing]).to(DEV)
        t0 = time.time()
        m, config = self.student.model, self.student.config
        from transformers.masking_utils import (
            create_sliding_window_causal_mask)
        from transformers.cache_utils import DynamicCache
        pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
        embeds = m.model.embed_tokens(ids)
        pe = {"main": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                         layer_type="main"),
              "compress": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                             layer_type="compress")}
        mask = create_sliding_window_causal_mask(
            config=config, inputs_embeds=embeds, attention_mask=None,
            past_key_values=DynamicCache(config=config), position_ids=pos)
        hidden = embeds.unsqueeze(2).expand(
            -1, -1, config.hc_mult, -1).contiguous()
        del embeds
        with torch.no_grad():
            for Li in range(FIRST_TRAIN):
                hidden = m.model.layers[Li](
                    hidden, position_embeddings=pe, position_ids=pos,
                    attention_mask=mask, input_ids=ids,
                    past_key_values=DynamicCache(config=config))
                if Li % 8 == 7:
                    torch.cuda.empty_cache()
        hidden = hidden.detach()
        secs = round(time.time() - t0, 1)
        for i, k in enumerate(missing):
            h = hidden[i:i + 1]
            p = self.path(k)
            tmp = Path(str(p) + ".tmp")
            torch.save({"h": h.cpu(), "win": k, "upto": FIRST_TRAIN,
                        "cache_id": CACHE_ID, "manifest_md5": AMD5,
                        "ts": time.time()}, tmp)
            os.replace(tmp, p)
            self.mem[k] = h
        emit(event="actcache_build_batch", wins=missing, secs=secs,
             secs_per_window=round(secs / len(missing), 1))
        torch.cuda.empty_cache()

    def get(self, corpus, k):
        if k not in self.mem:
            self.build_many(corpus, [k])
        return self.mem[k]


# ============================================== forward + loss
def fast_forward(student, hidden, ids, requires_grad):
    m, config = student.model, student.config
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.cache_utils import DynamicCache
    pos = torch.arange(ids.shape[1], device=DEV).unsqueeze(0)
    embeds = m.model.embed_tokens(ids[:1])
    pe = {"main": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                     layer_type="main"),
          "compress": m.model.rotary_emb(embeds[:1], position_ids=pos,
                                         layer_type="compress")}
    mask = create_sliding_window_causal_mask(
        config=config, inputs_embeds=m.model.embed_tokens(ids),
        attention_mask=None,
        past_key_values=DynamicCache(config=config), position_ids=pos)
    del embeds

    def run_layer(Li, h):
        return m.model.layers[Li](
            h, position_embeddings=pe, position_ids=pos,
            attention_mask=mask, input_ids=ids,
            past_key_values=DynamicCache(config=config))

    for Li in range(FIRST_TRAIN, config.num_hidden_layers):
        if requires_grad:
            hidden = checkpoint(run_layer, Li, hidden, use_reentrant=False)
        else:
            with torch.no_grad():
                hidden = run_layer(Li, hidden)
    return m.model.norm(m.model.hc_head(hidden))


def loss_window(student, h_j, T_j, k):
    """Support-renormalized KL (scorer-exact; full-vocab LSE cancels)."""
    idx, lp_n, p_n = T.teacher_rows(k)
    logits = student.model.lm_head(h_j[:T_j].to(torch.bfloat16))
    q = logits.gather(1, idx[:T_j]).float()
    lq_n = q - q.logsumexp(-1, keepdim=True)
    out = (p_n[:T_j] * (lp_n[:T_j] - lq_n)).sum(-1).mean()
    del idx, lp_n, p_n, logits
    return out


def batch_loss(student, corpus, acache, wins, requires_grad):
    rls = [T.window_ids(corpus, k)[1] for k in wins]
    ids = torch.stack([T.window_ids(corpus, k)[0] for k in wins]).to(DEV)
    hidden = torch.cat([acache.get(corpus, k) for k in wins], 0)
    h = fast_forward(student, hidden, ids, requires_grad)
    loss = 0.0
    for j, k in enumerate(wins):
        loss = loss + loss_window(student, h[j], rls[j], k)
    return loss / len(wins)


def kld_window(student, corpus, acache, k):
    with torch.no_grad():
        return float(batch_loss(student, corpus, acache, [k], False))


# ============================================== gradcheck
def gradcheck(student):
    te = student.experts[TRAINABLE[0]]
    e = next(e for e in range(E) if te.tiers["13"][e] == "vq3b")
    codes = te.vq3b["codes13"][e].to(DEV)
    sc = te.vq3b["sc13"][e].to(DEV)
    cb = nn.Parameter(te.vq3b["cb13"].float().to(DEV))
    torch.manual_seed(0)
    G = torch.randn(N13, K13, device=DEV)
    s_col = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=-1)

    def ref_loss(cb_val):
        cbq = cb_val.detach().to(torch.float16).float()
        w = cbq[codes.long()].reshape(codes.shape[0], -1) * s_col
        return (w * G).sum().item()

    loss = (Vq3bDeqFn.apply(cb, codes, sc).float() * G).sum()
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
    emit(event="gradcheck", layer=TRAINABLE[0], expert=e,
         autograd=ag, fd=fd, rel=round(rel, 6))
    assert rel < 0.02, f"gradcheck FAIL rel={rel}"


# ============================================== checkpointing
def state_named(student):
    return {f"L{L}": {"cb13": student.experts[L].cb13.detach().cpu(),
                      "cb2": student.experts[L].cb2.detach().cpu()}
            for L in TRAINABLE}


def save_ckpt(path, student, opt, next_step, base, chash, best_mean):
    payload = {"format": "binrepair-v1", "manifest_md5": AMD5,
               "codes_hash": chash, "trainable": TRAINABLE, "lr": LR,
               "steps_target": STEPS, "train_wins": TRAIN_WINS,
               "probe_wins": PROBE_WINS, "cache_id": CACHE_ID,
               "next_step": next_step, "baseline": base,
               "best_probe_mean": best_mean,
               "state": state_named(student),
               "optimizer": opt.state_dict(),
               "saved_ts": time.time(), "host": os.uname().nodename}
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_resume(student, opt, chash):
    if not LATEST.exists():
        return 0, None, None
    ck = torch.load(LATEST, map_location="cpu", weights_only=False)
    expected = {"format": "binrepair-v1", "manifest_md5": AMD5,
                "codes_hash": chash, "trainable": TRAINABLE, "lr": LR,
                "train_wins": TRAIN_WINS, "probe_wins": PROBE_WINS,
                "cache_id": CACHE_ID}
    bad = {k: (ck.get(k), v) for k, v in expected.items()
           if ck.get(k) != v}
    if bad:
        raise RuntimeError(f"resume identity mismatch: {list(bad)}")
    for L in TRAINABLE:
        te = student.experts[L]
        te.cb13.data.copy_(ck["state"][f"L{L}"]["cb13"].to(DEV))
        te.cb2.data.copy_(ck["state"][f"L{L}"]["cb2"].to(DEV))
    opt.load_state_dict(ck["optimizer"])
    emit(event="resumed", next_step=ck["next_step"])
    return ck["next_step"], ck.get("baseline"), ck.get("best_probe_mean")


# ============================================== main
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    status(state="starting", tag=TAG, manifest_md5=AMD5,
           trainable=TRAINABLE, lr=LR, steps=STEPS, batch=BATCH,
           train_wins=TRAIN_WINS, probe_wins=PROBE_WINS,
           cache_id=CACHE_ID)
    emit(event="start", tag=TAG, manifest_md5=AMD5, trainable=TRAINABLE,
         lr=LR, steps=STEPS, batch=BATCH, train_wins=TRAIN_WINS,
         probe_wins=PROBE_WINS, cache_only=CACHE_ONLY)

    chash = codes_hash()
    emit(event="codes_hash", **{f"L{k}": v for k, v in chash.items()})

    T.TrainableExperts = K4096Experts
    T.PILOT = tuple(TRAINABLE)
    student = T.Student()
    n_par = sum(p.numel() for L in TRAINABLE
                for p in (student.experts[L].cb13, student.experts[L].cb2))
    emit(event="assembled", n_trainable_params=n_par,
         pilot_layers=TRAINABLE)
    corpus = T.load_corpus()

    if GRADCHECK and not CACHE_ONLY:
        gradcheck(student)

    acache = ActCache(student)
    all_wins = TRAIN_WINS + [w for w in PROBE_WINS if w not in TRAIN_WINS]

    if CACHE_ONLY:
        for c0 in range(0, len(all_wins), CACHE_BATCH):
            wins = all_wins[c0:c0 + CACHE_BATCH]
            acache.build_many(corpus, wins)
            for w in wins:
                acache.mem.pop(w, None)
            torch.cuda.empty_cache()
            status(state="cache_building", cache_done=c0 + len(wins),
                   cache_total=len(all_wins))
        status(state="cache_completed")
        emit(event="cache_completed", count=len(all_wins))
        return

    params = [p for L in TRAINABLE
              for p in (student.experts[L].cb13, student.experts[L].cb2)]
    opt = torch.optim.Adam(params, lr=LR)
    start_step, base, best_mean = try_resume(student, opt, chash)

    ref = {}
    if REF_KLD_PATH and os.path.exists(os.path.expanduser(REF_KLD_PATH)):
        ref = {int(k): float(v) for k, v in json.loads(
            Path(os.path.expanduser(REF_KLD_PATH)).read_text()).items()}

    if base is None:
        base = {}
        for w in all_wins:
            t0 = time.time()
            val = kld_window(student, corpus, acache, w)
            assert val == val and val < 5.0, f"non-physical baseline {w}: {val}"
            base[w] = val
            r = ref.get(w)
            emit(event="baseline", win=w, kld=val, ledger_ref=r,
                 ledger_delta=None if r is None else round(val - r, 6),
                 secs=round(time.time() - t0, 1))
        save_ckpt(LATEST, student, opt, 0, base, chash, None)
    base = {int(k): float(v) for k, v in base.items()}

    def probe(step):
        vals = {}
        for w in PROBE_WINS:
            t0 = time.time()
            v = kld_window(student, corpus, acache, w)
            vals[w] = v
            emit(event="probe", step=step, win=w, kld=v, baseline=base.get(w),
                 delta_pct=None if w not in base else
                 round((base[w] - v) / base[w] * 100.0, 4),
                 secs=round(time.time() - t0, 1))
            torch.cuda.empty_cache()
        mean = sum(vals.values()) / len(vals)
        bmean = sum(base[w] for w in PROBE_WINS) / len(PROBE_WINS)
        emit(event="probe_mean", step=step, mean=round(mean, 6),
             baseline_mean=round(bmean, 6),
             delta_pct=round((bmean - mean) / bmean * 100.0, 4))
        status(state="running", last_probe_step=step,
               last_probe_mean=mean, baseline_probe_mean=bmean)
        return mean

    if start_step == 0:
        best_mean = probe(0)
        save_ckpt(BEST, student, opt, 0, base, chash, best_mean)
    if best_mean is None:
        best_mean = sum(base[w] for w in PROBE_WINS) / len(PROBE_WINS)

    t_run = time.time()
    n_groups = max(1, (len(TRAIN_WINS) + BATCH - 1) // BATCH)
    stall = 0
    for step in range(start_step, STEPS):
        if (time.time() - t_run) / 3600 > MAX_HOURS:
            emit(event="wall_guard", step=step)
            break
        gi = step % n_groups
        wins = TRAIN_WINS[gi * BATCH:(gi + 1) * BATCH]
        t0 = time.time()
        loss = batch_loss(student, corpus, acache, wins, True)
        kld_pre = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(sum((p.grad.norm() ** 2 for p in params
                        if p.grad is not None)) ** 0.5)
        opt.step()
        opt.zero_grad(set_to_none=True)
        del loss
        torch.cuda.empty_cache()
        next_step = step + 1
        milestone = next_step % PROBE_EVERY == 0 or next_step == STEPS
        save_ckpt(LATEST, student, opt, next_step, base, chash, best_mean)
        emit(event="step", step=next_step, train_wins=wins,
             kld_pre_update=kld_pre, grad_norm=gn,
             secs=round(time.time() - t0, 1),
             mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        status(state="running", next_step=next_step,
               last_kld_pre_update=kld_pre, last_grad_norm=gn)
        if milestone:
            mean = probe(next_step)
            if mean < best_mean - 1e-6:
                best_mean = mean
                stall = 0
                save_ckpt(BEST, student, opt, next_step, base, chash,
                          best_mean)
                emit(event="best", step=next_step, mean=round(mean, 6))
            else:
                stall += 1
                emit(event="stall", step=next_step, count=stall)
                if stall >= EARLY_STOP:
                    emit(event="early_stop", step=next_step)
                    break

    bmean = sum(base[w] for w in PROBE_WINS) / len(PROBE_WINS)
    result = {"state": "completed", "tag": TAG, "manifest_md5": AMD5,
              "codes_hash": chash, "trainable": TRAINABLE, "lr": LR,
              "baseline_probe_mean": bmean, "best_probe_mean": best_mean,
              "best_delta_pct": (bmean - best_mean) / bmean * 100.0,
              "best_checkpoint": str(BEST), "latest": str(LATEST),
              "host": os.uname().nodename, "ts": time.time()}
    atomic_json(FINAL, result)
    emit(event="completed", **{k: v for k, v in result.items()
                               if k not in ("codes_hash",)})
    status(**result)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        try:
            status(state="failed", error=f"{type(exc).__name__}: {exc}")
            emit(event="failed", error=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
