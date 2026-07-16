#!/usr/bin/env python3
"""KLD-HARNESS (t_41d518a2): the scorer. FP8-reference top-8192 rows +
candidate rows -> KL / JS / top-1 agreement at the madeby561 measurement shape
(prefill, ~1.05M positions, top-8192 support) -> KLD_LEDGER.jsonl row +
their-format markdown table.

CONVENTION (pre-registered; madeby561's exact code is unpublished):
  For each position, S = REFERENCE's top-8192 token support.
  Primary ("renorm"): p = ref probs on S renormalized; q = cand probs on S
    renormalized (cand logprobs are log_softmax over FULL vocab first, then
    gathered on S).  KL = sum_S p*(log p - log q).
  Secondary ("abs", diagnostics): same gather, NO renormalization (raw
    full-softmax masses on S).  Both are emitted so the calibration gate
    (rev3 == 0.177 +/- 0.01) can discriminate the convention; support-mass
    diagnostics quantify the difference (~1e-4 at mass .9999).
  JS = 0.5*KL(p||m) + 0.5*KL(q||m), m = (p+q)/2, computed on renormalized S.
  top1_agree = mean over positions of [cand full-vocab argmax == ref argmax].

INPUTS
  ref dir:  t8192_win<k>.pt  {"idx": int32 [T,8192] (sorted desc by ref lp),
                              "logprob": fp16 [T,8192] full-softmax lp}
  cand dir: q8192_win<k>.pt  {"q_lp_at_ref": fp16 [T,8192] full-softmax lp
                              gathered at ref idx, "q_argmax": int32 [T]}
            OR logits_win<k>.pt fp16 [T,V] full logits (auto-detected).

USAGE
  kld_score.py <ref_dir> <cand_dir> <tag> [--ledger PATH] [--support N]
               [--corpus-meta PATH] [--notes STR] [--pos-cutoff P]
Appends one row to the ledger (default: <cand_dir>/../KLD_LEDGER.jsonl) and
prints the summary. Per-window rows -> <cand_dir>/KLD_WINDOWS.jsonl.

CALIBRATED CONVENTION (v2, sealed 2026-07-07 run 319): --pos-cutoff 1024.
Scoring only the first 1024 positions of each banked 2048-token window is,
by the causal-prefill identity, EXACTLY scoring 1024-token windows at the
same start offsets. This is the convention that reproduces madeby561's
published rev3 anchor (measured 0.17748 vs 0.177 +/- 0.01 -> PASS; the
full-2048 read gives 0.16700 = FAIL, and support size 512..8192 / renorm
vs abs / direction / log-base all fail to move it into band — see
logs/rescore_sweep.log). Citable rows MUST use --pos-cutoff 1024.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("tag")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--support", type=int, default=8192)
    ap.add_argument("--corpus-meta", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--pos-cutoff", type=int, default=0,
                    help="score only the first P positions of each window "
                         "(= P-token windows by causal-prefill identity). "
                         "0 = all. Calibrated convention: 1024.")
    a = ap.parse_args()

    ledger = a.ledger or os.path.join(
        os.path.dirname(a.cand_dir.rstrip("/")), "KLD_LEDGER.jsonl")
    wjs = os.path.join(a.cand_dir, "KLD_WINDOWS.jsonl")

    wins = sorted(int(f[len("t8192_win"):-3]) for f in os.listdir(a.ref_dir)
                  if f.startswith("t8192_win") and f.endswith(".pt"))
    assert wins, f"no t8192_win*.pt in {a.ref_dir}"

    S = a.support
    tot = dict(kl=0.0, kl_abs=0.0, js=0.0, pos=0, t1=0.0, t1s=0.0,
               mass_p=0.0, mass_q=0.0)
    n_done = 0
    for w in wins:
        ref_p = f"{a.ref_dir}/t8192_win{w}.pt"
        cand_q8 = f"{a.cand_dir}/q8192_win{w}.pt"
        cand_full = f"{a.cand_dir}/logits_win{w}.pt"
        ref = torch.load(ref_p, map_location=DEV)
        idx = ref["idx"].long()[:, :S]
        ref_lp = ref["logprob"].float()[:, :S]
        if os.path.exists(cand_q8):
            c = torch.load(cand_q8, map_location=DEV)
            q_lp = c["q_lp_at_ref"].float()[:, :S]
            q_am = c["q_argmax"].long()
        elif os.path.exists(cand_full):
            ql = torch.load(cand_full, map_location=DEV).float()
            q_lp_full = torch.log_softmax(ql, dim=-1)
            T = min(idx.shape[0], q_lp_full.shape[0])
            idx, ref_lp = idx[:T], ref_lp[:T]
            q_lp = q_lp_full.gather(1, idx)
            q_am = q_lp_full.argmax(-1)
            del ql, q_lp_full
        else:
            continue
        T = min(idx.shape[0], q_lp.shape[0])
        if a.pos_cutoff > 0:
            T = min(T, a.pos_cutoff)
        idx, ref_lp, q_lp, q_am = idx[:T], ref_lp[:T], q_lp[:T], q_am[:T]

        p = ref_lp.exp()
        q = q_lp.exp()
        mass_p = p.sum(-1)
        mass_q = q.sum(-1)
        lp_n = ref_lp - ref_lp.logsumexp(-1, keepdim=True)
        lq_n = q_lp - q_lp.logsumexp(-1, keepdim=True)
        p_n = lp_n.exp()
        q_n = lq_n.exp()
        kl = (p_n * (lp_n - lq_n)).sum(-1)                    # renorm (primary)
        kl_abs = (p * (ref_lp - q_lp)).sum(-1)                # no renorm
        m = 0.5 * (p_n + q_n)
        lm = m.clamp_min(1e-12).log()
        js = 0.5 * (p_n * (lp_n - lm)).sum(-1) \
            + 0.5 * (q_n * (lq_n - lm)).sum(-1)
        t1 = (q_am == idx[:, 0]).float()
        t1s = (idx[:, :64] == q_am[:, None]).any(-1).float()

        row = dict(win=w, tag=a.tag,
                   kl_mean=round(kl.mean().item(), 6),
                   kl_p50=round(kl.median().item(), 6),
                   kl_p95=round(kl.quantile(0.95).item(), 6),
                   kl_abs_mean=round(kl_abs.mean().item(), 6),
                   js_mean=round(js.mean().item(), 6),
                   top1_agree=round(t1.mean().item(), 6),
                   top1_in_top64=round(t1s.mean().item(), 6),
                   support_mass_p=round(mass_p.mean().item(), 6),
                   support_mass_q=round(mass_q.mean().item(), 6),
                   n_pos=int(T))
        jrow(wjs, **row)
        tot["kl"] += kl.sum().item()
        tot["kl_abs"] += kl_abs.sum().item()
        tot["js"] += js.sum().item()
        tot["t1"] += t1.sum().item()
        tot["t1s"] += t1s.sum().item()
        tot["mass_p"] += mass_p.sum().item()
        tot["mass_q"] += mass_q.sum().item()
        tot["pos"] += T
        n_done += 1
        if n_done % 64 == 0:
            print(f"...{n_done}/{len(wins)} windows "
                  f"(running KL {tot['kl'] / tot['pos']:.4f})", flush=True)
        del ref, idx, ref_lp, q_lp, q_am, p, q, p_n, q_n, lp_n, lq_n, m, lm
        if DEV == "cuda":
            torch.cuda.empty_cache()

    assert tot["pos"] > 0, "no candidate rows found"
    corpus_md5 = None
    if a.corpus_meta and os.path.exists(a.corpus_meta):
        corpus_md5 = json.load(open(a.corpus_meta)).get("md5")
    summ = dict(variant=a.tag,
                kl_vs_fp8=round(tot["kl"] / tot["pos"], 6),
                kl_vs_fp8_abs=round(tot["kl_abs"] / tot["pos"], 6),
                js=round(tot["js"] / tot["pos"], 6),
                top1_agree=round(tot["t1"] / tot["pos"], 6),
                top1_in_top64=round(tot["t1s"] / tot["pos"], 6),
                n_positions=tot["pos"], n_windows=n_done,
                support_size=S,
                support_mass_ref=round(tot["mass_p"] / tot["pos"], 6),
                support_mass_cand=round(tot["mass_q"] / tot["pos"], 6),
                corpus_md5=corpus_md5,
                pos_cutoff=a.pos_cutoff,
                convention_notes=(
                    "prefill KL(fp8_ref||cand) on ref top-%d support, both "
                    "renormalized on S; cand lp = full-vocab log_softmax then "
                    "gather. kl_vs_fp8_abs = same without renorm. JS on "
                    "renormalized S. top1 = full-vocab argmax match. "
                    "pos_cutoff=%d (0=all; calibrated convention v2 = 1024 "
                    "-> 1024-token windows via causal-prefill identity). %s"
                    % (S, a.pos_cutoff, a.notes)).strip())
    jrow(ledger, **summ)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
