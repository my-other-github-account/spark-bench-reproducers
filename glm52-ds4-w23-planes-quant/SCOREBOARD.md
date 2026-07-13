# DS4F-KLD-MATRIX — the 6-row DS4-Flash quant-tricks table (t_b04fc3fe)

Banana Bae (Jul10, verbatim): "We should end up with DSV4F KLD numbers for Q2 /
Q3 / NVFP4 / Calibrated Q2 / Calibrated Q3 / Dynamic experts targeting a
bpw that fits with KV on 1xSpark."

Workspace: <orchestrator>/clawd/ds4-flash-kldmatrix/
Testbed serve: spark-4 <internal-host> (DS4-Flash 159B, ONE Spark,
G2-sealed: W2 sign-sym RTN planes 73G, kv fp8, MTP k=2, enforce-eager,
vllm-0.24.0-2ef3137a). Serve exclusivity: bench queries + CPU-side scoring
only; no second GPU tenant.

## The measurement stack (binding convention, all rows)

CORPUS  out/windows_ds4_eval.json  md5 1701920b4ba96dea0b18fe9df0151876
  512 windows, GOLD-CALIB EVAL lineage (t_279db38b, md5 0a6af811) decoded
  with the GLM tokenizer (752f6cd2) and re-encoded with the DS4-Flash
  tokenizer (3f75dbea, byte-identical to the live serve /tokenize —
  verified). real_len mean 2009.6 (min 1237 / max 2048; 318 truncated at
  2048, 194 shorter). Scored positions: first min(1024, real_len-1) per
  window = 524,288 positions — the exact v2 --pos-cutoff 1024 shape of the
  GLM 0.1775 rail. Class mix preserved (agentic 154 / reasoning 76 / code
  76 / prose 78 / multilingual 76 / chat 52).
  CAVEAT (uniform across rows, so comparisons stand): conversational
  windows carry GLM chat-template surface forms as literal text. A
  re-render through DS4's encoding_dsv4 is an optional follow-on; every
  row in this table uses THIS corpus so the deltas are apples-to-apples.

PRIMARY (per rung)  offline teacher-forced KLD vs the BF16/source teacher
  — same convention as the GLM rail (ref top-8192 support, renormalized,
  KL(ref||cand), --pos-cutoff 1024). REQUIRES the DS4 teacher rail
  (t8192 rows) — see "Teacher rail" below. NOT YET BUILT.

SERVE CROSS-CHECK (per rung, cheap, available TODAY)
  - R*_NLL_ROW.json: teacher-forced serve NLL (natural log, per-token mean
    over the 524,288 scored positions) via /v1/completions token-id
    prompts + prompt_logprobs=1, c=1 sequential. Banks per-position
    [actual_tok, logprob, rank, top1_tok, top1_logprob] in
    out/r*_positions/ (gzip) -> enables serve-vs-serve flip/top1-agreement
    joins between rungs WITHOUT the teacher, and NLL deltas per class.
  - R*_MMLU500_ROW.json: MMLU-500 0-shot choice loglik, the fleet
    DOWNSTREAM_LEDGER protocol verbatim (question_set_sha256 24d60b46...,
    same 500 questions as every GLM row; hendrycks tar md5 20bb2076
    verified; DS4-tokenized, ' A'..' D' -> single tokens [334,406,345,420]).

## The table

| row | variant                       | KLD vs teacher | serve NLL (524,288 pos) | MMLU-500 | status |
|-----|-------------------------------|----------------|-------------------------|----------|--------|
| R1  | Q2 (W2 sign-sym RTN planes)   | **0.390165** (js 0.0690, top1 0.809) | **1.5045** (ppl 4.50, SEM 0.0247) | **0.802** serve / **0.810** offline (405/500, gold lp -0.573, margin 2.52; x-check +0.8pt = no pipeline floor) | serve rows DONE; KLD row sealed (t_394f19e7, offline W2 NLL 1.4923 x-checks serve 1.5045 within 0.8%); offline MMLU t_e3f38867 |
| R2  | Q3 (W3 RTN planes)            | **0.373666** (js 0.0632, top1 0.815) | serve **1.499075** (ppl 4.48, SEM 0.0249) / offline 1.4877 | **0.792** serve (396/500, gold lp -0.582, margin 2.97) / **0.788** offline (394/500) — does NOT beat Q2 (Q3−Q2 = −2.2pt, CI [−5.3,+0.9], n.s.) | KLD row sealed (t_beb28ef4, s8 offline rail, shipped moe_w3_planes bytes). SERVE ROWS DONE (t_db7f8abc/t_f6892953, s7 W3 serve, BATTERY_R2_DONE 2026-07-12 00:30Z): serve-vs-offline NLL x-check 0.77% + MMLU +0.4pt = R1-class agreement. **DESIGN AUDIT VERDICT (t_eee6b0cc): row is a VALID measurement of the SHIPPED bytes, but the shipped W3 LUT is now PROVEN a bad 8-point quantizer** — see "W3 design audit" section. Corrected-design row = R2v2 below. Old finding "W3 RTN buys only 4.2% KL" applies to the -6..6 ladder specifically, NOT to 3-bit experts as an axis |
| R2v2 | Q3 (W3v2: DP-optimal LUT + refit scales) | **0.087660** (js 0.0175, top1 0.914) | offline **1.26672** | **0.842** (421/500, gold lp -0.432, margin 3.68 — M_Q3v2, −0.2pt vs anchor 0.844) | **SEALED** (t_eee6b0cc): ledger line R2_ds4flash_w3_planes_v2 on s8 + s2 mirror (md5 a52750db identical). **THE W3 row — inside the near-lossless band** (community Q3 bar 0.081 KLD/−0.6pt). W3-done-right beats W2 by 4.45x KLD; RTN-on-right-grid beats GPTQ-on-wrong-grid 1.8x → level placement ≫ calibration at 3 bits. moe_w3_planes_v2 = dp_asym8_fit LUT [-6.379,-3.4723,-1.8718,-0.8547,+0.137,+1.4651,+3.4796,+6.3792] + per-block SSE-refit UE8M0 scales, same 3.25 bpw wire. R5v2 (GPTQ onto this grid) in flight t_26055bf3 |
| R3  | NVFP4 / 4-bit anchor          | —              | —         | **0.844** offline (422/500, gold lp -0.451, margin 3.92) = M-ref, the source-teacher forward (DS4 has no public bf16; source ckpt is 4-bit-native mxfp4-flavor routed → ref IS the NVFP4-class anchor) | NOTE: base ckpt is 222G — NOT servable on 1 Spark; anchor row must come from the offline teacher-rail forward (teacher vs itself = the reference row) or a true modelopt-NVFP4 quant that fits. MMLU anchor sealed t_e3f38867 |
| R4  | Calibrated Q2 (GPTQ->W2 grid) | **0.311544** (js 0.0568, top1 0.832) | —         | **0.810** offline (405/500, gold lp -0.561, margin 2.92) — Q4−Q2 = **0.0pt** CI [−2.7,+2.7] p=1.0; paired gold-lp delta also null (+0.012 nats, t=0.7) | KLD row SEALED (t_fa509f27, s8 offline rail, calibrated GPTQ planes, offline NLL 1.4318, gate PASS). **Calibration PAYS: −20.2% KL vs R1 at identical 2.25 bpw**; also beats R2 W3-RTN (0.374) by −16.6% at ⅔ the bits. Full coverage 43 layers × 256 experts; calib = GOLD-CALIB CALIB split (windows_ds4_calib.json md5 d09b0069, disjoint from eval); solver ds4_gptq.py md5 8c7a7897, per-proj val-gated ship, scale bytes shipped-verbatim. MMLU (t_e5898cb2): −20% KL does NOT show up at task level — ref still +3.4pt over Q4 (p=0.016); honest read = null at n=500 AND at the continuous readout |
| R5  | Calibrated Q3 (GPTQ->W3 grid) | **0.159694** (js 0.0301, top1 0.880) | —         | **0.832** offline (416/500, gold lp -0.462, margin 3.58) — Q5−Q3 = **+4.4pt** CI [+1.5,+7.2] p=0.0038 SIGNIFICANT; ref−Q5 = +1.2pt p=0.36 n.s. → **task-level parity with the source-teacher anchor** (paired gold-lp delta +0.010 nats, t=0.5) | KLD row SEALED (t_fa509f27, s8 offline rail, calibrated GPTQ planes, offline NLL 1.30497, gate PASS). **Calibration pays HUGE on W3: −57.3% KL vs R2 at identical 3.25 bpw**; −48.7% vs R4 → with calibration the extra W3 bit DOES pay (revises the R2 RTN finding: the RTN family, not bit-count, was masking the headroom). Recovers 69% of the RTN→teacher NLL gap. Same coverage/provenance as R4. MMLU (t_e5898cb2): calibration EARNS the W3 rung at task level too — R5 is the first rung indistinguishable from ref on MMLU-500 |
| R6  | Dynamic experts @ 1-Spark budget | —           | —         | —        | CARDED t_29c4872c (atlaskernel5, parents=[t_26055bf3 R5v2]): damage map + knapsack allocator + per-expert manifest loader (tiers {W2, W3v2, native-FP4 passthrough}); spec per t_b04fc3fe comments 1318/1363 — re-run knapsack with W3 tier at 0.0877-class damage |

## UD-IQ comparison ladder — llama.cpp/Unsloth-UD quants (t_91e811e8, Banana Bae Jul12, s8+s3-RPC)

INSTRUMENT (mandatory caveat, all UD rows): llama.cpp native --kl-divergence.
Teacher = UD-Q8_K_XL GGUF (NOT our fp8-source rail); corpus = OUR
windows_ds4_eval.json (md5 1701920b) decoded to TEXT with the DS4 tokenizer
(1,028,912 source tokens, corpus md5 b559b14a) then RE-TOKENIZED by llama.cpp;
scoring = last 1023 positions of each 2048-token chunk over re-chunked
concatenated text (502 chunks), vs ours = first 1024 of each aligned window.
Same family, different instrument — do NOT subtract UD KLD from R-row KLD;
compare LADDER SHAPES and MMLU (task column is protocol-identical: same 500
qids sha 24d60b46, choice tokens ' A'..' D' -> [334,406,345,420], 0-shot,
llama.cpp --multiple-choice with MC_SEQUENTIAL patch for the DSV4 kv-cache).
BPW sealed from GGUF tensor tables via HTTP range reads (ggml block sizes
verified vs ggml.h); UD quants are LAYER-MIX (attn/dense kept higher-bit,
routed experts at tier bit) so "1-bit" IQ1_S is really 2.32/2.18 bpw.

| variant | instrument | KLD mean | KLD p95 | bpw(model) | bpw(experts) | MMLU-500 | status |
|---|---|---|---|---|---|---|---|
| UD-Q8_K_XL (teacher) | llama.cpp | 0 (self) | — | 4.554 | 4.250 | **0.840** (420/500 ±1.64) | SEALED; PPL 3.2793±0.011; instrument x-check: our M-ref anchor 0.844 → llama.cpp MC harness agrees within 0.4pt |
| UD-IQ1_S | llama.cpp | **0.2852** ±0.0010 | 1.368 | 2.322 | 2.182 | **0.818** (409/500 ±1.73) | SEALED (kld_UD-IQ1_S.log; PPL 3.8547 = 1.180x teacher; top1 83.5%; median KLD 0.042) |
| UD-IQ2_XXS | llama.cpp | | | 2.556 | 2.422 | | MMLU running 20:23Z |
| UD-IQ3_XXS | llama.cpp | | | 2.898 | 2.761 | | queued (download complete) |
| UD-IQ4_XS | llama.cpp | | | 3.880 | 3.757 | | queued (download complete; runs via s3 RPC, >121G) |

Early reads (IQ1 row): (1) UD-IQ1_S at 2.18 expert-bpw lands KLD 0.285 —
BELOW our W2-GPTQ 0.312 at 2.25 expert-bpw on a coarser instrument-adjacent
read, and MMLU 0.818 vs our 0.810: the UD layer-mix + imatrix recipe is
competitive-to-better at the 2-bit tier. NOT the pre-registered ~0.8-1.0
crater — that prediction assumed pure 1-bit; the UD mix spends 2.18 bits.
(2) Teacher-instrument sanity holds (0.840 vs 0.844 anchor). Remaining
comparisons wait on IQ3 (vs our W3v2 0.0877/0.0727 at 3.25) and IQ4.

R1 serve NLL detail (R1_NLL_ROW.json): per-class nll = reasoning 0.959 /
code 1.207 / agentic 1.392 / chat 1.500 / multilingual 1.797 / prose 2.266;
self_top1_rate 0.6736. Serve provenance: planes dir 73G/216 files
(layer_NNN.{meta.json,planes13.npy,planes2.npy,sc13.npy,sc2.npy}),
pid 32909 up 5h19m at measurement. MTP spec-decode does NOT touch these
rows: prompt_logprobs come from the prefill forward; generation discarded.

## Teacher rail (BUILT + MIRRORED — t_394f19e7 sealed 2026-07-10, s8 mirror t_beb28ef4 2026-07-11)

STATUS: DONE. 512/512 windows, sealed t8192 payload format, 48G.
  PRIMARY: spark-2 ~/missions/DS4_TEACHER/t8192_eval/
  MIRROR:  spark-8 ~/missions/DS4_TEACHER/t8192_eval/ (fabric rsync,
           DONE.jsonl payload-md5 spot-checks PASS, sealed-scorer re-gate
           reproduced R1 0.390165 EXACTLY on s8)
  R2 artifacts mirrored BACK s8->s2 (t_beb28ef4 close-out): KLD_LEDGER
  R2 line appended on s2 (R1 line diff-verified identical first),
  q8192_eval_w3 512/512 rows (8.1G, win0/127/317/511 md5 MATCH),
  W3_NLL.json + v3 builder + unpackers + audit + logs_s8_r2/.
Gates (sealed): teacher mean nll1024 = 1.22157 < serve anchor 1.5045;
readback smoke through sealed kld_score.py --pos-cutoff 1024: kl=0.0
top1=1.0 exact. Corpus md5 1701920b + scorer md5 8011368c verified on
both hosts.
Builders: t8192_ds4_build_v2.py (bf16 teacher / w2-snap cand), v3 adds
--mode planes: candidate forward from SHIPPED plane bytes (moe_w2/w3
wire format auto-detected, LUT from meta.json) = the exact interface
calibrated GPTQ planes (R4/R5) drop into. v3 gates on s8: loader
byte-equivalence (planes-mode output md5-identical to w2-snap control
on win0/win317) + plane-provenance audit (W2+W3 on-disk bytes ==
deterministic source requant, 36/36 expert-matrices PASS).

Original feasibility notes (superseded):

Teacher = DS4-Flash source weights (mxfp4-flavor base, bf16-dequant
reference), one teacher-forced forward over windows_ds4_eval.json ->
t8192_win<k>.pt rows ({idx int32 [T,8192] desc, logprob fp16 [T,8192]}),
cached exactly like the GLM rail (SWORK_BF16_TEACHER pattern, reuse
t8192_bf16_build.py skeleton with the DS4 HF graph + Ckpt reader).
HOST: swork (currently BUSY building the GLM t8192 rail, GPU 29G) or s6
(BUSY, GPTQ G4). DO NOT preempt either — the rail runs when a slot frees.
Feasibility check before launch: DS4 ckpt is 222G on s4 (fp8/mxfp4 on
disk); layer-streamed forward on a 121G-GPU-budget Spark is the same
rolling-shard pattern as the GLM rail (703G ckpt streamed fine).
Storage: ~51GB per t8192 rail (GLM precedent) — s4 has only 42G free;
rail should live on the teacher host or orchestrator-host mirror, NOT s4.

Once the teacher rail exists, the R1 KLD row does NOT need the serve:
candidate = W2-planes offline dequant forward (planes already on s4) OR
serve-side prompt_logprobs top-k dump if vLLM allows large k cheaply.
Cheapest correct path: offline candidate forward with the planes dequant
(prepack_planes.py lineage has the codebooks; W2 dequant is exact).

## Row files

- out/DS4_CORPUS_META.json / DS4_CORPUS_MANIFEST.jsonl — corpus identity
- out/R1_NLL_ROWS.jsonl + R1_NLL_ROW.json — R1 serve NLL (streamed + rollup)
- out/r1_positions/win*.json.gz — R1 per-position bank (flip joins)
- out/mmlu_questions_ds4.json + MMLU_QUESTION_SET_DS4.json — MMLU prep
- out/R1_MMLU500_QROWS.jsonl + R1_MMLU500_ROW.json — R1 MMLU
- src/ — all harnesses (md5s embedded in row files)

## Artifact md5s (sealed 2026-07-10, this card)

  R1_NLL_ROW.json        1127a0a142c2ab6ce665c1006564a309
  R1_NLL_ROWS.jsonl      cd54a6b88932fff2057934db942eb6dc
  R1_MMLU500_ROW.json    47637debd8cb1fd308f2fcf6614efb6b
  R1_MMLU500_QROWS.jsonl b6f94bcc374906f77c5983ffa5ae088e
  windows_ds4_eval.json  1701920b4ba96dea0b18fe9df0151876
  DS4_CORPUS_MANIFEST    214f645bd82aa50794a6bc2e5c67f1a9
  mmlu_questions_ds4     c5c34da93b221328d33dbc21963d4515

Coherence gates: PASS before (math/logic/completions probes) and after
(Rayleigh-scattering probe, coherent) the measurement runs. Serve stayed
the only GPU tenant on s4 throughout; c=1 sequential queries only.

## MMLU-500 precision ladder — ONE offline pipeline (t_e3f38867, s8, 2026-07-11)

Banana Bae (Jul11): "What is the Q2 Q3 NVFP4/ref MMLU gap for DS4 - we need that."
All three modes through the SAME v3-builder offline forward
(mmlu_ds4_offline.py, verbatim v3 loader/dequant code paths; readout =
full-vocab log_softmax at real_len-1, DOWNSTREAM_LEDGER choice-loglik
protocol, choice tokens [334,406,345,420], qset sha 24d60b46, 0-shot).
Smoke gate: 12/12 preds == serve R1 golden rows before the ladder ran.

  M-ref (NVFP4-class anchor) : 0.844 (422/500)  gold lp -0.451  margin 3.92
  M-Q2  (W2 RTN 2.25bpw)     : 0.810 (405/500)  gold lp -0.573  margin 2.52
  M-Q3  (W3 RTN 3.25bpw)     : 0.788 (394/500)  gold lp -0.580  margin 2.96

GAPS (paired McNemar, n=500; binomial per-mode CI ±3.2-3.6pt):
  ref−Q3 = +5.6pt  CI [+2.6,+8.6]  p=0.0004  SIGNIFICANT
  ref−Q2 = +3.4pt  CI [+0.4,+6.4]  p=0.036   SIGNIFICANT
  Q3−Q2  = −2.2pt  CI [−5.3,+0.9]  p=0.21    NOT significant — honest
    resolution limit: paired n=500 resolves ~≥3pt; full MMLU-14k would
    give ~±0.6pt if a finer read is ever needed.

READS (pre-registered):
  1. M-Q2 offline 0.810 vs serve 0.802 → +0.8pt < 1.5pt gate: NO offline
     pipeline floor at task level (audit item (c) evidence); 478/500 pred
     agreement.
  2. Q3 ≈ Q2 while ref >> both, AND the pipeline demonstrably resolves
     3-5pt gaps → the shared damage is the sign-sym RTN family itself,
     not expert bit-count. Task-level confirmation of the R2-vs-R1 KLD
     finding (4.2% delta). The recovery axis is grid/scale family or
     calibration (R4/R5, in flight on the same host), not RTN bit-width.
  3. M-ref doubles as the NVFP4 row: DS4 has no public bf16 — the source
     ckpt is 4-bit-native (mxfp4-flavor) routed. A separate modelopt-NVFP4
     requant would get its own row if ever built.

Artifacts: out/mmlu_ladder_s8/ (M_{ref,Q2,Q3}_MMLU500_{ROW.json,QROWS.jsonl},
MMLU_LADDER.json rollup, harness + chain); rail host s8:~/missions/DS4_MMLU
(mirrored s2:~/missions/DS4_MMLU).

### Calibrated rungs M-Q4 / M-Q5 (t_e5898cb2, s8, 2026-07-12)

Same harness verbatim (mmlu_ds4_offline.py md5 db57fd27, qset sha 24d60b46,
0-shot, planes mode) over the t_fa509f27 calibrated GPTQ planes:

  M-Q4 (GPTQ→W2 grid 2.25bpw) : 0.810 (405/500)  gold lp -0.561  margin 2.92
  M-Q5 (GPTQ→W3 grid 3.25bpw) : 0.832 (416/500)  gold lp -0.462  margin 3.58

PAIRED GAPS (McNemar n=500; + paired gold-logprob t as the finer readout):
  Q4−Q2  =  0.0pt  CI [−2.7,+2.7]  p=1.0    n.s.  (Δlp +0.012 nats, t=0.7 — null both readouts)
  Q5−Q3  = +4.4pt  CI [+1.5,+7.2]  p=0.0038 SIGNIFICANT (Δlp +0.118 nats, t=5.6)
  ref−Q4 = +3.4pt  CI [+0.8,+6.0]  p=0.016  SIGNIFICANT (Δlp +0.110 nats, t=4.3)
  ref−Q5 = +1.2pt  CI [−0.9,+3.3]  p=0.36   n.s.  (Δlp +0.010 nats, t=0.5 — ref parity both readouts)
  (Q5−Q4 = +2.2pt n.s. at n=500 but Δlp +0.099 nats t=5.1 — the W3 bit pays
   with calibration on the continuous readout even where accuracy can't resolve it)

READS:
  1. Calibration EARNS the W3 rung at task level: Q5>Q3 significant, and R5
     is the first ladder rung statistically indistinguishable from the
     source-teacher anchor on MMLU-500 (both accuracy and gold-logprob).
     Task column now tracks the KLD column ordinally (0.160→ref-parity).
  2. Honest null: R4's −20.2% KL does NOT translate into a detectable MMLU
     gain over W2-RTN — 24/24 discordant split, and even the continuous
     gold-logprob readout is null (+0.012±0.017 nats). At ~0.31 KL the task
     metric saturates against the RTN baseline; gains that small live below
     the n=500 floor (and below ~0.05 nats on the paired-lp readout).
  3. Ladder (KLD → MMLU): ref —/0.844 · R5 0.160/0.832 · R4 0.312/0.810 ·
     R1 0.390/0.810 · R2 0.374/0.788.

Artifacts: out/mmlu_ladder_s8/M_{Q4,Q5}_MMLU500_{ROW.json,QROWS.jsonl} +
GAPS_GPTQ.json + extended MMLU_LADDER.json (md5 335be4d3, identical s8/s2/
orchestrator-host); gap math src/mmlu_gptq_gaps.py (self-check reproduces the sealed
baseline gaps exactly); chain s8:~/missions/DS4_MMLU/chain_mmlu_gptq.sh.

## W3 design audit (t_eee6b0cc, 2026-07-11, s8) — Banana Bae's design question ANSWERED

Banana Bae (Jul11, verbatim): "Unclear to me whether forcing sign sym is good for
W2 but bad for W3 perhaps, or whether our W3 choices are just suboptimal for
it. As it should absolutely, without question, be significantly better if
done correctly."

VERDICT: **our W3 choices were suboptimal — level placement is the deficit;
sign-sym adds a smaller secondary cost; there was NO scale bug.**

1. SCALE-BUG HYPOTHESIS: REFUTED at byte level (two independent probes,
   t_fa509f27 + t_eee6b0cc re-verify). Shipped moe_w3_planes sc bytes ==
   amax→6.0 derivation exactly; differ from an amax→4.0 fit in 100% of
   blocks. The rms shrinkage 0.893 is LUT geometry: the e2m1 source lattice
   (u ∈ {0,±.5,±1,±1.5,±2,±3,±4,±6}) puts 22% of s²-weighted mass on ±2/±4,
   which the -6..6 ladder snaps −25% in magnitude; plus 3% top-clamp from
   UE8M0 exponent-down rounding.

2. LUT SHOOTOUT (weight-space relRMS, 24 held-out eval matrices, 8 layers,
   both tiers; all arms with per-block SSE-optimal UE8M0 scale fits;
   optimal quantizers = exact interval-DP on the DISCRETE s²-weighted
   e2m1 u-atom histogram, fit experts disjoint from eval experts):
     ship_w2 {-4,-1,1,4}                 0.380   ratio 1.003
     ship_w3 current ladder              0.200   ratio 0.893
     current LUT + MSE scales            0.200   ratio 0.893  <- scales can't fix it
     e2m1-subset {±1,±2,±4,±6}           0.209   ratio 0.979
     uniform8                            0.188   ratio 0.989
     NF3-quantile-8                      0.181   ratio 0.917
     DP sign-sym-optimal (4 mags mirror) 0.169   ratio 0.941
     per-expert DP (headroom bound)      0.180   ratio 0.984
     DP-asym-8 held-out fit  ** WINNER** 0.154   ratio 0.983
   Winner LUT [-6.379,-3.4723,-1.8718,-0.8547,+0.137,+1.4651,+3.4796,+6.3792]
   — asymmetric, near-zero level, covers the 1–2 and 3–4 mass bands,
   endpoints ±6.38 absorb the UE8M0 clamp. A SINGLE global LUT beats the
   per-expert-DP bound because scale refit + asymmetry recover more than
   per-expert placement does — one programmable-LUT kernel constant works.
   Reading on the pilot evidence: at 2 bits placement has no room (sign-sym
   fine); at 3 bits placement dominates (NF3-vs-e2m1 pilot repro'd), and
   lattice-aware DP beats Gaussian NF3 because the source is discrete.

3. Artifacts: s8:~/missions/W3_LUT_AUDIT/ (SHOOTOUT_RESULT.json,
   SHOOTOUT_EXTRA.json, w3_lut_shootout.py, w3_lut_extra_arms.py,
   w3v2_rebuild.py, w3v2_gate.py, chain_w3v2.sh, moe_w3_planes_v2/,
   GATE_W3V2.json when gated); orchestrator-host mirror in the t_eee6b0cc workspace.

## spark-7 testbed relocation (t_db7f8abc, 2026-07-11)

The DS4 tricks testbed moved to spark-7 (<internal-host>, fabric-only) to
free s4 (PP2 reservation) and s6 (GLM G4 solves). Full independent
rebuild on s7 — assets NOT rsynced from the reserved s4:
  - base ckpt 149G rsync'd from s6 over fabric (~7 min)
  - W2 planes prepacked LOCALLY (prepack_planes.py, 73G/215 files);
    layer_003 all-5-file md5 == s4 golden set (bit-identical lineage)
  - W3 planes prepacked LOCALLY (--codebook w3, 105G/43 layers); SDR
    layer-3 w3_prepack_check PASS worst_rel 2.740e-03 (== s4 run-506
    seal); layer_003 planes13/sc13 md5 == s4's W3 set
  - W3 serve glue: moe_w2_cubit.py patched to md5 bdeb3831 (the sealed
    R2 glue), .r1orig = f19d050a preserved
  - stack: vllm-moet 0.24.0 venv (swork recipe), fingerprint
    vllm-0.24.0-2ef3137a at serve, repo checkout 436d2a9.
    Bring-up pitfall fixed: python3.12-dev headers were missing on the
    fresh host -> triton JIT cuda_utils build failed at engine init;
    apt install python3.12-dev fixed it.

R1 CROSS-VALIDATION (s7 serve, G2 recipe verbatim, MTP k=2, planes
fully anon-resident): NLL 1.504492 / MMLU-500 0.802 (401/500) —
per-window sum_logprobs (512/512) AND per-question choice_logprobs
(500/500) byte-identical to the s4 golden rows. Zero diffs. Decode
19.2 tok/s (256 tok single-stream; s4 ref 21.8 — s7 is CPU-governor/
thermal-class variance, not a numerics concern). majflt delta = 0
across the entire battery (residency doctrine PASS).
Artifacts: out/s7_testbed/ (R1 rows md5 b756b169 / c8ba61c5,
battery.log) + spark-7:~/ds4kit/.

R2 serve rows (Q3 W3 RTN) COMPLETE on the s7 W3 serve (BATTERY_R2_DONE
1783818639 = 2026-07-12 00:30:39Z; unit ds4battery-r2, serve pid 98192,
fingerprint vllm-0.24.0-436d2a9). Residency split as designed: layers
0-33 anon ~83G, 34-42 file-backed via VLLM_MOE_W2_PLANES_MMAP=1 /
MMAP_FROM_LAYER=34 (105G W3 planes + ~12G dense > 121G box); kv 1G
pinned, no MTP, MBT 2048 — the sealed s4-run-506 measurement config.
  R2 serve NLL      = 1.499075 (ppl 4.4775, SEM 0.0249, 512/524,288;
                      runtime 592.8 min — mmap page-fault-bound pace)
  R2 serve MMLU-500 = 0.792 (396/500, gold lp -0.582, margin 2.97,
                      qset 24d60b46, runtime 102.5 min)
  per-class NLL: reasoning 0.980 / code 1.194 / agentic 1.323 /
                 chat 1.551 / multilingual 1.815 / prose 2.306
X-checks vs the s8 offline rail: NLL 1.499075 serve vs 1.4877 offline
(0.77% — R1 precedent 0.8%); MMLU 0.792 vs 0.788 (+0.4pt). Instrument
agreement holds on the W3 serve too. majflt: baseline 31,784 -> final
261,524 (delta 229,740) = page-ins of the 9 file-backed layers from
LOCAL NVMe, by design; coherence gates PASS pre (17*23=391) and post.
NOTE: the config string inside the R2 row JSONs says "fully
anon-resident MMAP_FROM_LAYER=99" — stale template text; the launcher
on disk (serve-ds4-w3-r2.sh) and the fault counts prove the =34 split.
Artifacts: s7:~/ds4kit/out/R2_* + battery_r2.log; orchestrator-host mirror
out/s7_testbed/r2_serve/ (md5s: NLL_ROW 6e55b165, NLL_ROWS a1bdf3e8,
MMLU500_ROW cb4849f5, MMLU500_QROWS 0d4bfaa5, log 1884f23e).

## R1/R3 boot-2 verification (t_79a4e035, 2026-07-12, s7 golden W2 serve)

Golden W2 serve re-booted post-R2 (t_ae82e9fb: ds4serve-w2.service,
01:11:42Z, fingerprint vllm-0.24.0-2ef3137a). R1 battery RE-RUN in full on
the fresh boot (unit ds4battery-r1v2):
  NLL 1.504492 / MMLU-500 0.802 (401/500) — per-window sum_logprob
  IDENTICAL 512/512 vs the boot-1 sealed rows (max abs diff 0.0); MMLU
  preds identical 500/500, choice_logprobs byte-identical 490/500 (10
  qids show lp deltas <=0.577, zero flips, gold-lp drift 0.0004 —
  serve-environment class, KV 6.95 vs 7.41 GiB across boots).
R1 is now proven stable across THREE serve instances (s4 golden, s7
boot-1, s7 boot-2). Residency doctrine: majflt bracket around generation
= 0 (boot evidence + fresh warm bracket); battery-long engine delta +46
attributed to file-backed code page-ins (VmSwap=0 => anon planes cannot
major-fault). R3 anchor re-verified live on both rail hosts (s2+s8:
512/512 teacher cache, readback kl=0.0 exact, teacher NLL 1.22157,
M-ref MMLU 0.844). Artifacts: out/s7_testbed/r1v2_boot2/ (md5s:
NLL_ROW 48f6ce8b, NLL_ROWS bbe29666, MMLU500_ROW 0e7d74b0,
MMLU500_QROWS 3d573653, battery log 2ada2260).

## GPQA-Diamond reference rows (OpenRouter, generative protocol — Jul12)

Same sealed 198-question rng(0) set (sha256 adaf4b6d), temp 0, 3 samples/question,
majority vote with sample-0 deterministic tie-break for a three-way split. INSTRUMENT
NOTE: generative protocol ≠ loglik protocol (our serve row 0.4444 is loglik), so do not
subtract across protocols.

### Tier 1 — no-thinking sanity floor (t_d6efa01b)

These calls had reasoning disabled and max_tokens=8. They were NOT truncated 4K runs;
no unanswered sample was scored wrong. Canonical labels are REF-DS4-NOTHINK and
REF-GLM-NOTHINK.

| reference | upstream | acc | consistency |
|---|---|---|---|
| REF-DS4-NOTHINK | deepseek/deepseek-v4-flash | 0.4697 (93/198) | 0.9040 |
| REF-GLM-NOTHINK | z-ai/glm-5.2 | 0.4596 (91/198) | 0.9091 |

### Tier 3 — 4K thinking with forced answer extraction (t_c048049b)

Reasoning budget=4096 tokens. Natural answers are parsed when present; at the cap (or
if no natural letter is produced), a second max_tokens=8 call continues from the
partial reasoning and forces an A-D letter. Thus all 1,188 samples have real parsed
answers; none is treated as wrong merely for reaching the budget. AtlasCloud ignored
DS4's server-side cap, so affected DS4 samples were rerun with the native tokenizer and
a client-side stream stop at exactly 4096 tokens. Novita honored the GLM cap.

| reference | upstream | acc | consistency | forced termination | mean think tokens |
|---|---|---|---|---|---|
| REF-DS4-4K | deepseek/deepseek-v4-flash (AtlasCloud/DeepInfra) | **0.7677 (152/198)** | 0.8081 | 48.82% (290/594) | 2803.3 |
| REF-GLM-4K | z-ai/glm-5.2 (Novita) | **0.7576 (150/198)** | 0.8283 | 41.08% (244/594) | 1832.5 |

Artifacts: out/gpqa_ref_4k/ (sealed ledgers, summaries, rollup, question manifest,
reproduction runners, native DS4 tokenizer). Tier 4 full/high-budget gold remains a
separate future instrument.
