# Resume the KLD-sharpening campaign in under one hour

This is the pick-up-and-go document. Read [`LEARNINGS.md`](LEARNINGS.md) first, then use this checklist. Public paths use `$MISSION_ROOT`; bind it to the private mission parent on each host. Headline tables live in [`RESULTS.md`](RESULTS.md); every number there traces to a JSON receipt.

## Status at this cut (2026-07-16)

**SEALED — do not re-run to debug anything new:**

- **Repair replication seal** — `repair/SEALED_REPAIR_REPLICATION.json`, verdict `REPLICATED_ABOVE_FLOOR_HELDOUT_KLD_REPAIR_ZERO_LEAKAGE`, baseline `0.056753578`, 8 disjoint probes `[4,84,160,236,304,373,442,511]`, zero leakage. Canonical fleet copy is on **spark-8** at `$MISSION_ROOT/BINREPAIR_t_2956f863/out/` (spark-1 holds the arm4 *source outputs*, not the seal JSON — earlier briefs placing the seal on spark-1 are wrong).
- **arm4** (spark-1, lr 1e-2, 16w): **+6.4218% mean-window / +5.6473% pooled final** (+6.2215% pooled best, step 48). **arm5** (spark-8, lr 3e-3, 64 disjoint w): **+5.9809% mean-window / +5.3413% pooled** (step-32 lower bound). The "+6.42/+5.98" numbers in briefs are the mean-window convention; keep both conventions labeled, never mix them.
- **RMSNorm-gamma orthogonal replication**: **+11.1590/+10.8783 mean-window = +13.5531/+13.4922 pooled** (source b2 / rep1_rot8). Sealed rows in `repair/sealed-rows/`.
- **24-window external gate** — FINAL at **~+5.1% held-out** (`repair/external-gate/CAMPAIGN_RECORD.json`). The independent remeasure was killed by the dead-arm order; no further remeasure is coming. Claims-grade exported-artifact evidence is the 512 rail.
- **512-window rail — SEALED 2026-07-16 12:5x PDT.** Authoritative row `repair/rail512/RAIL512_ARM4_FINAL.json` + `SEALED_SUMMARY.json`. FULL 512: `0.098950 -> 0.092240` (**+6.781%**, top-1 0.9100). CLEAN 496 train-excluded, paired: `0.099431 -> 0.094284` (**+5.176%**, top-1 0.9086). TRAIN 16: +65.6% — train contamination, excluded from claims. 454/496 clean windows improved (91.5%), median +5.10%. Versus the UD-IQ4_XS bar `0.0927`: FULL passes by 0.50%; CLEAN `0.0943` is 1.7% above the bar — **T1 stays OPEN on the clean row.** Use CLEAN for generalization claims, always.

**IN-FLIGHT — update this file and RESULTS.md when each seals:**

1. **Q2 companion rail** (t_149edab4 Part B): live on spark-1 (windows 0–255) + spark-7 (windows 256–511), tag `arm4v2q2`, Q2 bin 95.75 GB manifest. Tests whether arm4 carries to the smaller Q2 bin. Not in the repo yet — when it seals, add the measured row + shard receipts and a RESULTS.md row.
2. **arm8** (spark-6): the ONLY surviving training arm. Refresh `repair/results/live-snapshots/arm8/` when it completes. Do NOT relaunch a successor without an explicit order.
3. **IQ3 full-menu measured rail** (t_a7e65a83, spark-4): converts PRED `0.088589` (T1 PASS by −4.44%) into a measured 512-window row that should supersede the k4096-menu row. New RESULTS.md row when sealed.
4. **Bonsai claims-grade re-rail** (t_fdf1edbd): the banked Q1_0 row is a self-consistency footnote ONLY (scored vs Bonsai's own F16 export). Do not publish Bonsai compression numbers until the re-rail vs the Qwen base teacher lands.

## 0–15 minutes: verify identity and receipts

```bash
cd glm52-ds4-w23-planes-quant
export MISSION_ROOT=/path/to/campaign-missions
export CORPUS_JSON=/path/to/windows_ds4_eval.json
export ECORPUS="$CORPUS_JSON"
md5sum "$ECORPUS"
# expected campaign corpus: 1701920b4ba96dea0b18fe9df0151876

python3 tools/scrub_audit.py
python3 eval/validate_published_artifacts.py
python3 tooling/fix_runpilot_env.py --check repair/code/run_pilot.sh
```

Stop immediately on a corpus, manifest, teacher, plane, or checkpoint hash mismatch. Do not "repair" a receipt to fit the files — two corpus poisonings taught us this the hard way.

## 15–30 minutes: inspect the last evidence

```bash
python3 tooling/sweeps.py \
  'repair/results/arm3/*.jsonl' \
  'repair/results/arm4/*.jsonl' \
  'repair/results/arm5/*.jsonl' \
  'repair/results/nuclear/*.jsonl' \
  'repair/results/live-snapshots/*/*.jsonl' \
  'repair/altrepair/results/*.jsonl'

# The 512 rail is sealed: strict aggregation must pass with no flags.
python3 tooling/agg_rail.py repair/rail512/shards/*.jsonl --expected-windows 512
```

Use `repair/SEALED_REPAIR_REPLICATION.json` for formal pooled-KLD claims and `sweeps.py` for LR/step decisions. Reserve `--allow-partial` for monitoring a NEW in-flight rail; partial metrics are never release evidence.

## 30–60 minutes: stage the next arm

1. Claim one idle host and verify there is no GPU co-tenant (`run_pilot.sh` refuses without a valid `HOST_CLAIM` owner and free memory).
2. Copy the exact manifest, delta pack, 43 base planes, teacher receipt, corpus, and chosen checkpoint.
3. Run the instant shape/hash/corpus gates.
4. Run one short mechanics smoke.
5. Launch with explicit environment overrides; do not edit staged defaults by hand.
6. Confirm the first status row, activation-cache identity, and step-0 baseline before leaving it unattended.

## Current artifact map

### Repository copies

- Formal replication seal: `repair/SEALED_REPAIR_REPLICATION.json`
- Sealed measured rows (both conventions): `repair/sealed-rows/`
- Generated probe tables: `repair/PROBE_TABLES.{md,json}`
- Raw finalized arms: `repair/results/{arm3,arm4,arm5,nuclear}/` (arm3 excluded — faded to +2.5709% final, below the ±2.6% floor; pilots sub-floor zero from the 3-layer capacity bug)
- Later/partial arm snapshots: `repair/results/live-snapshots/{arm6,arm7,arm8,arm9,arm10}/`
- RMSNorm and output-scale arms: `repair/altrepair/{code,results}/` (output-scale diverges at lr 1e-2 — published as a negative result)
- 24-window gate record: `repair/external-gate/`
- Sealed 512-rail row, paired clean split, domain map, and shard receipts: `repair/rail512/`
- Checkpoint SHA-256 receipts (checkpoints themselves are NOT committed): `repair/CHECKPOINT_MANIFEST.json`
- Carry-through tools: `tooling/`
- Serving receipts and baseline A/B material: `serving/`

### Fleet source-of-truth paths at this cut

| host | authoritative campaign paths |
|---|---|
| `spark-1` | arm4 source outputs `$MISSION_ROOT/BINREPAIR_t_2956f863/out/BINREPAIR_arm4_all43_lr1e2.*`; `$MISSION_ROOT/RAIL512/` incl. `arm4.best.pt` + `ARM4_OVERRIDE_RECEIPT.json` and the merged-row + ledger receipts in `RAIL512/MERGE_ARM4_IQ3_512/`; Q2 companion rail (windows 0–255, live) |
| `spark-2` | `$MISSION_ROOT/BINREPAIR_t_890f5f95/`; historical rail shard 73–145 |
| `spark-3` | arm6 `best.pt` (banked after disk-full SIGKILL — do NOT relaunch) and arm10 trajectory; historical rail shard staging |
| `spark-4` | IQ3 full-menu measured rail build (in-flight) |
| `spark-5` | `$MISSION_ROOT/ALTREPAIR_t_7a65a4c6/` RMSNorm source/replica; `$MISSION_ROOT/SERVED_AB/remeasure24/` (tombstoned); historical rail shard 292–364 |
| `spark-6` | arm8 trajectory (LIVE, only surviving arm); historical rail shard 219–291 |
| `spark-7` | arm7 trajectory (killed); Q2 companion rail (windows 256–511, live); historical rail shard 365–437 |
| `spark-8` | **seal JSON canonical** (`$MISSION_ROOT/BINREPAIR_t_2956f863/out/SEALED_REPAIR_REPLICATION.json`); arm5 source; exported arm4 planes `$MISSION_ROOT/SERVED_AB/planes_arm4/`; historical rail shard 438–511 |

The exported arm4 rail sealed exact windows 0–511 with no conflicting duplicates. Keep the historical shard paths only as provenance; do not relaunch this rail to debug a new artifact — a new artifact gets its own rail tag.

## Exact next-run plan

### Phase 1 — selective VQ-GPTQ rebuild

- Rebuild GPTQ code assignments for all 43 layers against the same pinned activation/corpus receipts.
- Keep codebooks, scales, wire budget, and evaluator fixed.
- Emit per-layer/per-projection nearest-vs-GPTQ validation rows.
- Adopt GPTQ only where held-out KLD wins; retain nearest assignment elsewhere (GPTQ benefit is layer-dependent).
- Hash every emitted plane and produce one complete-plane receipt.

Exit: a selective 43-layer plane set passes shape/hash/decode parity and has a measured external improvement, not only raw relRMS.

### Phase 2 — re-solve once

- Add the selective GPTQ rung and every sealed d4/d8 rung to one menu.
- Re-solve both 95.75 GB and 101.95 GB caps.
- Reject manifests with unmeasured anchors or mixed metric/corpus identities.
- Build only the Pareto candidates; do not rail every solver hypothesis.

Exit: exact byte-capped manifests with receipts and a short smoke against the canonical ledger.

### Phase 3 — combo repair on 256–384 windows

Run 2–3 independent replica arms for each promoted candidate. Use 256–384 disjoint training windows and all viable parameter classes:

- VQ codebooks;
- RMSNorm gamma;
- low-rate output scales;
- selective GPTQ assignments where the representation permits alternation.

Bracket LR per parameter class. Preserve best/latest separately and bind every activation cache to corpus + manifest + checkpoint hashes.

Exit: at least two replicas exceed the `2.6%` floor on the same external panel without train/probe overlap.

### Phase 4 — alternate, then seal once

Alternate discrete assignment and continuous repair rather than summing independent percentages:

1. selective assignment;
2. codebook/norm repair;
3. reassign only where validation wins;
4. short repair refresh;
5. export and parity gates;
6. one fleet-sharded 512-window seal.

Exit: one hash-complete artifact and one sealed release rail. Do not spend a second 512 rail to debug a failed carry-through gate.

## Copy-paste: launch a repair arm

The staged mission must preserve the repository layout expected by `run_pilot.sh`, including its host claim. The five historically hardcoded values (`BR_VQ3B_DIR`, `BR_TRAIN`, `BR_PROBE`, `BR_REF_KLD`, `BR_OUTDIR`) are env-overridable; the hardcoding cost three relaunches, so always run `--check` first.

```bash
export MISSION_ROOT=/path/to/campaign-missions
cd "$MISSION_ROOT/BINREPAIR_t_2956f863"
python3 /path/to/repo/tooling/fix_runpilot_env.py --check code/run_pilot.sh

BR_VQ3B_DIR="$MISSION_ROOT/SERVED_AB/planes_candidate" \
BR_TRAIN="7,44,86,118,151,186,217,250,282,313,348,377,409,441,472,505" \
BR_PROBE="4,84,160,236,304,373,442,511" \
BR_REF_KLD="$MISSION_ROOT/BINREPAIR_t_2956f863/code/ledger_ref.json" \
BR_OUTDIR="$MISSION_ROOT/BINREPAIR_t_2956f863/out/next-arm" \
BR_TRAINABLE="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42" \
BR_TAG="combo_replica_a" BR_STEPS=96 BR_LR=1e-2 BR_BATCH=2 \
bash code/run_pilot.sh
```

For Phase 3, replace `BR_TRAIN` with the 256–384-window disjoint list and keep `BR_PROBE` disjoint from it. For RMSNorm, use `repair/altrepair/code/run_norm_tune.sh` with `1e-4` as the known-good center. Do not transfer the codebook LR across parameter classes.

## Copy-paste: launch the seven-way rail

`tooling/rail512_shard.sh` exports arm4 planes locally if `EXPORT_META.json` is missing, then evaluates the assigned windows through the patched `run_pilot.sh` at `BR_STEPS=0` (pure measurement). Generate disjoint shard lists once, then launch one command on each claimed host:

```bash
export MISSION_ROOT=/path/to/campaign-missions
REPO=/path/to/repo/glm52-ds4-w23-planes-quant

bash "$REPO/tooling/rail512_shard.sh" "0,1,...,72" spark-1
# spark-2: 73,...,145
# spark-3: 146,...,218
# spark-6: 219,...,291
# spark-5: 292,...,364
# spark-7: 365,...,437
# spark-8: 438,...,511
```

Collect `BINREPAIR_rail512_*.jsonl` into `repair/rail512/shards/`, then:

```bash
python3 "$REPO/tooling/agg_rail.py" repair/rail512/shards/*.jsonl \
  --expected-windows 512 --output repair/rail512/AGGREGATE.json
```

A nonzero exit means the rail is not sealed. The aggregator fails closed on duplicate conflicts, missing windows, or mixed manifest/corpus identities — that is the gate, do not weaken it.

## Copy-paste: export and external-eval loop

```bash
export MISSION_ROOT=/path/to/campaign-missions
export ARM4_CKPT="$MISSION_ROOT/BINREPAIR_t_2956f863/out/BINREPAIR_arm4_all43_lr1e2.best.pt"
export BASE_PLANES_DIR="$MISSION_ROOT/BINREPAIR_t_2956f863/planes"
export EXPORTED_PLANES_DIR="$MISSION_ROOT/SERVED_AB/planes_arm4"

python3 tooling/export_arm4.py \
  --checkpoint "$ARM4_CKPT" \
  --base-planes "$BASE_PLANES_DIR" \
  --output "$EXPORTED_PLANES_DIR"

export TWOBIN_DELTA_DIR="$MISSION_ROOT/TWOBIN/delta_packs/IQ3_BIN"
export VQ3U_STREAM_CACHE="$MISSION_ROOT/SERVED_AB/stream-cache"
export VQ3U_STREAM_RECEIPT="$MISSION_ROOT/receipts/CANONICAL_PLANES.json"
export VQ3U_STREAM_LEDGER="$MISSION_ROOT/SERVED_AB/STREAMED_PLANES_VERIFICATION.jsonl"
export VQ3U_STREAM_S8="spark-8:$MISSION_ROOT/VQ3_K4096/planes"
export VQ3U_STREAM_S6="spark-6:$MISSION_ROOT/VQ3_K4096/planes"
export VQ3U_OVERRIDE_DIR="$EXPORTED_PLANES_DIR"
export VQ3U_OVERRIDE_RECEIPT="$EXPORTED_PLANES_DIR/EXPORT_META.json"
```

`TWOBIN_DELTA_DIR` and the streaming/override receipts are mandatory — the first smoke failed solely because the delta pack was not bound. `VQ3U_OVERRIDE_DIR` is honored only when `VQ3U_OVERRIDE_RECEIPT` is set; the source accepts both export (`md5`) and canonical (`canonical_md5`) receipt schemas and fails closed on a missing digest. Run a four-window smoke, compare evaluator output to the immutable ledger, then launch the external panel or fleet rail on the sealed streamer. Never run the full rail on the 2.2-minute/window training evaluator — the sealed streamer is 16.5 s/window, an ~8x difference that decides whether a rail takes hours or days.

## Copy-paste: served A/B

Do this only after trainer → exported → offline parity passes.

```bash
# Baseline
bash scripts/serve_arm.sh planes_baseline 262144 3221225472
python3 serving/run_spark8_tps.py

# Stop the baseline cleanly, keep every launch parameter fixed, then candidate
bash scripts/serve_arm.sh planes_arm4 262144 3221225472
python3 serving/run_spark8_tps.py
```

Before throughput, compare one fixed prompt's prefill logits and the same external NLL windows. `serving/WIRE_GATE_win0.json` records why the first served repair was not claimable; do not bypass it.

## Hard-won operational pitfalls

1. **Corpus identity is sacred.** Two corpus poisonings occurred; the `ECORPUS` + md5 (`1701920b...`) pattern exists because of them. Hash mismatch = full stop. Never adjust a receipt to match the files.
2. **The five `run_pilot.sh` env overrides cost three relaunches** before they were parameterized. Always `fix_runpilot_env.py --check` before launch; a second patch run must print `NOCHANGE`.
3. **`TWOBIN_DELTA_DIR` is a separate mandatory binding** from the override pair. Smoke v1 failed by omitting it. Override dir without override receipt is rejected.
4. **Always eval on the sealed streamer** (16.5 s/window), never the training harness (~2.2 min/window), for anything bigger than a smoke.
5. **LR is per parameter class and monotone for codebooks**: 1e-2 > 3e-3 > 1e-3 (nuclear dose-response: lr 1e-3 gave +0.84% vs +5.65% at 1e-2). Norms want 1e-4. Output scales diverge at 1e-2. Judge competing arms by steps×lr displacement, not wall-clock.
6. **Capacity was the bug, not the method**: all-43-layer arms hit +6.4% where 3-layer pilots flatlined at ~+0.7%. Do not launch reduced-layer pilots expecting representative signal.
7. **Two percentage conventions coexist** (mean-window vs pooled). Both are published side by side; label every number or it will be misquoted.
8. **Dead arms stay dead.** arm6's `best.pt` is banked on spark-3 after a disk-full SIGKILL — do not relaunch it; arm7/arm9/arm10 were killed by explicit order; arm8 successors require an explicit order.
9. **Train-window contamination inflates results wildly** (+65.6% on the 16 training windows vs +5.18% clean). Claims come from the train-excluded paired subset only; T1 is judged on the clean row.
10. **One seal rail per artifact.** Parity gates (shape/hash/decode, offline-vs-trainer, wire gate) come first; a 512-window rail is the last step, not a debugging tool.
11. **Host discipline**: `run_pilot.sh` hard-refuses without a `HOST_CLAIM` owner match and staging receipt; verify no GPU co-tenant before claiming. Confirm step-0 baseline and activation-cache identity before leaving an arm unattended.
12. **Partial aggregates are for monitoring only.** `agg_rail.py` without flags is the seal gate; `--allow-partial` output must never appear in a results table.

## Definition of done for the next iteration

- exact corpus/teacher/manifest/plane/checkpoint hashes;
- disjoint train, binding probe, and external-gate windows;
- 2–3 above-floor replicas;
- export and offline/served parity;
- one complete 512-window rail with the clean train-excluded split reported;
- per-domain rows with multilingual regressions called out;
- updated `LEARNINGS.md`, `RESUME.md`, `RESULTS.md`, and scrubbed receipts committed together.
