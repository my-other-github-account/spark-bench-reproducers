# Resume the KLD-sharpening campaign in under one hour

This is the pick-up-and-go document. Read [`LEARNINGS.md`](LEARNINGS.md) first, then use this checklist. Public paths use `$MISSION_ROOT`; bind it to the private mission parent on each host.

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

Stop immediately on a corpus, manifest, teacher, plane, or checkpoint hash mismatch. Do not "repair" a receipt to fit the files.

## 15–30 minutes: inspect the last evidence

```bash
python3 tooling/sweeps.py \
  'repair/results/arm3/*.jsonl' \
  'repair/results/arm4/*.jsonl' \
  'repair/results/arm5/*.jsonl' \
  'repair/results/nuclear/*.jsonl' \
  'repair/results/live-snapshots/*/*.jsonl' \
  'repair/altrepair/results/*.jsonl'

python3 tooling/agg_rail.py repair/rail512/shards/*.jsonl \
  --expected-windows 512 --allow-partial
```

Use `repair/SEALED_REPAIR_REPLICATION.json` for formal pooled-KLD claims. Use the trajectory printer for LR/step decisions. The two percentage conventions are intentionally both preserved.

## 30–60 minutes: stage the next arm

1. Claim one idle host and verify there is no GPU co-tenant.
2. Copy the exact manifest, delta pack, 43 base planes, teacher receipt, corpus, and chosen checkpoint.
3. Run the instant shape/hash/corpus gates.
4. Run one short mechanics smoke.
5. Launch with explicit environment overrides; do not edit staged defaults by hand.
6. Confirm the first status row, activation-cache identity, and step-0 baseline before leaving it unattended.

## Current artifact map

### Repository copies

- Formal replication seal: `repair/SEALED_REPAIR_REPLICATION.json`
- Generated probe tables: `repair/PROBE_TABLES.{md,json}`
- Raw finalized arms: `repair/results/{arm3,arm4,arm5,nuclear}/`
- Later/partial arm snapshots: `repair/results/live-snapshots/{arm6,arm7,arm8,arm9,arm10}/`
- RMSNorm and output-scale arms: `repair/altrepair/results/`
- 24-window gate notes: `repair/external-gate/README.md`
- Sealed 512-rail row, paired clean split, and shard receipts: `repair/rail512/`
- Carry-through tools: `tooling/`
- Serving receipts and baseline A/B material: `serving/`

### Fleet source-of-truth paths at the refresh cut

| host | authoritative campaign paths |
|---|---|
| `spark-1` | `$MISSION_ROOT/BINREPAIR_t_2956f863/out/BINREPAIR_arm4_all43_lr1e2.*`; arm9 trajectory; `$MISSION_ROOT/RAIL512/` shard 0–72 |
| `spark-2` | `$MISSION_ROOT/BINREPAIR_t_890f5f95/`; `$MISSION_ROOT/RAIL512/` shard 73–145 |
| `spark-3` | arm6 and arm10 trajectories; `$MISSION_ROOT/RAIL512/` shard 146–218 staging/retry |
| `spark-5` | `$MISSION_ROOT/ALTREPAIR_t_7a65a4c6/` RMSNorm source/replica; `$MISSION_ROOT/SERVED_AB/remeasure24/`; `$MISSION_ROOT/RAIL512/` shard 292–364 |
| `spark-6` | arm8 trajectory; `$MISSION_ROOT/RAIL512/` shard 219–291 |
| `spark-7` | arm7 trajectory; `$MISSION_ROOT/RAIL512/` shard 365–437 |
| `spark-8` | arm5 source, normalized seal source, exported arm4 planes, and `$MISSION_ROOT/RAIL512/` shard 438–511 |

The exported arm4 rail subsequently sealed exact windows 0–511 with no conflicting duplicates.
`repair/rail512/RAIL512_ARM4_FINAL.json` is the authoritative row and
`repair/rail512/SEALED_SUMMARY.json` separates the full, training, clean, and per-domain results.
Keep the historical shard paths above only as provenance; do not relaunch this rail to debug a new artifact.

## Exact next-run plan

### Phase 1 — selective VQ-GPTQ rebuild

- Rebuild GPTQ code assignments for all 43 layers against the same pinned activation/corpus receipts.
- Keep codebooks, scales, wire budget, and evaluator fixed.
- Emit per-layer/per-projection nearest-vs-GPTQ validation rows.
- Adopt GPTQ only where held-out KLD wins; retain nearest assignment elsewhere.
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

The staged mission must preserve the repository layout expected by `run_pilot.sh`, including its host claim. The five historically hardcoded values are now overridable.

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

For RMSNorm, use `repair/altrepair/code/run_norm_tune.sh` with `1e-4` as the known-good center. Do not transfer the codebook LR.

## Copy-paste: launch the seven-way rail

Generate disjoint shard lists once, then launch one command on each claimed host:

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

A nonzero exit means the rail is not sealed.

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

`TWOBIN_DELTA_DIR` and the streaming/override receipts are mandatory. The first smoke failed because the delta pack was not bound. Run a four-window smoke, compare evaluator output to the immutable ledger, then launch the external panel or fleet rail on the sealed streamer. Do not use the 2.2-minute/window training evaluator for the full rail.

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

Before throughput, compare one fixed prompt's prefill logits and the same external NLL windows. The existing `serving/WIRE_GATE_win0.json` records why the first served repair was not claimable; do not bypass it.

## Definition of done for the next iteration

- exact corpus/teacher/manifest/plane/checkpoint hashes;
- disjoint train, binding probe, and external-gate windows;
- 2–3 above-floor replicas;
- export and offline/served parity;
- one complete 512-window rail;
- per-domain rows with multilingual regressions called out;
- updated `LEARNINGS.md`, `RESUME.md`, and scrubbed receipts committed together.
