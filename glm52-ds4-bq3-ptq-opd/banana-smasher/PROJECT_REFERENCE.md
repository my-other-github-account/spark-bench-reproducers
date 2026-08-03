# Banana Smasher project reference

> **Status date:** 2026-08-02  
> **Purpose:** Single operational and research reference for taking a new model through Banana Smasher quantization/repair, packaging it as a fail-closed `bs-pack`, serving it with vLLM, and producing a defensible competitive comparison report.

## 1. Executive summary

Banana Smasher is a reproducible mixed-precision model-production system. Its intended product is not merely a quantizer: it is an end-to-end, receipt-bound path from a source model to a deployable OpenAI-compatible API and a publication-grade quality/size/speed report.

The project combines:

1. calibration and teacher-bank capture;
2. exact QTIP/vector-quantization solving;
3. anchor and byte-envelope composition;
4. repair training;
5. deterministic pack export and validation;
6. stock-semantics vLLM serving; and
7. paired quality, behavioral, footprint, latency, and throughput evaluation.

The current campaign has an accepted U012-backed service on Spark-8, but the clean public-source final-form serving lane and several generic production stages remain incomplete. This document distinguishes **implemented**, **tested**, **physically proven**, **deployed**, and **accepted** throughout.

## 2. GitHub identity and repository

### Repository

- GitHub account/owner: [`my-other-github-account`](https://github.com/my-other-github-account)
- Repository: [`my-other-github-account/spark-bench-reproducers`](https://github.com/my-other-github-account/spark-bench-reproducers)
- Project directory: `glm52-ds4-bq3-ptq-opd/banana-smasher`
- SSH remote: `git@github.com:my-other-github-account/spark-bench-reproducers.git`
- Public development branch: `t_63769bff-public-source`
- Inspected branch head: `834e1b34c1f0cb11c9d318efd928c0aa632de1bd`
- Head subject: `fix(plugin): bind complete sparse indexer API`

### Active identity on the production Mac

The SSH key active for `git@github.com` authenticates as:

```text
my-other-github-account
```

The repository-local commit identity is:

```text
banana_baeee <banana_baeee@users.noreply.github.com>
```

The current `gh` CLI home is not logged in. Git fetch/push access is supplied by SSH, not by the current `gh` profile. Do not assume that `gh api` is authenticated merely because `git push` works.

### Publication state

At the status date, `t_63769bff-public-source` was diverged from `main`:

- 40 commits ahead;
- 12 commits behind;
- not merged into `main`; and
- not accepted as a complete clean-source stock-service build.

The branch contains substantial public runtime/plugin work, but branch availability is not equivalent to final deployment acceptance.

## 3. Project objective

The project is trying to produce a model that is simultaneously:

- materially smaller than the FP8/source artifact;
- competitive with or better than public IQ3/IQ4/Q2-style quantizations;
- low-loss under a same-teacher, same-corpus KLD protocol;
- behaviorally competitive on coding, knowledge, and tool-use evaluations;
- fast enough to serve as a practical API on Spark-class hardware;
- reproducible from public source and immutable inputs;
- resumable after process or host interruption; and
- accompanied by hashes, receipts, commands, and honest status labels sufficient for an independent reviewer.

The deliverable is therefore a **model + pack + runtime + API + comparison report**, not any one of those in isolation.

## 4. Competitive landscape and targets

### 4.1 Campaign size/quality targets

The historical campaign defines:

| Target | Requirement | Meaning |
|---|---:|---|
| **T1** | KLD `< 0.0927` at `≤ 101.95 GB` | IQ3-size quality target |
| **T2** | KLD `< 0.0927` at `≤ 95.75 GB` | smaller Q2-size target |
| **NVFP4 reference bar** | KLD `≤ 0.05936` and top-1 `≥ 93.01%` | demanding external lossless-grade reference; different model/protocol, not a formal same-row DS4 delta |

The canonical KLD direction is:

```text
KL(teacher || candidate)
```

For the main DS4 rail it is evaluated on the teacher top-8,192 support, renormalized, over the first `min(1024, real_len - 1)` positions of each common window.

### 4.2 Same-basis Flash-Full 0731 competitors

These rows use the exact DeepSeek-V4-Flash-0731 FP8 teacher under basis prefix `98efab45` and the same BALANCED64 protocol:

| Artifact | Declared bytes | BALANCED64 KLD | Status |
|---|---:|---:|---|
| Unsloth `UD-IQ4_XS` 0731 | `136,662,446,656` | `0.0683488486737012` | sealed 64/64 |
| Unsloth `UD-IQ3_XXS` 0731 | `104,207,848,032` | `0.17770788160865483` | sealed 64/64 |
| DwarfStar asymmetric Q2 mix | `86,720,111,200` staged artifact bytes | `0.30952134732070036` | sealed 64/64 |
| Unsloth IQ2 0731 | approximately 90.86–90.93 GB variants exist upstream | pending | no canonical own-base pack selection/seal |
| Our BQ3-0731 backpack, pre-repair | pending final STEP5 pack | pending | anchor/knapsack/export gap |
| Our BQ3-0731 backpack, post-repair | pending pre-repair artifact | pending | future comparison row |

Current same-basis top-1 agreement receipts:

| Artifact | Candidate top-1 agreement | Teacher agreement | Candidate/teacher mismatch |
|---|---:|---:|---:|
| DwarfStar | `99.508667%` | `99.508667%` | `0%` |
| IQ3 | `99.868774%` | `99.868774%` | `0.131226%` |
| IQ4 | `99.533081%` | `99.681091%` | `0.307861%` |

### 4.3 Existing Banana Smasher quality evidence

On the earlier common DS4 teacher rail, the strongest deployable IQ3-size repair result is:

- artifact: repaired IQ3 COMBO;
- size: `101.95 GB`;
- KLD: `0.077061044921875`;
- top-1: `0.916632`;
- JS: `0.016803`;
- paired improvement from `0.0989496484375`: `22.1210%`;
- improved windows: `501/512`;
- T1 verdict: pass by `16.87%`.

Associated behavioral rows include:

- ToolEvalBench: `86.60 ± 1.20`, N=5;
- HumanEval: `95.73%`;
- HumanEval+: `90.85%`;
- MMLU-500: `85.0%` (`425/500`).

These results demonstrate the method, but they must not be silently mixed with the newer exact-0731 BALANCED64 rail. Every report must name the teacher, corpus, evaluator, basis hash, and candidate artifact for every row.

### 4.4 U012 deployment decision

The U012 service candidate passed its deployment quality gate:

- U012 HOLDOUT512 KLD: `0.0529736484`;
- shipped comparison: `0.05418`;
- relative improvement: `2.22656%`;
- campaign decision: activate U012/V5 and skip the V4-era upload.

This is a separate deployment gate. It must not be presented as though it were the same cell as the exact-0731 BALANCED64 competitor table.

## 5. System architecture

### 5.1 Physical production stages

1. **Base-model seal** — immutable model index, shards, tokenizer, config, architecture, and source revision.
2. **Teacher/calibration bank** — exact token windows, class map, teacher support/logits, hashes, and exposure policy.
3. **Capture/Hessian preparation** — resumable physical inputs for solving and repair.
4. **Tier solving** — QTIP2/QTIP3 and true-VQ families solved per layer, expert, projection, and declared tier.
5. **Anchoring and damage pricing** — calibration/rebalance surfaces used to compare tiers.
6. **Knapsack composition** — deterministic assignment under an exact byte envelope.
7. **Repair update** — resumable correction of codebooks, normalization values, and serving-side gains.
8. **Pack export** — selection, row packing, repair binding, base metadata, manifest-last publication, and self-verification.
9. **Runtime preflight** — pack/kernel ABI and architecture checks.
10. **Serving** — stock-semantics vLLM with the Banana Smasher quantization/plugin contract.
11. **Acceptance/evaluation** — health, model listing, coherent generations, load ladder, KLD, top-1, behavioral metrics, and provenance seal.

### 5.2 `bs-pack v1`

`bs-pack` is the versioned boundary between the quantizer and serving runtime. The frozen contract is documented in [`PACK_FORMAT.md`](PACK_FORMAT.md).

Key properties:

- schema/version and quantization method are explicit;
- model and pack instance identities are explicit;
- every regular file has an exact byte count and SHA-256;
- unknown or extra files fail validation;
- symlinks are forbidden;
- layer metadata and tensor layout are manifest-bound;
- tier maps retain global expert IDs;
- source planes can be hard-linked, copied, or repacked to safetensors;
- safetensors conversion must retain tensor name, dtype, shape, and raw payload SHA-256;
- serving configuration auto-selects the Banana Smasher method; and
- pack/kernel architecture or ABI mismatch fails before model allocation.

The public three-step product path is:

```bash
smash export ...
smash validate-pack /model
vllm serve /model
```

## 6. Public versus campaign command surfaces

### 6.1 Public branch surface

The current public branch provides four primary verbs plus the `validate-pack` compatibility spelling:

```text
export
verify
serve-check
validate
validate-pack  # alias of verify
```

`export` supports:

- immutable source root;
- serving-model metadata root;
- model and pack instance IDs;
- hardlink/copy/auto materialization;
- safetensors repack and verified plane removal;
- metadata-only refresh;
- runtime-floor accounting; and
- SHA-bound repair checkpoint, overlay, assignment, and update materialization.

### 6.2 Expanded campaign surface

The latest consolidated campaign wheel/source contains:

```text
export
verify
serve-check
validate
bootstrap
solve
qtip-configs
hessian
capture
update
anchor
knapsack
knapsack-index
status
bank
evaluate
```

This expanded surface is the intended full pipeline, but it is not yet equivalent to the clean public branch. A new operator must record the wheel/source commit used and must not assume that installing public `main` exposes every campaign verb.

### 6.3 Current command status

| Command/group | Current evidence |
|---|---|
| `export`, `verify`/`validate-pack` | physically accepted on U012 exports |
| `bank`, `evaluate` | physically used for IQ3, IQ4, and Dwarf KLD workflows |
| `solve` | proven for campaign QTIP2/QTIP3; generic arbitrary-ring physical closure incomplete |
| `qtip-configs` | materializes hash-bound tier/ring configs |
| `capture`, `hessian` | implemented and integrated; clean generalized timing/acceptance incomplete |
| `update` | resumable and parity-tested; new physical speed target unproven |
| `anchor` | implemented; current production composition lacks all required sealed producers |
| `knapsack-index`, `knapsack` | correctly fail closed when basis, anchor manifests, or damage rows are missing |
| `bootstrap` | multiple source images built; final clean-source stock API did not reach acceptance |
| `serve-check`, `validate`, `status` | implemented/tested; physical acceptance depends on the lane |

## 7. End-to-end new-model playbook

This section is the required procedure for a new human or agent. Replace placeholders deliberately; do not infer missing identities.

### Phase 0 — establish ownership and immutable roots

Record:

```text
MODEL_ID
MODEL_ROOT
MODEL_INDEX_SHA256
SOURCE_REVISION
TOKENIZER_HASHES
ARCHITECTURE
TARGET_DEVICE_ARCHITECTURE
PACK_INSTANCE_ID
RUN_ROOT
TEACHER_MODEL_ROOT
CORPUS_MANIFEST
CLASS_MAP
BYTE_ENVELOPE
```

Required gates:

- source model index and every referenced shard exist;
- architecture and expert geometry are supported;
- tokenizer/config files are present and hash-bound;
- teacher and candidate basis relationship is explicit;
- calibration/evaluation windows are immutable;
- the training/evaluation exposure policy is declared; and
- host/GPU ownership is acquired before physical work.

Never select an ambiguous model variant merely because it is available upstream. The IQ2 lane is intentionally blocked for exactly this reason.

### Phase 1 — obtain the source and pin the tool

```bash
git clone git@github.com:my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/glm52-ds4-bq3-ptq-opd/banana-smasher
git checkout t_63769bff-public-source
git rev-parse HEAD

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the full campaign workflow, install the explicitly sealed expanded wheel/source instead and record its SHA-256 and source commit. Do not mix modules from different revisions through an ambient `PYTHONPATH`.

### Phase 2 — build the teacher and calibration evidence

Expanded campaign form:

```bash
smash bank \
  --model-root "$TEACHER_MODEL_ROOT" \
  --corpus "$CORPUS" \
  --windows-manifest "$WINDOWS_MANIFEST" \
  --output "$RUN_ROOT/teacher-bank"
```

The resulting bank must bind:

- model/basis hash;
- corpus/window identities;
- token IDs and position count;
- class map;
- teacher support/logits convention;
- evaluator version; and
- complete member hashes.

A smoke/gate bank can guide decisions, but only complete required coverage may become a canonical leaderboard row.

### Phase 3 — capture and Hessian preparation

```bash
smash capture \
  --run-root "$RUN_ROOT" \
  --model-root "$MODEL_ROOT" \
  --meta-root "$META_ROOT" \
  --corpus "$CORPUS" \
  --builder "$CAPTURE_BUILDER" \
  --layers 0-42 \
  --windows 64 \
  --microbatch 4 \
  --detach
```

```bash
smash hessian \
  --run-root "$RUN_ROOT" \
  --layers 0-42 \
  --windows 64 \
  --detach
```

Acceptance requires complete member counts, expected vector dimensions, bytes, hashes, and a manifest that the later solver reads without reinterpretation.

### Phase 4 — materialize and solve quantization tiers

```bash
smash qtip-configs \
  --manifest "$RUN_ROOT/RUN_MANIFEST.json" \
  --tier qtip2 \
  --layers 0-42 \
  --output "$RUN_ROOT/qtip-configs/qtip2"
```

```bash
smash solve \
  --source-root "$RUN_ROOT/qtip-configs/qtip2" \
  --root "$RUN_ROOT" \
  --tier qtip2 \
  --all-cells \
  --layers 0-42 \
  --device cuda \
  --detach
```

Repeat only for declared tiers. Every solve receipt must bind layer, expert, projection, tier, config, Hessian/capture input, output bytes, hashes, elapsed time, and status.

Current warning: the generalized 2.00-bpw physical path has failed on a whitened-vector shape `(16,)` versus expected `(128,)`. A new run must resolve the producer/consumer geometry contract rather than padding or silently reshaping data.

### Phase 5 — anchors, damage model, and exact byte composition

```bash
smash anchor --run-root "$RUN_ROOT" --detach
```

```bash
smash knapsack-index \
  --run-root "$RUN_ROOT" \
  --basis-sha256 "$MODEL_INDEX_SHA256" \
  --envelope-bytes "$BYTE_ENVELOPE" \
  --output "$RUN_ROOT/knapsack/INDEX.json"
```

```bash
smash knapsack \
  --run-root "$RUN_ROOT" \
  --envelope-bytes "$BYTE_ENVELOPE" \
  --output "$RUN_ROOT/knapsack/ASSIGNMENT.json" \
  --receipt "$RUN_ROOT/knapsack/RECEIPT.json"
```

Required inputs include exact basis identity, complete anchor manifests, damage rows, tier prices/bytes, expert/projection coverage, and source receipts. Refusal here is preferable to a plausible but unbound assignment.

### Phase 6 — repair training

```bash
smash update \
  --runtime-root "$RUNTIME_ROOT" \
  --model-root "$MODEL_ROOT" \
  --aot "$AOT_ROOT" \
  --output "$RUN_ROOT/repair/update" \
  --receipt "$RUN_ROOT/repair/UPDATE_RECEIPT.json" \
  --segments 8 \
  --window 27 \
  --tokens 1024 \
  --layers 43 \
  --learning-rate 1e-4 \
  --resume
```

The repair output must bind the base model, active assignment, source windows, learning parameters, segment checkpoints, consumed codebooks, and output payloads. Restarting must resume from the last committed segment; it must not create two writers to one output.

Current warning: resumability and parity are implemented, but the new `<200 s/segment` physical target is not yet accepted. Do not advertise that speed until a real segment receipt contains start, end, elapsed time, and output hash.

### Phase 7 — export a serveable pack

Public command for an existing canonical plane source:

```bash
smash export \
  --source-root "$SOURCE_PLANES" \
  --serving-model-root "$MODEL_ROOT" \
  --output "$PACK_ROOT" \
  --model-id "$MODEL_ID" \
  --instance-id "$PACK_INSTANCE_ID" \
  --link-mode hardlink
```

For safetensors:

```bash
smash export \
  --source-root "$SOURCE_PLANES" \
  --serving-model-root "$MODEL_ROOT" \
  --output "$PACK_ROOT" \
  --model-id "$MODEL_ID" \
  --instance-id "$PACK_INSTANCE_ID" \
  --link-mode hardlink \
  --safetensors \
  --drop-planes
```

For a bound repair checkpoint, all bound arguments are required together:

```bash
smash export \
  --source-root "$SOURCE_PLANES" \
  --serving-model-root "$MODEL_ROOT" \
  --output "$PACK_ROOT" \
  --model-id "$MODEL_ID" \
  --instance-id "$PACK_INSTANCE_ID" \
  --link-mode hardlink \
  --repair-checkpoint "$REPAIR_CHECKPOINT" \
  --repair-checkpoint-sha256 "$REPAIR_CHECKPOINT_SHA256" \
  --active-overlay "$ACTIVE_OVERLAY" \
  --active-overlay-sha256 "$ACTIVE_OVERLAY_SHA256" \
  --assignment "$ASSIGNMENT" \
  --assignment-sha256 "$ASSIGNMENT_SHA256" \
  --repair-update "$REPAIR_UPDATE"
```

Then validate:

```bash
smash validate-pack "$PACK_ROOT"
```

Optional runtime compatibility gate:

```bash
smash serve-check "$PACK_ROOT" \
  --kernel-cache "$KERNEL_CACHE" \
  --architecture sm_120
```

Do not serve a pack unless validation passes after all metadata and tensor materialization is complete.

### Phase 8 — build/bootstrap the runtime

Expanded campaign form:

```bash
smash bootstrap \
  --context . \
  --image banana_smasher-serve:MODEL_CANDIDATE \
  --receipt "$RUN_ROOT/BOOTSTRAP_RECEIPT.json"
```

The clean-source image receipt should include source commit/tree, container recipe hash, dependency revisions, wheel hashes, image ID, SBOM or equivalent inventory, target architecture, no-cache build-log hash, and direct physical kernel seams.

Current warning: the public-source lane crossed many DeepGEMM, FlashInfer, FP8, and TileLang seams, but the final stock service did not reach accepted HTTP health. Do not substitute the known-golden image acceptance for clean-source closure.

### Phase 9 — serve with vLLM

Generic public product command:

```bash
vllm serve "$PACK_ROOT"
```

The accepted U012 deployment used:

```bash
vllm serve /model \
  --served-model-name DeepSeek-V4-Flash-BQ3 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-memory-bytes 3221225472 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 16 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --no-scheduler-reserve-full-isl \
  --generation-config vllm \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --host 0.0.0.0 \
  --port 8000
```

A new model may require different tokenizer, reasoning, tool parser, context, KV-cache, and memory settings. Those must be derived from its actual architecture and measured memory envelope, not copied blindly.

### Phase 10 — service acceptance

Required API checks:

```bash
curl -fsS "$BASE_URL/health"
curl -fsS "$BASE_URL/v1/models"
```

```bash
curl -fsS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_NAME",
    "messages": [{"role": "user", "content": "Reply with exactly: API READY"}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

Streaming request:

```bash
curl -N -fsS "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MODEL_NAME",
    "messages": [{"role": "user", "content": "Explain binary search briefly."}],
    "temperature": 0,
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Minimum acceptance gate:

1. HTTP 200 `/health`;
2. HTTP 200 `/v1/models` with the expected model ID and context limit;
3. three distinct HTTP 200, nonempty, coherent completions;
4. exact process/container/image identity;
5. exact pack/model-index basis hash;
6. no queue leak after requests;
7. C1/C2/C4/C8/C16 short-chat ladder;
8. streaming TTFT/prefill measurement;
9. final health/models recheck without service mutation; and
10. immutable acceptance and live-handoff receipts.

### Phase 11 — quality evaluation

Expanded campaign form:

```bash
smash evaluate \
  --model-root "$MODEL_ROOT" \
  --candidate "$CANDIDATE_CONTRACT" \
  --reference "$REFERENCE_PACK" \
  --bank "$TEACHER_BANK" \
  --output "$RUN_ROOT/evaluation" \
  --resume-from-layer 0 \
  --verbose-receipts
```

Public minimal validator:

```bash
smash validate "$PACK_ROOT" \
  --bank "$BANK_ID" \
  --check-exposure \
  --receipt "$RUN_ROOT/VALIDATION_RECEIPT.json" \
  --bank-teacher-logits "$TEACHER_LOGITS"
```

Run a smoke window and three-generation coherence gate before full BALANCED64 or HOLDOUT512 evaluation. Full rows must not reuse invalid, wrong-basis, partial, or alternate-instrument measurements.

## 8. Current deployed API

The accepted service handoff records:

- host: Spark-8;
- cluster-private endpoint: `http://192.168.200.9:8000`;
- model: `DeepSeek-V4-Flash-BQ3`;
- basis SHA-256: `58c9d59dfe8fd1e7e833be131043f4b45bfa27064fc19b9fa4fffa6475f2d0fc`;
- maximum advertised context: `8192`;
- serving stack fingerprint: `vllm-0.24.0-a9555d0e`;
- accepted short-chat endpoints: `/health`, `/v1/models`, `/v1/chat/completions`;
- monitoring endpoint used: `/metrics`.

Three acceptance answers included:

```text
V5 API READY
Two plus three equals five.
Jupiter is the largest planet in our solar system.
```

The preserved concurrency ladder measured:

| Concurrency | HTTP 200/nonempty | Completion throughput |
|---:|---:|---:|
| C1 | 1/1 | `10.94 tok/s` |
| C2 | 2/2 | `20.43 tok/s` |
| C4 | 4/4 | `31.35 tok/s` |
| C8 | 8/8 | `54.17 tok/s` |
| C16 | 16/16 | `79.16 tok/s` |

Known limitation: the long-prefill streaming test did not produce a first generated token before its approximately 362-second timeout. Short-chat service acceptance does not prove long-context performance.

The service was intentionally left running by its handoff. The last sealed preservation evidence in the campaign records was 2026-08-02 16:12 PDT. This document does not claim a fresh live probe.

## 9. Current broad-goal status

### Goal 1 — V5 equivalent to V4 with U012

**Status: operationally accepted; final public-source closure incomplete.**

Accepted:

- U012 quality decision;
- export and pack verification;
- Spark-8 OpenAI-compatible API;
- coherent short chat;
- C1–C16 ladder;
- basis/process preservation.

Incomplete:

- clean public-source stock API acceptance;
- branch merge into public `main`;
- long-prefill acceptance; and
- a fully public, stranger-reproducible clean-box build.

### Goal 2 — accelerate QTIP construction and anchors

**Status: partial.**

Accepted/proven:

- exact QTIP2/QTIP3 campaign solving;
- 43-layer QTIP2 aggregate;
- best QTIP2 rate `1.548 s/unit`;
- typical accepted QTIP2 rate approximately `1.75–1.91 s/unit`;
- QTIP3 approximately `1.90979 s/unit`;
- hash-bound config materialization.

Incomplete:

- generic arbitrary 0.25-bpw ring physical closure;
- current 2.00-bpw vector-shape contract;
- complete anchor manifests and damage rows; and
- accepted anchor-to-knapsack end-to-end timing.

### Goal 3 — accelerate repair training

**Status: implementation exists; target speed unproven.**

Accepted/proven:

- resumable update checkpoints;
- multi-window composition;
- failure receipts and resume location;
- reference/accelerated parity tests;
- legacy U012 physical update.

Incomplete:

- first accepted segment from the new persistent acceleration harness; and
- proof of the `<200 s/segment` target.

### Goal 4 — competitor KLD and top-1 matrix

**Status: mostly complete.**

Sealed:

- IQ3 BALANCED64 KLD/top-1;
- IQ4 BALANCED64 KLD/top-1;
- DwarfStar BALANCED64 KLD/top-1; and
- aggregate comparison table.

Incomplete:

- canonical IQ2 own-0731-base row;
- our BQ3-0731 pre-repair row;
- our post-repair row;
- paused IQ3/IQ4 HOLDOUT512 completions; and
- unified behavioral/throughput rows for every exact artifact.

## 10. Receipt-backed timing guide

There is no trustworthy single stopwatch from brand-new model arrival through accepted service. Use stage-specific receipts:

| Stage | Observed timing |
|---|---:|
| Legacy/production U012 repair update | `519.41 s` |
| Update forward portion | `202.72 s` |
| Update backward portion | `316.13 s` |
| Best exact QTIP2 unit | `1.548 s/unit` |
| Typical QTIP2 unit | approximately `1.75–1.91 s/unit` |
| 512 QTIP2 units for one layer | approximately 15.9–16.1 minutes |
| QTIP3 | `1.90979 s/unit`, approximately 16.3 minutes per 512 units |
| Projected all-QTIP2 workload | `0.543 spark-days` |
| Three generic QTIP ring configs | `613 s` |
| U012 export task | `800.65 s` |
| Export integrity peer review | `33.87 s` |
| Accepted API launch to acceptance | `500.56 s` |
| U012 go-decision to accepted API | approximately 6h58m campaign wall |
| DwarfStar BALANCED64 KLD task | approximately 78 minutes |
| IQ3 BALANCED64 campaign wall | approximately 2h24m |
| IQ4 BALANCED64 campaign wall | approximately 6h12m |
| IQ3/IQ4 top-1 recomputation | approximately 20 minutes |
| Dwarf top-1 recomputation | approximately 10 minutes |
| Final top-1 aggregation | approximately 5 minutes |

Campaign wall time includes retries, staging, integrity gates, runtime seam repair, and coordination. It is not pure compute time.

## 11. Required final comparison report

The ultimate publication should contain one artifact-bound row per candidate and never borrow cells from a different artifact or evaluation rail.

### 11.1 Required candidates

At minimum:

1. exact FP8/source teacher;
2. shipped V4 reference;
3. U012/V5 deployment candidate;
4. our BQ3-0731 backpack pre-repair;
5. our BQ3-0731 backpack post-repair;
6. Unsloth IQ2 0731;
7. Unsloth IQ3 0731;
8. Unsloth IQ4 0731;
9. DwarfStar Q2 0731; and
10. any external NVFP4/OpenRouter reference, clearly marked as non-paired context.

### 11.2 Required columns

#### Identity and provenance

- display label;
- model ID and source revision;
- model-index SHA-256;
- pack manifest SHA-256;
- assignment SHA-256;
- repair checkpoint/overlay SHA-256;
- evaluator and runtime commits;
- teacher basis;
- corpus/window/class-map identities;
- receipt paths and hashes;
- evidence status.

#### Footprint

- exact decimal bytes and GB;
- GiB where useful;
- total byte-derived bpw;
- tensor-only bpw, labeled separately;
- expert payload bytes;
- base/shared weight bytes;
- runtime residency and peak GPU memory.

#### Distributional quality

- BALANCED64 global KLD;
- HOLDOUT512 global KLD;
- top-1 agreement;
- candidate/teacher mismatch;
- JS divergence where available;
- six-class means: agentic, chat, code, multilingual, prose, reasoning;
- p90/p95/p99/max and `% > 0.5` where available;
- paired per-window improvement count;
- uncertainty/effect-size floor.

#### Behavioral quality

- ToolEvalBench mean, spread, and N;
- HumanEval pass@1;
- HumanEval+ pass@1;
- MMLU-500 accuracy and count;
- exact prompt/evaluator/version/cap settings; and
- null/error handling policy.

#### Serving performance

- model load time;
- C1/C2/C4/C8/C16 throughput;
- per-request latency distributions;
- prompt prefill throughput;
- TTFT;
- long-context success/failure;
- maximum context;
- peak memory;
- scheduler queue drain;
- tool/reasoning parser behavior; and
- service/runtime/image identity.

#### Production cost

- capture/Hessian duration;
- solve seconds per unit and total spark-days;
- anchor/composition time;
- repair seconds per segment;
- export/verify time;
- image build time;
- model-load/acceptance time;
- number and class of hosts used; and
- resumability/retry overhead.

### 11.3 Status vocabulary

Use only explicit states such as:

- `IMPLEMENTED`
- `UNIT_TESTED`
- `PHYSICALLY_PROVEN`
- `MEASURED_GATE`
- `MEASURED_64`
- `MEASURED_512`
- `DEPLOYED`
- `ACCEPTED`
- `PREDICTED`
- `PAUSED_RESUMABLE`
- `BLOCKED_MISSING_PRODUCER`
- `INVALID_WRONG_BASIS`
- `PENDING`

A predicted or partial row is never formatted as a canonical winner.

### 11.4 Comparison rules

1. Never subtract values from different teacher rails as though they were paired.
2. Never borrow a size, KLD, or behavioral cell from a different artifact revision.
3. Use exact bytes; state whether GB is decimal and whether bpw includes non-tensor files.
4. Keep BALANCED64 gates separate from HOLDOUT512 canonical rows.
5. Keep quality rows separate from serving-speed rows unless the served artifact hash matches exactly.
6. Report invalid and failed attempts when they explain missing cells.
7. Preserve null/pending cells rather than filling them with a nearby value.
8. Bind every headline to a receipt path and digest.
9. Separate external context bars from same-model paired comparisons.
10. State exposure/leakage ancestry honestly.

## 12. Principal current gaps

1. **Public-source serving:** the accepted Spark-8 service is known-golden; the independently rebuildable public stock image remains incomplete.
2. **GitHub integration:** the public-source branch is not merged into `main`.
3. **Generic QTIP geometry:** the current arbitrary-ring 2.00-bpw solve has an input-vector shape-contract failure.
4. **Anchor/knapsack producer closure:** complete basis, anchor manifests, and damage rows are not yet available in one accepted chain.
5. **Repair speed:** the accelerated persistent harness has not sealed a real `<200 s` segment.
6. **IQ2 provenance:** no canonical exact-0731 IQ2 pack has been selected and sealed.
7. **Our exact-0731 candidate:** the pre-repair pack has not completed STEP5 composition/export/evaluation.
8. **Post-repair exact-0731 row:** blocked on the pre-repair artifact.
9. **Holdout matrix:** comparator HOLDOUT512 walks were paused/preempted before full closure.
10. **Long-context serving:** the accepted API's long-prefill test timed out.
11. **Turnkey public reproduction:** some teacher banks, large artifacts, and campaign receipts remain internal rather than mirrored publicly.
12. **Single automated final report:** the schema is defined here, but the complete artifact-bound matrix has not yet been generated in one machine-readable and rendered bundle.

## 13. Fail-closed operating rules

- Never mutate a sealed source tree.
- Never hand-edit a pack manifest or model index to force a pass.
- Never invent a missing model variant, teacher, anchor, damage row, or assignment.
- Never run two writers against the same run root.
- Never restart detached healthy work merely because an observer timed out.
- Never reuse wrong-basis or alternate-harness rows.
- Never call an HTTP 200 alone “accepted”; require coherent content and identity gates.
- Never call a branch “deployed” because it builds locally.
- Never call a speed target achieved without physical elapsed-time receipts.
- Always write the pack manifest last and validate after final metadata materialization.
- Always preserve source commit, wheel/image identity, command line, inputs, outputs, hashes, and timing.

## 14. Evidence map

Public project files:

- [`README.md`](README.md) — public release path
- [`PACK_FORMAT.md`](PACK_FORMAT.md) — frozen `bs-pack v1` contract
- [`PIPELINE.md`](PIPELINE.md) — V4 wire/export parity pipeline
- [`BANANA_PACK_SPEC.md`](BANANA_PACK_SPEC.md) — additional pack specification
- [`NIGHTLY_SEALED_RESULTS.md`](NIGHTLY_SEALED_RESULTS.md) — public sealed test evidence

Canonical campaign records on the production Mac:

```text
/Users/macmini/clawd/ds4-flash-kldmatrix/DRIVER_GOALS.md
/Users/macmini/clawd/ds4-flash-kldmatrix/RESULTS.md
/Users/macmini/clawd/ds4-flash-kldmatrix/docs/RESULTS_LADDER.md
/Users/macmini/.hermes/kanban/boards/glm52-humming-w3/kanban.db
```

Accepted API benchmark bundle:

```text
/Users/macmini/.hermes/kanban/boards/glm52-humming-w3/attachments/t_adab1a37/SPARK8_BENCHMARK_SUMMARY.json
/Users/macmini/.hermes/kanban/boards/glm52-humming-w3/attachments/t_adab1a37/SPARK8_BENCHMARK_REPORT.md
```

Important campaign task IDs:

| Task | Purpose/status |
|---|---|
| `t_a5ee9b12` | accepted independent U012/V5 API on Spark-8 |
| `t_adab1a37` | preserved accepted-service benchmark and evidence bundle |
| `t_becf6641` | final-form stock V5 API; blocked |
| `t_63769bff` | clean public-source lineage; blocked/retired |
| `t_627545c0` | generic QTIP ring physical unit |
| `t_72369ab4` | broad QTIP campaign owner |
| `t_119da9b5` | repair-training persistent-segment work |
| `t_60d9568b` | broad repair-training owner |
| `t_32b46c5e` | BAL64 top-1 comparison table |
| `t_93687ad4` | sealed DwarfStar BAL64 KLD |
| `t_d72b1f73` | IQ2 provenance blocker |

## 15. Definition of project success

The project is complete only when a stranger can:

1. clone a public, pinned revision;
2. obtain documented immutable source inputs;
3. build calibration/teacher evidence;
4. solve all declared tiers;
5. produce anchors and an exact byte-envelope assignment;
6. run resumable repair;
7. export and validate a complete `bs-pack`;
8. build the serving image from public source;
9. launch stock-semantics vLLM without hidden patches;
10. pass health, coherent generation, load, prefill, and preservation gates;
11. reproduce the same-basis quality and behavioral evaluations; and
12. generate the final comparison report with no borrowed, mixed-basis, or unreceipted cells.

Until then, Banana Smasher is a strong working campaign system with an accepted U012 deployment and substantial public components—not yet a fully turnkey public product.
