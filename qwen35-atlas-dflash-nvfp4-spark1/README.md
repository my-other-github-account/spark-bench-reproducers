# Qwen3.5 27B NVFP4 Atlas DFlash on Spark-1 — clean-room repro bundle

This folder is the public, from-scratch reproduction guide for the Spark-1 Rust/Atlas NVFP4 DFlash workstream. It preserves the exact legal full-72-prompt result that passed the fair NVFP4 speedup gate, plus the patch, prompts, scripts, and verification artifacts needed for a different person on a fresh DGX Spark system to rebuild and rerun it.

**Current status:** legal fair speedup artifact shipped; Big D's newer 30 tok/s target is active but not yet achieved in this bundle.

## Results — measured full72 NVFP4 contract

- Hardware: DGX Spark GB10 Blackwell, **sm_121**, aarch64, 128 GiB unified memory
- Engine: native Rust Atlas `spark serve`, OpenAI-compatible API
- Target model: Qwen3.5 27B NVFP4 checkpoint (`spark6-Qwen3.5-27B-NVFP4` in the original run)
- Drafter: matched Qwen3.5 DFlash drafter (`qwen35-dflash` in the original run)
- Prompt battery: `prompts/atlas_diverse_72.jsonl`, 72 prompts
- Decode contract: `max_tokens=160`, `temperature=0.0`, `concurrency=1`, full vocab, NVFP4 KV, `max_seq_len=512`, `max_num_seqs=1`, `max_batch_size=1`
- Token accounting: Atlas response `usage.completion_tokens / aggregate wall-clock seconds`

Measured artifact: `results/nvfp4_final_summary_legal_21tps.json`

- AR output tok/s: `13.144712755655855`
- DFlash output tok/s: `21.28429140169363`
- Speedup: `1.619228338978766x`
- Prompt count: `72`
- Model format: `nvfp4`
- Fair baseline: `True`
- Quality spotcheck: `True`

## Important 30 tok/s retarget

The result above is **not** the final aspirational goal anymore. The active follow-on target is recorded in `docs/USER_DIRECTIVE_NVFP4_DFLASH_30TPS_20260525T055149Z.md`:

- retain the legal 21.284 tok/s full72 result as baseline only
- produce a fresh full72 NVFP4 DFlash artifact with `dflash_output_tps >= 30.0`
- keep the same/fair benchmark contract and quality spotcheck

This repo folder is still useful: it is the exact clean baseline a fresh person can rebuild before pushing toward the 30 tok/s gate.

## Repository contents

- `README.md` — this guide.
- `patches/0001-atlas-nvfp4-dflash-all-quant-pass-state.patch` — exact Atlas source patch from base `87b7bb3279c33b130cf98ced6f13d79d5cde013c` to local pass-state commit `f118b7265e499ac19914c522682b80524afe6b2f`.
- `prompts/atlas_diverse_72.jsonl` — fixed 72-prompt battery.
- `results/nvfp4_final_summary_legal_21tps.json` — canonical legal final summary.
- `results/ar_benchmark_q35_nvfp4_full72_c1.json` — raw AR full72 receipt.
- `results/dflash_benchmark_q35_nvfp4_full72_c1_gamma3_all.json` — raw DFlash full72 receipt.
- `results/speedup_instrumentation_summary_20260525T030650Z.json` — before/after fine-grained timing summary for the structural all-quant DFlash win.
- `results/ar_vs_dflash_output_text_audit_20260525.json` — text-level AR-vs-DFlash audit; do not claim exact output identity.
- `docs/exact-working-state-20260525-nvfp4-dflash.md` — detailed provenance ledger copied from the original driver workspace.
- `scripts/prepare_atlas.sh` — clone Atlas, reset to base, apply patch, build `target/release/spark`.
- `scripts/launch_ar.sh` / `scripts/launch_dflash.sh` — exact server launch shape.
- `scripts/bench_full72.sh` — rerun the full72 benchmark against a live server.
- `scripts/verify_saved_artifacts.py` — local verification of the saved bundle.

## From-scratch reproduction on a fresh Spark

Assumptions:

- Ubuntu 24.04 aarch64 on DGX Spark / GB10.
- NVIDIA driver/CUDA runtime already working.
- Rust/Cargo toolchain installed and able to build Atlas.
- Model weights are available locally under `~/models` or you have Hugging Face access to download equivalent mirrored repos.
- Run commands from this recipe directory after cloning `spark-bench-reproducers`.

### 1. Clone the reproducer repo

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/qwen35-atlas-dflash-nvfp4-spark1
```

### 2. Obtain model assets

```bash
bash scripts/download_models.sh
```

The original measured paths were:

```text
/home/banana_bae/models/spark6-Qwen3.5-27B-NVFP4
/home/banana_bae/atlas-dflash-spark1/drafters/qwen35-dflash
```

For a fresh machine, either mirror those paths or set:

```bash
export MODEL_PATH=/path/to/spark6-Qwen3.5-27B-NVFP4
export DRAFT_MODEL=/path/to/qwen35-dflash
```

### 3. Rebuild Atlas from base + patch

```bash
bash scripts/prepare_atlas.sh
```

Expected result: a patched Atlas checkout under `work/atlas` and binary at `work/atlas/target/release/spark`.

### 4. Launch and benchmark AR

Terminal A:

```bash
export MODEL_PATH=${MODEL_PATH:-$HOME/models/spark6-Qwen3.5-27B-NVFP4}
PORT=9155 ATLAS_DIR=$PWD/work/atlas MODEL_PATH=$MODEL_PATH bash scripts/launch_ar.sh 2>&1 | tee ar_server.log
```

Terminal B:

```bash
PORT=9155 BASE_URL=http://127.0.0.1:9155 bash scripts/wait_for_server.sh
MODE=ar BASE_URL=http://127.0.0.1:9155 OUT=results/live_ar_benchmark_q35_nvfp4_full72_c1.json bash scripts/bench_full72.sh
```

Stop the AR server after the benchmark.

### 5. Launch and benchmark DFlash

Terminal A:

```bash
export MODEL_PATH=${MODEL_PATH:-$HOME/models/spark6-Qwen3.5-27B-NVFP4}
export DRAFT_MODEL=${DRAFT_MODEL:-$PWD/work/qwen35-dflash}
PORT=9156 ATLAS_DIR=$PWD/work/atlas MODEL_PATH=$MODEL_PATH DRAFT_MODEL=$DRAFT_MODEL bash scripts/launch_dflash.sh 2>&1 | tee dflash_server.log
```

Terminal B:

```bash
PORT=9156 BASE_URL=http://127.0.0.1:9156 bash scripts/wait_for_server.sh
MODE=dflash BASE_URL=http://127.0.0.1:9156 OUT=results/live_dflash_benchmark_q35_nvfp4_full72_c1_gamma3_all.json bash scripts/bench_full72.sh
```

### 6. Compare with the shipped legal artifact

```bash
python3 scripts/verify_saved_artifacts.py
python3 - <<'PY'
import json
ar=json.load(open('results/live_ar_benchmark_q35_nvfp4_full72_c1.json'))
df=json.load(open('results/live_dflash_benchmark_q35_nvfp4_full72_c1_gamma3_all.json'))
print('Inspect live JSON fields and compute output tok/s using the same usage.completion_tokens / wall time contract.')
print('Saved canonical summary is in results/nvfp4_final_summary_legal_21tps.json')
PY
```

A faithful rerun should land near the saved full72 numbers, subject to normal thermal/host variance:

- AR ~13.14 output tok/s
- DFlash ~21.28 output tok/s
- speedup >=1.50x

## What changed in Atlas

The important structural win was not a small knob sweep. Atlas DFlash's proposer/forward-block hot path was still paying BF16-style work while the fair target was NVFP4. The patch adds DFlash quantization modes and quantized copies for DFlash forward-block projections:

```text
ATLAS_DFLASH_QUANTIZATION=bf16|mlp|all
```

The retained legal run used:

```text
ATLAS_DFLASH_QUANTIZATION=all
ATLAS_DFLASH_INLINE_REPROPOSE=1
ATLAS_DFLASH_FORCE_GENERIC_VERIFY=1
ATLAS_SSM_ENABLE_F32_DECODE=1
ATLAS_FORCE_FULL_VOCAB=1
```

Key source areas touched include `dflash_head`, `forward_block_layer`, `from_weights`, SSM decode/verify paths, scheduler verify steps, and Qwen3.5 NVFP4 weight loading.

## Correctness / quality caveat

This folder does **not** prove exact AR-vs-DFlash output identity. The saved audit says:

- exact text matches: 6 / 72
- same completion-token count: 57 / 72
- mean text similarity ratio: 0.5862

The legal PASS criterion was fair throughput plus sanity/quality spotcheck. If you need exact speculative-decoding equivalence, add a new hard gate that compares token IDs/text against AR and require 72/72 exact matches.

## Secret hygiene

No credentials, tokens, private connection strings, or host passwords are required or included. Any machine-specific paths in provenance docs are local filesystem paths from the original Spark-1 run.

By [@banana_baeee](https://x.com/banana_baeee)
