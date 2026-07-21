# Reproducing PTQ-OPD from scratch

This repository contains source, schemas, tests, and exact experiment contracts. It intentionally contains **no model weights, checkpoints, teacher-score tensors, generated solutions, or private infrastructure paths**.

The BQ3 construction is a prerequisite and is reproduced in [`../glm52-ds4-w23-planes-quant`](../glm52-ds4-w23-planes-quant/). PTQ-OPD starts after that folder's exact `combo-V4-step32` checkpoint and 101,360,840,912-byte wire have been built.

## 1. Inputs you must obtain

Download or build these inputs yourself:

1. The official DeepSeek-V4-Flash source checkpoint used as teacher.
2. The exact tokenizer and prompt renderer for that checkpoint.
3. BQ3 step0 from the prior reproducer.
4. A prompt manifest with a frozen train/eval split.
5. Fixed static-anchor teacher rows.

For a shippable generalization run, the PTQ-OPD training manifest must contain diverse **non-benchmark** prompts: real-repository code, math/logic, agentic/tool traces, long-form chat/analysis, and multilingual reasoning. HumanEval/EvalPlus stays evaluation-only.

## 2. Environment

The campaign ran on NVIDIA GB10 / sm_121 / aarch64 with PyTorch `2.11.0+cu130`, NumPy `2.3.5`, safetensors `0.8.0`, Transformers `5.13.0`, and a vLLM-Moet-derived BQ3 runtime. Install the NVIDIA PyTorch build matching your CUDA stack first; `requirements.txt` pins the remaining production Python packages. The pure bank/objective tests also run on CPU.

```bash
cd glm52-ds4-bq3-ptq-opd
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install your platform's PyTorch wheel first, then:
python -m pip install -r requirements.txt
PYTHONPATH=reference python -m unittest discover -s reference/tests -p 'test_*.py' -v
```

The production adapter imports the same BQ3 layer construction used by the prior quant reproducer. Put model inputs outside the repository and provide them through environment variables; never vendor model bytes into this tree.

## 3. Generate student trajectories

For every prompt, persist:

- prompt ID and SHA-256;
- split and tokenizer SHA-256;
- student checkpoint SHA-256;
- serve fingerprint;
- exact generation settings and their canonical SHA-256;
- full token IDs;
- `score_start` for the completion suffix;
- finish reason and visible/null state.

Frozen mechanism recipe:

```text
max completion tokens = 4096
n                     = 1
greedy temperature    = 0.0
top_p                 = 1.0
```

Temperature rows may be added to a sealed exploration bank, but their role and seed namespace must be explicit. Do not truncate a sequence to make it fit: exclude it and report the exclusion.

## 4. Teacher-score the exact student sequence

Teacher-force the FP model over each student sequence and write one tensor payload per sample:

```python
{
    "teacher_topk_ids":        int64[n, k],
    "teacher_topk_logprobs":   float[n, k],
    "teacher_target_logprobs": float[n],
    "teacher_tail_logmass":    float[n],
}
```

Requirements:

- top-k log probabilities are absolute full-softmax values;
- `exp(topk).sum() + exp(tail_logmass) == 1` within `1e-6`;
- target log probability equals the gathered top-k value when present;
- otherwise it must not exceed the kth log probability;
- payload files are nonsymlinked, hash-bound, and referenced by a traversal-free relative path.

Create a JSONL bank following the fields enforced by `reference/ptq_opd.py`. Validate before any GPU model assembly:

```python
from pathlib import Path
from ptq_opd import load_bank

rows, receipt = load_bank(
    Path("repro-data/bank/bank.jsonl"),
    mode="rolling",
    min_rows=16,
    min_prompts=16,
    min_topk=32,
)
print(receipt)
```

## 5. Configure the full trainer

The production reference is `reference/train_ptq_opd.py`; its BQ3 model adapter and source dependencies are under `reference/adapter/`. Supply these external inputs:

```bash
export PTQ_OPD_ROOT="$(pwd)/repro-data"
export PTQ_OPD_BANK="$PTQ_OPD_ROOT/bank/bank.jsonl"
export PTQ_OPD_BANK_MODE=rolling
export PTQ_OPD_MIN_BANK_ROWS=16
export PTQ_OPD_MIN_PROMPTS=16
export PTQ_OPD_TARGET=4
export PTQ_OPD_START_UPDATE=0
export PTQ_OPD_OBJECTIVE=jsd
export PTQ_OPD_BETA=0.5
export PTQ_OPD_ANCHOR_WEIGHT=0.5
export PTQ_OPD_LOGIT_CHUNK=512
export PTQ_OPD_ROLLOUT_SEGMENT_ROWS=2
export PTQ_OPD_OWN_MICROBATCH=2
export PTQ_OPD_EXPECT_CORPUS_TRACK=benchmark_distribution_mechanism
export PTQ_OPD_OUTDIR="$PTQ_OPD_ROOT/runs/ptq-opd-step4"

# Paths consumed by the BQ3 adapter; build these via the prior reproducer.
export BQ3_ASSET_ROOT="$PTQ_OPD_ROOT/bq3-assets"
export BQ3_SOURCE_CHECKPOINT="$PTQ_OPD_ROOT/deepseek-v4-flash"
export BQ3_PACK_DIR="$PTQ_OPD_ROOT/bq3-pack"
export BQ3_STATIC_TEACHER_DIR="$PTQ_OPD_ROOT/static-anchor/teacher-rows"
export BQ3_STATIC_CORPUS="$PTQ_OPD_ROOT/static-anchor/windows.json"
export BQ3_CALIB_SELECTION="$PTQ_OPD_ROOT/static-anchor/selection.json"
export BQ3_BASE_MANIFEST="$PTQ_OPD_ROOT/bq3-base-manifest.json"
export BR_BASE_MANIFEST="$BQ3_BASE_MANIFEST"
export BR_MANIFEST="$PTQ_OPD_ROOT/bq3-manifest.json"
export BR_DELTA_DIR="$PTQ_OPD_ROOT/bq3-delta-pack"
export BR_VQ3B_DIR="$PTQ_OPD_ROOT/bq3-vq3b"
export BR_WIRE_DIR="$PTQ_OPD_ROOT/bq3-wire"
export BR_TEACH="$BQ3_STATIC_TEACHER_DIR"
export BR_CORPUS="$BQ3_STATIC_CORPUS"
export BR_TRAIN=0,1,2,3,4,5,6,7,8,9,10,11,50,51,52,53,54,55,56,57,58,59,60,61,100,101,102,103,104,105,106,107,108,109,110,111,150,151,152,153,154,155,156,157,158,159,160,161
export BR_PROBE=12,13,14,15,62,63,64,65,112,113,114,115,162,163,164,165
export TAILFIX_START_CKPT="$PTQ_OPD_ROOT/bq3-step0.pt"
```

`BR_TRAIN` and `BR_PROBE` select static-corpus window indices; use the frozen schedule from `reference/plans/static_anchor_control.json` for campaign reproduction. `BR_DELTA_DIR` must contain `DELTA_PACK.COMPLETE`; `BR_VQ3B_DIR` must contain all 43 `vq3u_layer_NNN.pt` planes; and `BR_WIRE_DIR` must contain `PACK_COMPLETE`, `PACK_MANIFEST.json`, the wire receipt, and every layer's NumPy planes. The source-only adapter dependencies are vendored under `reference/adapter/vendor/`; no private source tree is required.

The sealed step4 artifact used beta-0.5 JSD. `PTQ_OPD_OBJECTIVE=reverse_kl` is supported for a new PTQ-OPD experiment, but it is not an exact reproduction of the sealed checkpoint and must produce new checkpoint, static-gate, and behavioral-gate identities.

The public source pins the campaign's exact starting checkpoint identity. If you intentionally port PTQ-OPD to another BQ artifact, change the identity constant and regenerate a source seal; do not bypass it at runtime.

Verify the committed production-source seal before launch:

```bash
PYTHONPATH=reference python reference/build_source_seal.py --check
```

`--write` is only for an intentional source change followed by review and recommit.

## 6. Run the dose

Before launch:

- verify the GPU is otherwise empty;
- verify the BQ3 checkpoint and wire hashes;
- verify the bank receipt;
- run the no-optimizer full-model gradcheck;
- confirm exactly 1,855,147 trainable parameters;
- capture a static baseline on the held-out anchor.

Then run:

```bash
PYTHONPATH=reference python reference/train_ptq_opd.py
```

Expected durable products:

```text
LATEST.pt
CHECKPOINT_STEP1.pt ... CHECKPOINT_STEP4.pt
CANDIDATE_STEP4.pt
STATIC_BASELINE.json
STATIC_DOSE_STEP4.json
STATIC_GATE_STEP4.json
DOSE_LEDGER.jsonl
events.jsonl
STATUS.json
```

A static pass remains `promotable=false` until behavioral gates pass.

### Reproduce the step4 → transfer-step8 continuation

Build a separate 16-row `tailfix_general_shippable` bank, then bind the exact gate-passed step4 parent and continue Adam without reset:

```bash
export PTQ_OPD_BANK="$PTQ_OPD_ROOT/bank/transfer.jsonl"
export PTQ_OPD_EXPECT_CORPUS_TRACK=tailfix_general_shippable
export PTQ_OPD_START_UPDATE=4
export PTQ_OPD_TARGET=8
export PTQ_OPD_START_CHECKPOINT="$PTQ_OPD_ROOT/parents/step4/LATEST.pt"
export PTQ_OPD_START_CHECKPOINT_SHA256=<sha256-of-gate-bound-step4-LATEST>
export PTQ_OPD_START_CANDIDATE_SHA256=<sha256-of-CANDIDATE_STEP4.pt>
export PTQ_OPD_START_GATE="$PTQ_OPD_ROOT/parents/step4/STATIC_GATE_STEP4.json"
export PTQ_OPD_START_GATE_CANONICAL_SHA256=<canonical-sha256-of-step4-gate>
export PTQ_OPD_START_BANK_SHA256=<step4-parent-bank-manifest-sha256>
export PTQ_OPD_OUTDIR="$PTQ_OPD_ROOT/runs/ptq-opd-transfer-step8"
PYTHONPATH=reference python reference/train_ptq_opd.py
```

The trainer accepts only the exact continuation pairs `4→8` and `8→16`, verifies the parent gate/candidate/checkpoint/Adam lineage, and consumes each 16-row transfer shard exactly once across updates 5–8.

## 7. Evaluate

Run all compared checkpoints under one serving build and one exact request shape.

### Frozen correctness

- greedy;
- max completion tokens 4096;
- EvalPlus v0.1.10 / pinned commit `26d6d00`;
- network disabled during scoring;
- retain true nulls once as empty/fail; never retry for a favorable sample.

Report HumanEval base and plus counts, plus separate trained and held-out splits.

### Static KLD

Use the exact fixed windows and FP teacher rows. Report global and per-class mean/p90/p95/p99. The campaign's full table used 512 windows / 524,288 positions / teacher top-8192 and an independent NumPy-versus-Torch reload check.

### Reasoning-length panel

Persist hidden reasoning token count, visible token count, total completion tokens, finish reason, and null status. Use at least 12 unique prompts and repeat designated sentinels. Report median percent change, sign count, and a conservative replicate-derived variance floor.

### Uncapped diagnostic

Use a separate label, `DIAGNOSTIC_UNCAPPED`, with max completion tokens 16,384. Never merge this instrument with frozen correctness.

## 8. Verify receipts and privacy

Before publication:

```bash
python tools/publication_audit.py
python ../glm52-ds4-w23-planes-quant/tools/scrub_audit.py .
python ../glm52-ds4-w23-planes-quant/tools/normalize_public_names.py --check .
```

Also search for real user names, home directories, host aliases, IPs, task IDs, process IDs, tokens, provider account identifiers, and local model paths. Public docs should contain only portable relative paths and cryptographic identities.
