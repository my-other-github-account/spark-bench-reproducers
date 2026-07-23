# End-to-end reproduction

This repository publishes code, manifests, result ledgers, hashes, and environment snapshots. It intentionally does not publish model weights, teacher tensors, or private infrastructure.

For the fastest campaign continuation, use [`RESUME.md`](RESUME.md). For experiment design and
failure history, read [`LEARNINGS.md`](LEARNINGS.md). This file remains the clean-room reproduction
sequence.

## 1. Inputs

Provide these environment variables:

```bash
export MODEL_ROOT=/path/to/base-model
export MISSION_ROOT=/path/to/campaign-work
export CORPUS_JSON=/path/to/windows_ds4_eval.json
```

Verify the corpus before work:

```bash
python3 - <<'PY'
import hashlib, os
p = os.environ['CORPUS_JSON']
print(hashlib.md5(open(p, 'rb').read()).hexdigest())
PY
# expected: 1701920b4ba96dea0b18fe9df0151876
```

Use the exact package snapshot matching the role under `environments/`. Model weights are independently acquired and verified by their upstream manifests.

For HumanEval(+), additionally bind EvalPlus commit `26d6d00`, HumanEvalPlus dataset SHA-256
`42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`, N=1 greedy,
temperature 0, top-p 1, seed 0, and a true 4,096-token completion cap. KV dtype, context,
server parallelism, client concurrency, and homogeneous-from-row policy are part of the receipt;
temperature zero does not make llama.cpp batching invariant.

## 2. Build uniform anchors

The legacy d4 scripts and playbook remain at the repository root. For each K, build all 43 expert planes, persist codebooks together with codes, and produce an integrity sidecar. Never resume codes against regenerated codebooks.

Required gates:

- 43/43 layers present;
- plane hashes read back;
- corpus MD5 matches;
- evaluator records 512 windows and 524,288 positions;
- row label is `MEASURED`, never `PRED`.

The sealed reference rows are listed in `ladder/ANCHOR_TABLE.md`; raw JSON receipts are in `ladder/anchors/`.

## 3. Solve the two budgets

Run the desired solver from `ladder/solvers/` with portable input paths. The solver must consume measured anchor/bin rows and write both a solve summary and a manifest.

```bash
python3 ladder/solvers/solve_dual_vq3_two_bins.py --help
python3 ladder/solvers/solve_quad_vq_two_bins.py --help
python3 ladder/solvers/solve_penta_vq_two_bins.py --help
python3 ladder/solvers/solve_hexa_vq_two_bins.py --help
```

Verify byte caps and manifest hashes before quantization. Solver output is `PRED` until the exact manifest is built and railed.

## 4. Rail KLD

The canonical evaluator contract is in `eval/README.md`. Teacher tensors are built with `eval/teacher-build/`; set `DS4_MODEL_REMOTE` only when using the optional remote fetch path. Append one JSON object per window, flush after every row, and resume by window ID.

Validation:

```bash
python3 -m py_compile eval/*.py eval/teacher-build/*.py
python3 eval/validate_published_artifacts.py
```

## 5. Repair

Run codebook arms with `repair/code/binrepair_e2e.py` and the shell configurations in `repair/configs/`. The binding probes are `[4,84,160,236,304,373,442,511]`; training windows must be disjoint. Step 0 must reproduce the same baseline before training.

For RMSNorm-gamma and output-scale mechanisms use `repair/altrepair/code/`. Generate the publication tables directly from append-only ledgers:

```bash
python3 repair/summarize_probes.py
```

Only pooled results greater than the `2.6%` effect floor may seal. `repair/SEALED_REPAIR_REPLICATION.json`, the sidecar rows, and `repair/SEAL_NOTES.md` are the formal result.

## 6. Export and serve

Follow `serving/README.md` and `serving/SERVED_BASELINES.md`. Export parity (B equals C) is necessary but not sufficient; the checkpoint-to-wire A/B tolerance must pass before a served quality rail is claimable.

The portable carry-through tools are under `tooling/`. `export_arm4.py` materializes
`state['L{n}']['cb13/cb2']` into a complete hash-bound VQ3U plane set. For the sealed fast evaluator,
set both `VQ3U_OVERRIDE_DIR`/`VQ3U_OVERRIDE_RECEIPT` and `TWOBIN_DELTA_DIR`; the first smoke that
omitted the delta-pack binding was invalid. See `RESUME.md` for the complete environment block.

## 7. Release gates

```bash
python3 tools/normalize_public_names.py
python3 tools/scrub_audit.py
python3 repair/summarize_probes.py
python3 -m py_compile tools/*.py tooling/*.py repair/*.py eval/*.py eval/teacher-build/*.py ladder/solvers/*.py serving/*.py
bash -n tooling/*.sh repair/code/*.sh repair/configs/*.sh repair/altrepair/code/*.sh serving/*.sh
python3 tooling/fix_runpilot_env.py --check repair/code/run_pilot.sh
git diff --check
```

Before every EvalPlus rescore, delete any prior `samples.eval_results.json`, prepare the exact nested
response content, sanitize inside the pinned network-isolated container, and assert that the prepared
row count is nonzero. A stale cache produced three false `0.000` summaries during this campaign.

The campaign-specific identity-twin, wash-out, dose-4 screening, serve-receipt, and pre-registered
decision-matrix templates are in [`PTQ_OPD_CAMPAIGN.md`](PTQ_OPD_CAMPAIGN.md#reproduction-appendix).

The scrub gate rejects credentials, private addresses, local host aliases, machine-specific home paths, and connection strings. Read `SCRUB_POLICY.md` before adding artifacts.
