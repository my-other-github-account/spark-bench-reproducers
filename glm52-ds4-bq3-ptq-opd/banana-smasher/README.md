# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

## Three-command release path

```bash
smash export --source-root /path/to/quantizer-output --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash validate-pack /model
vllm serve /model
```

The first command builds `/model` and writes `BANANA_PACK_MANIFEST.json` last after self-verification. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

BananaSmasher is only the proper name of the first sealed model instance. Reusable package, schema, CLI, and documentation names remain `banana-smasher`, `bs-pack`, and `smash`.

## Exact accelerated solve

`smash solve` uses the exact full-codebook GEMM search by default. The input
directory must contain `solve.json` with schema
`banana-smasher-solve-input-v1` plus the relative NPY vector and codebook files
declared by each cell. Run it with `smash solve --source-root
/path/to/solve-input --output /path/to/solve-output`. The accelerated path
requires CUDA, Triton, D=4 codewords, and a candidate count divisible by 64; it
fails loudly rather than silently switching implementations. Install the
optional runtime with `pip install -e '.[solve]'` on the CUDA host.

The command atomically publishes `winners.npz` and a concise
`SOLVE_RECEIPT.json` containing the backend, layer/shape, elapsed wall time,
and artifact location. Independent fast-versus-reference parity checks remain
in CI and do not add proof work to normal user runs.

## Accelerated update

Install the CUDA update dependencies with `pip install -e '.[update]'`, then run
the default full-depth update with `smash update --runtime-root
/path/to/runtime --model-root /path/to/DeepSeek-V4-Flash --aot
/path/to/aot/_C.so --output /path/to/updated.pt`. It is shown as prose so the
release section above remains the exact three-command copy/paste surface.

The shipped path is accelerated and fails loudly if its CUDA, AOT, runtime, or
geometry requirements are unavailable. It processes one 8,192-item logical
mean as eight 1,024-item physical segments and performs exactly one optimizer
step. Each completed backward is atomically checkpointed next to the output;
re-running the same command resumes automatically, while `--restart` explicitly
discards an incomplete run. A completed replay is idempotent and does not run
forward or the optimizer again.

The default receipt, `/path/to/updated.pt.receipt.json`, contains the backend,
geometry, elapsed wall time, output path, resume counts, and durable-completion
state. `--verbose-receipts` adds per-segment phases, parameter diffs, memory
details, and fallback metadata. Backend parity is enforced separately in CI and
does not add proof work or runtime fallback to this command.

## Complete teacher banks and paired evaluation

`smash bank` builds the whole ordered population declared by a
`bs-real-axis-windows-v1` manifest. It resumes valid manifest-bound members
automatically and publishes `bank.json` followed by `BANK_COMPLETE` only after
the exact member set, bytes, hashes, tensor schemas, order, instrument, and
population all verify. Re-running the same command is idempotent; incomplete or
mutated banks fail closed. The public invocation is `smash bank --model-root
/path/to/native-model --corpus /path/to/corpus --windows-manifest
/path/to/windows.json --output /path/to/bank`.

`smash evaluate` always performs a paired candidate/reference physical layer
walk over that complete bank. Use `smash evaluate --model-root
/path/to/native-model --candidate /path/to/candidate-pack --reference
/path/to/reference-pack --bank /path/to/bank --output /path/to/evaluation`.
Candidate and reference packs each declare a `real_axis.json` profile whose
per-layer tensor identities and descriptors drive the walk; topology is resolved
again for every layer rather than copied from a model-wide literal. The packaged
`real-axis-v1` instrument supplies support, cutoff, KLD direction, attention,
and estimator values.

Both arms checkpoint exact hidden states at each common completed layer.
Automatic resume selects only the greatest contiguous validated pair;
`--resume-from-layer N` requires the pair checkpoint ending at layer `N-1` and
never skips state. `evaluation.json` binds the arm artifact manifests,
per-position KLD, global/per-class/per-window summaries, teacher/candidate top-1
parity, paired deltas, pack identities, and layer descriptors.
`EVALUATION_COMPLETE` is published last. Normal stdout stays concise;
`--verbose-receipts` includes the durable evaluation object. Numerical parity
against the deterministic reference rail remains in CI, not the user runtime.

## Reproduction and evidence

- [`NIGHTLY_SEALED_RESULTS.md`](NIGHTLY_SEALED_RESULTS.md) is the compact
  receipt-bound release ledger.
- [`../FINAL_TABLE.md`](../FINAL_TABLE.md) is the newcomer-first size, quality,
  serving, and training comparison.
- [`../RESULTS.md`](../RESULTS.md) preserves the full experimental narrative;
  [`../LEARNINGS.md`](../LEARNINGS.md) records the reusable operational lessons.
- `SOURCE_MANIFEST.json` content-addresses every admitted package file, while
  `PUBLICATION_TRANSFORM.json` records the fail-closed public redaction contract.
  `pytest -q tests/test_unified_repo_contract.py` verifies those release
  surfaces together with the copy-paste commands and pinned source hashes.

Static image construction is reproducibility evidence, not a runtime quality
seal. The image remains **not GOLDEN** until a full model pack passes the three
commands above inside the release container.
