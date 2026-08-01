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

QTIP tiers use the same public verb with a sealed config file or an ordered
config directory: `smash solve --source-root /path/to/source --root
/path/to/run --layer 39 --qtip-profile-config /path/to/config-or-directory`.
For a directory, `--qtip-units 64` selects the first resident batch. The
resident exact trellis kernel is the only runtime QTIP implementation: missing
Torch/Triton/CUDA, malformed or incomplete configs, and unsupported geometry
fail the verb loudly. There is no scalar/reference fallback after dispatch.

## Flash-full anchor campaign status

`smash status --run-root /path/to/run` is the human-first dashboard for the
0731 flash-full anchor campaign. It prints all five ordered tiers (`qtip3`,
`qtip2`, `d4_k2048`, `d4_k4096`, and the `mxfp4` reference), every required
layer from `L000` through `L042`, completed/active/missing unit totals,
manifest-derived percentages, current layer/batch/unit positions, newest
receipt age/path, mergeability, readiness, and explicit blockers. A truncated
d4 baseline manifest fails rather than redefining flash-full to a smaller set.
Missing layer ranges are compressed in the summary, but the detail section still
prints every layer token, so a range can never disappear behind a percentage.
Use `smash status --run-root /path/to/run --json` for the same exhaustive data
under `banana-smasher-anchor-campaign-status-v1`; tier order and all coverage
arrays are deterministic.

Status is a pure manifest/receipt reader. It requires `WORKFLOW_CHAIN.json`
and `anchors/MANIFEST.json`, validates every referenced byte count and SHA-256,
and reads only the declared aggregate manifests plus the exact
`anchors/<tier>/SHARDS.json` and `anchors/<tier>/RUNS.json` registries. It does
not glob mission trees or infer progress from filenames. An undeclared tier is
reported as entirely missing with the `smash solve`/`smash merge` producer
commands. A missing or malformed referenced artifact, stale hash, overlapping
shard, or expired active-run heartbeat exits nonzero and names both the exact
artifact and the public verb that must refresh it. Percentages use the expected
per-layer unit counts sealed by the two agreeing d4 anchor manifests rather
than a hard-coded denominator.

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
again for every layer rather than copied from a model-wide literal. Evaluation-
ready `BANANA_PACK_MANIFEST.json` files bind the complete runtime manifest,
every ordered layer-descriptor digest, and the head digest through their
`real_axis` seal; an unsealed or drifted profile is rejected before the walk.
The packaged `real-axis-v1` instrument supplies support, cutoff, KLD direction,
attention, and estimator values.

Both arms checkpoint exact hidden states at each common completed layer.
Automatic resume selects only the greatest contiguous validated pair;
`--resume-from-layer N` requires the pair checkpoint ending at layer `N-1` and
never skips state. `evaluation.json` binds the arm artifact manifests,
per-position KLD, global/per-class/per-window summaries, teacher/candidate top-1
parity, paired deltas plus the actual-population 95% paired interval, pack
identities, and layer descriptors. Verification recomputes KLD and top-1 parity
from the persisted support log-probabilities/argmax tensors, then recomputes all
arm and paired aggregates before accepting a coherently hashed receipt.
Every layer applies all declared windows in one batched candidate forward and
one batched reference forward; each arm also projects all final states in one
batched head forward. The sealed performance contract records the layer/head
forward counts, batch width, wall time, evaluated throughput, peak VRAM,
candidate/reference KLD, KLD delta, exact kernel name, and `fallback_used=false`.
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
