# Banana Smasher

Banana Smasher is a self-contained, SHA-sealed prototype that profiles a Hugging Face model, prices a mixed quantization menu, wires the solver, repair, pack, serving, and EvalPlus stages, and preserves the real builders and runtime sources needed to execute those stages.

The package is deliberately honest about validity. `--dry-run` performs no network access and writes nothing. `init` performs a real local or Hugging Face config profile. The other live commands currently seal a `PASS_PROTOTYPE_CONTRACT` execution plan with `PROJECTED` or `UNMEASURED` validity; they do not pretend that a GPU build or measurement happened. The vendored tools are the implementation surface for the physical executor.

The recovered research bundle is shipped under
`vendor/recovered/glm52_research_source_bundle_v1`. It contains all 89 recovered
source entries, their receipts, adoption map, missing-source ledger, and
privacy-bounded provenance. `tools/verify_recovered_sources.py` checks every
shipped hash and requires working code plus a receipt gate for each promotable
family. P526, P948, and P950 remain fail-closed research/hold evidence and are
not promoted as product wins.

## Five-minute quick start

```bash
cd banana-smasher

# Complete offline wiring check: all subcommands, no writes and no network.
for stage in init capture anchors anchor-mix grid solve retrodict build measure calibrate resolve repair pack serve eval status; do
  ./smash --workspace ./workspace "$stage" --dry-run
done

# Real model profiling from a local config directory or a Hugging Face model ID.
./smash --workspace ./workspace init \
  --model organization/model \
  --budget-bytes 91000000000 \
  --node-ram 128

./smash --workspace ./workspace status
./smash verify --manifest --self-contained
```

A local model path is fully offline. A Hugging Face identifier downloads only `config.json` during `init`; `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is read from the environment for gated models and is never recorded.

## Command surface

| Command | Contract |
|---|---|
| `init` | Profile architecture, expert/projection shapes, budget, and emit a menu template. |
| `capture` | Plan teacher-forward tensor capture and Balanced64 or full-window KLD banks. |
| `anchors` | Build and price uniform QTIP3, QTIP2, true D=4 VQ at K 1024/2048/4096, and MXFP4 anchors. |
| `anchor-mix` | Measure the one deliberate mixed-tier interaction term. |
| `grid` | Assemble the priced grid and salience shares. |
| `solve` | Seed by paired swaps, then wire the SCIP-backed exact-envelope solve. |
| `retrodict` | Refuse when any measured-wire prediction error exceeds 5 percent. |
| `build` | Materialize the assignment with per-cell payload seals and shard receipts. |
| `measure` | Read the same-instrument six-class and global KLD rail. |
| `calibrate` | Fit residual families from all measured wires and seal corrected pricing. |
| `resolve` | Solve definitively on the corrected grid. |
| `repair` | Apply 24 exact-inventory sparse repair updates with per-update seals. |
| `pack` | Seal a serving pack and resident-envelope inventory. |
| `serve` | Wire the mixed-tier backend, prefill server, residency, throughput, and logit gate. |
| `eval` | Generate through the endpoint, pair raw and sanitized rows, and score pinned EvalPlus. |
| `status` | Seal and print the receipt ledger. |

Every stage accepts `--dry-run`. Dry runs validate contracts and emit JSON to stdout without creating `workspace`.

## Durability and isolation

- Each stage writes only `workspace/<stage-name>`.
- Prerequisite stage receipts are read-only inputs; their SHA-256 values are sealed into the next receipt.
- Files are written through a temporary file, flushed, fsynced, and atomically renamed.
- The receipt is committed last.
- An identical rerun returns `ALREADY_COMPLETE` without rewriting outputs.
- Input drift against an existing receipt refuses and requires a new workspace.
- Projections, plans, and unmeasured output can never be relabeled as measurements.

The exact machine-readable contract is `contracts/STAGE_CONTRACTS.json`; receipt and output schemas are in `schemas`.

## Vendored implementation surface

`vendor/VENDOR_INDEX.json` maps every redistributed file to its capability, shipped SHA-256, and original source SHA-256. It includes:

- QTIP REP16 and rate-aware builders.
- True D=4 VQ wire builders and streaming materializers.
- DeepSeek-v4 MXFP4 selector logic.
- Balanced64 and full-rail measurement mechanics.
- Seeded exact-envelope and corrected-grid solver sources.
- P930 residual fitting and pricing adapters.
- P959 corrected sealed-wire inventory seed logic and sparse repair sources.
- Streaming packers, mixed-tier runtime, CUDA kernel source, and build contract.
- EvalPlus at commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`, HumanEvalPlus-v0.1.10, endpoint generation, paired sanitization, and four timing cells.
- The dense-all prefill route, all four decode classes, architecture-specific kernel-cache warmup, 128 GB UMA residency checks, replay-gated accelerated rail, >=8-stream mover, hardlink-first materialization, deterministic four-way sharding, per-codebook streaming, and speculative warm/revoke gates. `configs/EXPECTED_PERF.json` is the fail-closed READY authority; historical measurements in `receipts/ACCELERATION_RECEIPTS.json` are evidence, never substitutes for a fresh 2K run.

No symlink or package file reference escapes this folder. `./smash verify --self-contained` scans that invariant and the privacy law.

## Container

The package-root `Dockerfile` installs the pinned vLLM runtime, copies only package-local runtime and kernel assets, builds the CUDA extension, validates a mounted pack, starts the OpenAI-compatible server, and seals startup smoke metrics. Build from this folder:

```bash
docker build -t banana-smasher:0.1 .
docker run --rm --gpus all -p 8000:8000 -v "$PWD/workspace/pack:/model:ro" banana-smasher:0.1
```

The model pack is intentionally mounted rather than embedded in the image. Startup bakes and verifies every shipped kernel class, sends a fresh 2K smoke request, and emits `READY` only for prefill >=1000 tok/s, decode >=15 tok/s, TTFT <=2.5 seconds, exact residency, and complete cache proof. Any failed gate writes `DEGRADED` and exits without advertising readiness.

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q banana_smasher tools vendor
python3 tools/build_manifests.py
python3 tools/verify_recovered_sources.py
./smash verify --manifest --self-contained
```

The package's historical mixed-tier throughput receipts do not constitute an
Atlas-on-spark-6 served-throughput result or a Hugging Face kernel publication.
Those remain separate success gates and must not be inferred from this package.

See `docs/PROVENANCE.md` for upstream versus shipped identities and `docs/VALIDITY.md` for promotion gates.
