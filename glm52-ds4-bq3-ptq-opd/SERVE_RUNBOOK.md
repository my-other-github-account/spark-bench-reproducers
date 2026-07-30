# TRUE-C Wire Serving Runbook

Public operator identity: **banana_bae**

This is the public, fail-closed runbook for reproducing the product-serving claims. Replace bracketed locations with your own local paths. Model and tensor payloads are not redistributed here.

## 1. Bind the artifact before launching

The serving artifact is valid only when all authorities agree:

```text
byte envelope          101346700411
P943 overlay SHA-256   9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62
pack manifest SHA-256  3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220
planes manifest SHA-256 b524c5a67bbcad6aef14d70b464b46097302bf004bb75c1265f2ff683bae083d
```

Before every boot:

1. Check the canonical public status and manifests before claiming an authority is absent, then rehash the pack and planes manifests.
2. Check every expected layer and family identity against the manifests.
3. Verify the runtime source/launcher lock and purge stale bytecode caches.
4. Record the exact environment and argument vector.
5. Reject surrogate trees, empty stubs, mixed manifest generations, or copied headline numbers.

## 2. Prepacked-path preflight

The most expensive same-day false start came from a configured prepacked directory that held metadata stubs but no plane payloads. The loader silently fell back to the slow Python route.

Fail closed before launch:

```bash
test -s "[planes-root]/layer_000.meta.json"
test -s "[planes-root]/layer_000.planes13.npy"
test -s "[planes-root]/layer_000.sc13.npy"
test -s "[planes-root]/layer_000.planes2.npy"
test -s "[planes-root]/layer_000.sc2.npy"
```

Then verify that the installed runtime actually reads `VLLM_MOE_W2_PREPACKED_DIR`; finding the symbol only in backup files is a failure. Record the runtime source SHA. The current P943 adapter uses `p943_native_all_family` state; do not replace it wholesale with a pre-P943 module merely because an older concurrency patch lives there.

## 3. Launch contract

The passing serial product boot used:

- official vLLM OpenAI-compatible API;
- the P943 mixed four-family adapter;
- dense-all prefill path;
- both decode-graph controls enabled;
- `--enforce-eager` absent;
- `max-num-seqs=1` for the sealed serial product receipt;
- no speculative configuration and no MTP configuration;
- bounded KV cache and an 8 GiB runtime memory safety floor.

Use a versioned launcher, detached process group, stdin from `/dev/null`, unbuffered output, and durable PID/start-time/log receipts. Do not mutate a launcher in place while another boot is reading it.

Illustrative shape:

```bash
export VLLM_MOE_W2_PREPACKED_DIR="[planes-root]"
export VLLM_MOE_W2_DECODE_GRAPH=1
export P1016_TRUE_C_DECODE_GRAPH=1
export P530_PREFILL_MODE=dense_all

setsid vllm serve "[model-root]" \
  --served-model-name "[public-model-name]" \
  --max-num-seqs 1 \
  --kv-cache-memory-bytes "[bounded-bytes]" \
  --compilation-config '{"cudagraph_capture_sizes":[1]}' \
  >"[log-root]/server.log" 2>&1 </dev/null &
```

The exact passing serial boot predates the capture-limited concurrency experiment; do not claim the illustrative compilation setting is part of the serial receipt unless your own receipt proves it. Never add `--enforce-eager` by template habit: it disables CUDA graph capture.

## 4. Memory and swap law

Before boot and before every measured cell:

1. require enough free filesystem capacity for logs/cache while preserving the operator floor;
2. require at least 12 GiB `MemAvailable` before measurement and never permit less than 8 GiB;
3. fully drain operating-system swap when authorized, then verify `SwapFree==SwapTotal`;
4. verify API and engine `VmSwap=0`;
5. seal the preflight as a receipt.

A mid-rail swap reset does not retroactively make earlier rows VmSwap0. Writable private pack pages can fault into swap despite apparently adequate `MemAvailable`; this caused a run with good decode/prefill to fail at `51.695707` s warm TTFT and `4.611 GB` maximum session swap.

## 5. Warmup before READY

A cold boot is not READY. The observed cold-to-warm decode curve needed three decode warmups before settling near 16 tok/s.

Required unmeasured warmup set:

1. three serial decode requests with the same 256-token completion shape used for measurement;
2. one exact-2048-token prefill request;
3. one exact-8192-token prefill request;
4. verify health, model identity, memory floor, and `VmSwap=0` again.

Bake the resulting Triton/kernel cache into a portable image only after binding its cache manifest to the same runtime and architecture. A baked cache is an optimization, not an excuse to skip the warm READY gate.

## 6. Client measurement protocol

### Decode256

Use OpenAI-compatible streaming with `stream_options.include_usage=true`. Count `usage.completion_tokens`, not SSE lines, and calculate:

```text
post_TTFT_decode_tok_s = (completion_tokens - 1) / post_TTFT_seconds
```

The sealed rows each had exactly 256 completion tokens. Aggregate/server throughput is not single-client decode evidence.

Serial product gate:

- five consecutive decode256 rows;
- each row ≥10 tok/s;
- CV ≤15%;
- same boot, same artifact, same endpoint identity;
- raw token counts and timing fields retained.

### Prefill

Use exact token-ID prompts so the receipt can prove 2048/2048 and 8192/8192 prompt tokens. Record TTFT and server-side prefill throughput. Product bars:

- exact-2048 prefill ≥400 tok/s;
- warm TTFT ≤2.5 s;
- exact-8192 completion present.

### Coherence and stability

On the same boot, run the canonical semantic/logit repeat probe: fixed prompts, temperature zero, token-0/top-k/argmax stability, repeated outputs, and a deliberate-corruption negative control where applicable. Small quantization differences are allowed; a bit-exact language-model gate is not required.

## 7. Concurrency protocol and current status

For C=1/2/4/8:

1. use identical prompt/output/token accounting;
2. run three rounds per new M shape;
3. exclude round 1 only as the preregistered first-shape compile warmup;
4. prove request overlap and zero hidden queue serialization;
5. record per-stream and aggregate tok/s, TTFT, residency, `MemAvailable`, and `VmSwap`;
6. require strict aggregate monotonicity `C1 < C2 < C4 < C8`.

Current status at the publication cut:

| concurrency | aggregate tok/s | TTFT | memory | validity |
|---:|---:|---:|---:|---|
| 1 | 15.351984 | 3.20878 s decode / 2.44139 s exact-2K | 48.8 GiB residency; 157,532,160 B swap | measured, strict-safety FAIL |
| 2 | 2.845162 | retained in private raw receipt | swap reappeared | measured NO-GO; `0.185329×` C1 |
| 4 | TBD | TBD | TBD | blocked until same-method C2 >1.2× C1 |
| 8 | TBD | TBD | TBD | blocked until same-method C2 >1.2× C1 |

No valid actual-Wire-C monotonic ladder exists at this cut. C=4/C=8 must not run or be published until a repaired same-method C=2 row exceeds `1.2×` C=1. Kernel-only M4/M8 harness results are implementation support, not product-server rows.

## 8. Failure-signature catalog

| signature | likely mechanism | required response |
|---|---|---|
| both graph flags are `0` | conservative debug launcher survived | fix only the versioned launcher, then prove capture in startup logs |
| hardcoded `--enforce-eager` | graph capture disabled by command line | remove it; rehash launcher; run cheap decode probe first |
| malformed compilation JSON | concurrent/mutable launcher edit | immutable launcher file; validate argv before allocating |
| configured prepacked directory has only metadata | silent Python fallback | require all four plane payloads before boot |
| runtime does not read `PREPACKED_DIR` | reverted installed source | compare source SHAs, restore reviewed runtime, purge bytecode |
| `KeyError: planes13` / missing `sc13` | pre-P943 source cannot consume P943 adapter state | restore P943 state registration/dispatch; do not synthesize keys |
| capture error at `torch._assert` | tensor boolean/assert executed during capture | use reviewed capture guard or piecewise capture |
| capture error at `torch.unique_consecutive` or `.item()` | native mblock contains capture-unsafe host sync | limit outer capture shapes or use reviewed capture-safe branch |
| MTP changes T=1 to T=2 and crashes capture | M=1 packed route no longer selected | remove MTP for bare proof; do not relabel speculative throughput |
| 2–3 tok/s despite capture logs | Python grouping/FWHT remains on hot path | verify native kernel counters; flags cannot repair Python dispatch |
| C2 collapses below C1 | eager C>1 fell off packed fast path or first-shape compile | exclude only preregistered first round; inspect x-shape and `valid_m` guards |
| C8 aliases C4 | fixed four-row `valid_m=min(C,4)` software contract | M8 support needs its own reviewed path and product rerun |
| warm TTFT tens of seconds plus nonzero swap | pack pages faulted into swap | full pre-rail swap drain; prove both process `VmSwap=0` |
| health exits at memory floor under queued load | concurrency/KV headroom violation | serialize consumers, compact before boot, preserve 8 GiB floor |
| source pack cannot fit lawful capacity | storage/headroom capability blocker | do not copy/delete blindly; select a host with coherent read-only assets |
| low first warmup, normal later rows | cold JIT/M-shape compile | three unmeasured warmups; never publish cold row as steady state |
| aggregate throughput presented as decode | wrong measurement authority | retain usage-counted client timing and exact completion-token count |
| surrogate row lacks P943/pack/planes line | wrong artifact | reject; do not compose with TRUE-C receipts |

## 9. Receipt schema

Every public claim should preserve at least:

```json
{
  "artifact": {
    "byte_envelope": 101346700411,
    "overlay_sha256": "9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62",
    "pack_manifest_sha256": "3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220",
    "planes_manifest_sha256": "b524c5a67bbcad6aef14d70b464b46097302bf004bb75c1265f2ff683bae083d"
  },
  "boot": {
    "runtime_sha256": "<sha256>",
    "launcher_sha256": "<sha256>",
    "speculative_config": null,
    "same_boot": true
  },
  "measurement": {
    "usage_counted": true,
    "completion_tokens_per_decode_row": 256,
    "validity": "MEASURED_PASS|MEASURED_NO_GO|BLOCKED_UNAVAILABLE"
  }
}
```

Public receipts must scrub local paths, private host labels, addresses, process identifiers, and internal task IDs while preserving numeric measurements and source/public SHA relationships.

## 10. Current sealed serial result

The product receipt passed 18/18 gates with receipt SHA-256:

`3117274cf826804437509475a2294ea773d9ee5e64723df9f657c0123c28a413`

See [`CURRENT_BEST.md`](CURRENT_BEST.md) for the exact rows and validity boundaries.
