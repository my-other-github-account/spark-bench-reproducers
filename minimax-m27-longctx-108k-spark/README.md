# MiniMax-M2.7-UD-IQ4_XS @ 108K context — long-context stability reproducer

End-to-end Docker reproduction of **back-to-back-stable serving of
108K-token contexts** on a single NVIDIA DGX Spark (GB10 Blackwell, 128 GiB
unified memory, aarch64, Ubuntu 24.04) for `unsloth/MiniMax-M2.7-GGUF`
(UD-IQ4_XS, ~101 GB, 4 shards) on llama.cpp.

Sister recipe to [`minimax-m27-ngram-spec-spark`](../minimax-m27-ngram-spec-spark)
(which targets the leaderboard 30.98 t/s headline at c=32K with n-gram spec
decode). This recipe answers a different question: **can we serve 100K-token
prompts back-to-back with prefix caching, no OOM, and no run-by-run decode
collapse?**

Same image, same llama.cpp commit, same CUDA 13.0 NGC base. Only the launch
flags differ — see "Stability flags" below.

## Headline (5-phase battery + 10-run torture, single boot, c=108000 + the stability config)

```
Phase A — 5× back-to-back unique 100K       PASS  R5/R1 = 1.047  (decode trends UP)
Phase B — same-prompt 3× prefix-cache       PASS  speedup 144.7× (cold→cache hit)
Phase C/D — d=0 leak check (3+3 runs)       PASS  drift +1.18% over 60s settle
Phase E — d=0 headliner pp=128 tg=128 n=5   24.58 t/s median  (std 0.11)
Torture (10× back-to-back unique 100K)      PASS  R10/R1 = 1.006  median 5.56 t/s

Decode @ d=100100 (10 unique runs, n=10):    median 5.56 t/s, range 5.12–5.77
Prefill @ d=100100 (cold, ~94K tokens):       210 – 264 t/s (varies with depth)
Decode @ d=0 (post-stress, n=12 runs):       24.30 – 24.59 t/s (Phase C+D+E mean)
Prefix-cache reuse (Phase B 95K-token):      496.7s cold → 3.4s warm → 1.6s warm
Boot to ready                                ~485s (~8 min)
Peak unified memory                          ~120 / 128 GiB
Long-context decode tg2048 @ d=100K           6.05 t/s (n=2)
```

All phases auto-PASSED by the battery + torture script verdicts:
- Phase A: R5/R1 ≥ 0.9 AND 5/5 survived → 1.047, 5/5 ✅
- Phase B: send2 ≤ 5% of send1 → 0.69% (144.7×) ✅
- Phase C/D: |drift| ≤ 5% → 1.18% ✅
- Torture: R10/R1 ≥ 0.9 AND 10/10 survived → 1.006, 10/10 ✅

## Why this recipe exists

A naïve attempt at long-context (just bump `-c` to 108000) blows up in
multiple non-obvious ways:

1. **Prompt-cache pool OOM**: at end of each 100K request the server saves
   ~12 GiB of slot KV to a CPU-side cache pool. After 2-3 requests this
   pushes past the 128 GiB UMA ceiling and the kernel OOM-kills llama-server.
2. **Decode-rate collapse with naïve `-cram` cap**: `-cram 4096` tries to
   bookkeep the cap, but eviction logic only evicts entries *smaller* than
   the one being added. A single 12 GiB save never gets evicted, eviction
   thrashes during decode, and t/s collapses run-by-run: 5.6 → 3.9 → 2.0
   → 1.0.
3. **`--no-mmap --mlock` doubles memory use** by forcing the model into
   anonymous heap pages on top of the existing CUDA UMA buffer — this OOM-
   kills the server twice in a row at 64K. The intuition is wrong on UMA.

This recipe encodes the empirically-validated config:

```
-c 108000 -ctk q8_0 -ctv q8_0 -fa on
--cache-reuse 256 --ctx-checkpoints 0 --cache-prompt -cram 100
--no-context-shift -ngl 99 -np 1 --no-warmup -t 20
```

Verified to:
- Boot in ~7-9 min, fit in ~117 GiB / 128 GiB UMA (~11 GiB headroom)
- Survive N back-to-back unique 100K-token prompts (R_N/R_1 ≥ 0.9)
- Maintain prefix-cache hits (huge speedup on identical-prompt resends)
- Recover to baseline d=0 t/s after long-context stress (no slow leak)

## Quick start

Prerequisites on the host:

- DGX Spark (GB10 Blackwell, sm_121, aarch64, Ubuntu 24.04)
- NVIDIA driver 580.x (verify with `nvidia-smi`)
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~110 GB free disk for GGUF + ~10 GB for image
- **≥64 GiB swap space** (extra `swapfile-extra` recommended; default 16 GiB
  is enough for boot but may not absorb transient pressure during torture
  testing)
- HuggingFace CLI: `pip install --user huggingface_hub[cli]`

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers
cd spark-bench-reproducers/minimax-m27-longctx-108k-spark

# 1. Download model + tokenizer to ~/models (~101 GB GGUF + 17 MB tokenizer)
bash scripts/download_models.sh

# 2. Build the image (~15 min on first run; rebuilds cached <30 sec)
docker build -t minimax-m27-longctx-108k .

# 3. Verify the build (CRITICAL — silent corruption if this fails)
docker run --rm --entrypoint bash minimax-m27-longctx-108k -c \
  'readlink /usr/local/cuda/lib64/libcublas.so.13'
# expected: libcublas.so.13.0.0.19  (NOT 13.2.x or 13.3.x)
# If wrong, see "Base-image trap" in ../minimax-m27-ngram-spec-spark/README.md

# 4. Boot the 108K server (foreground; 7-9 min weight load)
docker run --rm -d --name mm-srv-longctx-108k --runtime=nvidia --gpus all \
    --network=host --shm-size=32g \
    -v ~/models:/models:ro \
    -v "$(pwd)/results:/repro/results" \
    minimax-m27-longctx-108k

# 5. Wait for /health and run the 5-phase battery (~75 min wall)
docker exec mm-srv-longctx-108k bash /repro/scripts/wait_for_server.sh
bash scripts/bench-battery.sh mm-srv-longctx-108k ./results

# Or measure single cells:
#   d=0 tg128 n=30 (the d=0 headline)
docker exec mm-srv-longctx-108k bash /repro/scripts/bench-tg.sh 0 30 128
#   d=100K tg128 n=5 (long-context decode rate; depth ≈ ctx-2K)
docker exec mm-srv-longctx-108k bash /repro/scripts/bench-tg.sh 100100 5 128
#   d=100K tg2048 n=3 (long decode tail; verifies decode rate holds for full 2K output)
docker exec mm-srv-longctx-108k bash /repro/scripts/bench-tg.sh 100100 3 2048

# 6. Optional torture: 10× back-to-back unique 100K (extends Phase A)
bash scripts/bench-torture.sh mm-srv-longctx-108k ./results 10
```

## Results — the 5-phase battery

The shipped result JSONs in `results/` came from a single-boot battery run
on a healthy DGX Spark (no swap pressure at start, no other GPU consumers).
Reproduce by running step 5 above; numbers should land within the cited
ranges. Server eval-time from the container log is the truth source for
all decode rates (llama-benchy aggregate occasionally has wall-clock
artifacts that look like outliers — see "Reading the results" below).

### Phase A — Stability (5× back-to-back unique 100K)

| Run | Decode tg128 (t/s) | Prefill (t/s) | TTFR (s) |
|---|---|---|---|
| 1 | 5.51 | 237.3 | 398 |
| 2 | 5.60 | 237.6 | 397 |
| 3 | 5.56 | 238.0 | 396 |
| 4 | 5.71 | 234.6 | 402 |
| 5 | **5.77** | 237.3 | 398 |

**R5/R1 = 1.047** (decode trends slightly *up* — zero degradation). 5/5 runs
survived. ✅ Pass bar: R5/R1 ≥ 0.9 AND 5/5 survive.

Source JSONs: `results/phaseA-r{1..5}-d100100.json`.

### Phase B — Prefix cache validation (same prompt 3×)

| Send | Wall (s) | Speedup vs cold |
|---|---|---|
| 1 (cold prefill ~95K tok) | 496.73 | 1× |
| 2 (cache hit) | 3.43 | **144.7×** |
| 3 (cache hit) | 1.63 | **304.7×** |

`--cache-reuse 256` is doing its job: send 2 reused the entire 95K-token
prefix at <1% of the cold-prefill wall-clock. ✅ Pass bar: send2 ≤ 5% of
send1 (achieved 0.69%).

### Phase C — d=0 post-stress (3× d=0 tg128)

Run 1: 24.46 t/s, Run 2: 24.05 t/s, Run 3: 24.39 t/s → mean **24.30 t/s**

After Phase A + B (5× unique 100K + 3× same-prompt long-context request),
d=0 inference is exactly at the documented baseline (skill notes 23-24 t/s
on this hardware). The long-context stress did not damage d=0 throughput.

Source: `results/phaseC-r{1..3}-d0.json`.

### Phase D — d=0 leak check (60s settle + 3× d=0 tg128)

Run 1: 24.48 t/s, Run 2: 24.26 t/s, Run 3: 25.02 t/s → mean **24.59 t/s**

**Drift vs Phase C: +1.18%** (well inside the ±5% pass bar). No slow
memory or state leak from cumulative stress. ✅

Source: `results/phaseD-r{1..3}-d0.json`.

### Phase E — Headliner d=0 grid (n=5, pp=128 tg=128)

```
tg_throughput (warm)   median 24.58  mean 24.53  std 0.11  n=4
ttfr ms (warm)         median 842    mean 844
pp_throughput          mean 104.1 tok/s
```

This is the d=0 throughput at c=108000. Notably tighter variance (std 0.11
vs ~4 t/s for the c=32K spec-decode headline) because spec decode is NOT
in play here — at long context, n-gram self-spec gets 0 acceptances on
free-form QA outputs (verified, see skill notes). Decode is purely
KV-bandwidth-bound, hence the low variance.

Source: `results/phaseE-d0-tg128-n5.json`.

### Prefill / decode characterization across depths

| Depth | Prefill (t/s) | Decode tg128 (t/s) | TTFR (s) | n | Source |
|---|---|---|---|---|---|
| 0      | 104.1 | **24.58** (median, std 0.11) | 0.84 |  5 | `phaseE-d0-tg128-n5.json` |
| 50,000 | 264.3 | **9.29** (mean, std 0.13)    | 197 |  3 | `depth-d50000-tg128-n3.json` |
| 75,000 | 233.6 | **6.91** (mean, std 0.21)    | 328 |  3 | `depth-d75000-tg128-n3.json` |
| 100,100 | 210.0 | **5.73** (mean, std 0.08)   | 480 |  3 | `depth-d100100-tg128-n3.json` |
| 100,100 (tg=2048) | 212.6 | **6.05** (mean) | 499 | 2 | `depth-d100100-tg2048-n2.json` |

**Decode is roughly KV-bandwidth-bound at long context: t/s ≈ 1/context after
~32K** (skill rule of thumb, confirmed empirically). Going from d=0 to
d=100K, decode drops 4.3× because every output token must attend over a 100K
KV cache instead of a near-empty one. Prefill stays ~210-260 t/s since
prompt processing is compute-bound on flash-attention.

The d=100K tg2048 cell (long output) actually runs slightly *faster* per
token than tg128 (6.05 vs 5.73) — once the cache is warm and the kernel-
selection autotuner has settled, longer outputs amortize the per-request
overhead. This is honest data, not a typo.

## Torture test — 10× back-to-back unique 100K (single boot)

Phase A's 5 runs + 5 more on the same boot, no restart. Each run sends a
fresh ~95K-token sherlock prompt through `llama-benchy --no-cache` and
decodes 128 tokens.

| Run | Decode tg128 (t/s) |
|---|---|
| 1  | 5.51 |
| 2  | 5.60 |
| 3  | 5.56 |
| 4  | 5.71 |
| 5  | 5.77 |
| 6  | 5.12 ← single mild dip |
| 7  | 5.56 |
| 8  | 5.69 |
| 9  | 5.54 |
| 10 | 5.54 |

```
Median over 10 runs : 5.56 t/s
Min                 : 5.12  Max: 5.77
R10 / R1            : 1.006   ← decode rate identical after 10 stress runs
```

**This is the strongest possible stability signal**: 10 unique 100K-prefix
requests on a single boot, with prefix caching enabled, and the 10th run
lands within 0.6% of the 1st. The skill canonical bar is R5/R1 ≥ 0.9; we
achieved R10/R1 = 1.006. ✅

Source JSONs: `results/phaseA-r{1..5}-d100100.json`,
`results/torture-r{6..10}-d100100.json`.

## Stability flags — discovery story

These flags were each discovered by failing N test cycles before landing on
the working combo. The skill `llamacpp-minimax-spark-bench` has the full
sweep table; below is the short version.

### `-cram 100` (NOT `-cram 4096`, NOT `-cram 0`)

`-cram` caps the CPU-side prompt-cache pool size in MiB. The cap is
**silently violated** when a single saved entry exceeds it: eviction logic
walks the cache from oldest → newest looking for entries to drop, but it
only drops entries **smaller** than the one being added. At c=108000 every
saved prompt is ~12 GiB, so once one is in, eviction can't dislodge it.

- `-cram 4096`: cap bookkept but violated. Eviction thrashes during decode.
  R1 5.6 → R2 3.9 → R3 2.0 → R4 1.0 t/s. **Container alive but UX dead.**
- **`-cram 100`**: cap so small the save-attempt path becomes effectively a
  no-op. No thrash. Decode steady at ~5.5 t/s across N runs. ✅ **THIS IS
  THE FIX.**
- `-cram 0`: would also work (per `--help`: "0 = unified KV and cache-ram"
  — disables the separate pool entirely) but cross-request prefix-cache
  pool is then disabled. Only chooses this if you don't want any cross-
  request cache.

The trade-off: `-cram 100` functionally disables the cross-request prompt-
cache pool. **Within-conversation cache via `--cache-reuse 256` still
works** — that's all you need for a single conversation reusing the same
context. Multi-tenant scenarios that re-use prefixes across users would
need a different fix (out of scope for this recipe).

### `--cache-reuse 256` + `--ctx-checkpoints 0`

- `--cache-reuse 256`: enables prefix-cache reuse for shared prefixes
  within the slot. Required for the Phase B prefix-cache speedup.
- `--ctx-checkpoints 0`: disables per-slot rewind checkpoints. Helps slow
  the leak but does NOT stop it on its own (verified — `cram 100` is the
  real fix). Kept here because it tightens memory use at zero cost.

### Why NOT `--mlock` / `--no-mmap`

- `--mlock` does NOT stop the leak (verified, R4 OOM at swap=80%).
- `--no-mmap`: gives **+27% decode at d=100K** (5.64 → 7.23 t/s) but does
  NOT stop the leak. **Use only for single-shot data collection** where
  you fresh-boot per request anyway, never for back-to-back stability.

### Why `--no-context-shift`

When the conversation reaches `-c`, llama-server's default behavior is to
evict the oldest tokens to make room (a "context shift"). For honest
long-context measurement we want OOM if we exceed `-c`, not silent
eviction. Set this flag to prevent silent shifts.

## What this image contains

| Component | Version / Source |
|---|---|
| llama.cpp | commit `45cac7ca7` (verified working on GB10 sm_121) |
| llama-server build flags | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121` |
| llama-benchy | ≥0.3.6 (installed via `uvx` at runtime) |
| Base image | `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` (NGC) |
| Model GGUF | `unsloth/MiniMax-M2.7-GGUF` path `UD-IQ4_XS/` (4 shards, ~101 GB) |
| Tokenizer | `MiniMaxAI/MiniMax-M2.7` (tokenizer.json, vocab.json, merges.txt only) |

**Do NOT** edit the `FROM` line to a Docker Hub `nvidia/cuda:13.2.x` or
`13.3.x` tag — those silently corrupt matmul on GB10. See `../minimax-m27-
ngram-spec-spark/README.md` "Base-image trap" for the full story.

## Reading the results

### Server eval-time is truth, llama-benchy aggregate is suggestion

llama-benchy occasionally reports artifact decode rates (e.g. 1.27 t/s on
an individual run while the server log shows 5.5 t/s for the same task).
The wall-clock includes a stall outside the eval window (HTTP finalization,
JSON serialization under brief swap pressure). **Always cross-check
`grep "eval time =" mm-srv-longctx-108k-server.log` for the real decode
rate** before declaring a regression.

### Pass bars (the user's "rock-stable" definition)

- **R_N/R_1 ≥ 0.9** for N=5 back-to-back unique 100K runs. Container
  surviving N runs is necessary but not sufficient — the canonical failure
  mode (-cram 4096) passes the alive-check but kills decode.
- **Phase B send2/send1 ≥ 20×** for prefix-cache validity.
- **|Phase D mean − Phase C mean| / Phase C mean ≤ 5%** for no-slow-leak.

### Variance expectations

n-gram self-spec is NOT used at long context here (verified: 0 acceptances
at depth=100K because the model's answer prose doesn't share 4-grams with
the long Sherlock prompt — see the spec-decode recipe for the c=32K case
where it works). At long context, throughput is decode-bandwidth-bound and
relatively low-variance: std ≈ 0.1-0.5 t/s on tg128 cells (vs ~4 t/s for
the 32K spec-decode cell whose variance is dominated by spec acceptance
luck).

## Pre-flight check — verify before trusting numbers

Same as the spec-decode recipe — see the
"Base-image trap" section in `../minimax-m27-ngram-spec-spark/README.md`.
Run the readlink + sha256sum check after `docker build`. The
`bench-tg.sh` script will also error out cleanly if the server isn't
healthy.

## License

MIT. Patches (none in this recipe) would be Apache-2.0 to match llama.cpp.

By [@banana_baeee](https://x.com/banana_baeee).
