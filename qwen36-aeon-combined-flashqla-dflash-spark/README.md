# Qwen3.6 Aeon combined FlashQLA + DFlash on DGX Spark

This is the corrected **Aeon combined** configuration folder for Qwen3.6-27B on DGX Spark: FlashQLA for long-prompt prefill plus DFlash/nspec15 for decode, served from one long-context vLLM deployment.

**Status:** N=30 measured, raw JSON committed. This is intentionally truthful: **4 of 5 prompt depths beat the reference TG baseline; PP32768 does not.** Do not cite this as a uniform win across the grid.

- Target: Qwen3.6-27B NVFP4 at `/models/Qwen3.6-27B-NVFP4`
- Drafter/spec model: `/models/Qwen3.6-27B-DFlash`
- Hardware: DGX Spark GB10 Blackwell, sm_121, aarch64, Ubuntu 24.04
- Server: vLLM `v0.19.2rc1.dev213+g9558f4390.d20260426`
- Bench: `llama-benchy` 0.3.7
- Shape: `TG=128`, `C=1`, `MBT=2048`, `nspec=15`, vLLM default sampling/temp 0.6
- Long-context server: `max_model_len=262144`
- Measurement policy: one single long-lived server, one same-shape warmup per row excluded, `N=30` measured per row, no server restart/reconfig between rows

## Results — N=30 measured grid

| PP | TG | max_model_len | mean PP tok/s | median PP | mean TG tok/s | median TG | TG CV | reference TG | ratio | verdict | raw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2048 | 128 | 262144 | 2371.62 | 2338.60 | 20.39 | 20.07 | 12.72% | 11.60 | 1.76x | ✅ faster | [`json`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/single-server-llamabenchy-v34-mlen262144-warm1-n30-pp2048/measured-pp2048-tg128-c1-n30.json) |
| 16384 | 128 | 262144 | 2781.08 | 2781.93 | 18.97 | 18.49 | 11.88% | 11.08 | 1.71x | ✅ faster | [`json`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/single-server-llamabenchy-v34-mlen262144-warm1-n30-pp16384/measured-pp16384-tg128-c1-n30.json) |
| 32768 | 128 | 262144 | 2521.29 | 2524.24 | 17.87 | 17.64 | 12.09% | 24.44 | 0.73x | ❌ slower | [`json`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/single-server-llamabenchy-v34-mlen262144-warm1-n30-pp32768/measured-pp32768-tg128-c1-n30.json) |
| 65536 | 128 | 262144 | 2086.81 | 2087.64 | 17.10 | 16.68 | 21.72% | 9.66 | 1.77x | ✅ faster | [`json`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/single-server-llamabenchy-v34-mlen262144-warm1-n30-pp65536/measured-pp65536-tg128-c1-n30.json) |
| 131072 | 128 | 262144 | 1543.34 | 1543.33 | 12.62 | 12.54 | 12.44% | 8.23 | 1.53x | ✅ faster | [`json`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/single-server-llamabenchy-v34-mlen262144-warm1-n30-pp131072/measured-pp131072-tg128-c1-n30.json) |

## Read this before using the headline

The stable combined folder is useful because it proves a single full-context deployment can run all five depths with FlashQLA+DFlash markers active. It is **not** a clean 5/5 TG speedup claim:

- Passes TG reference at PP2048, PP16384, PP65536, PP131072.
- Fails TG reference at PP32768: mean `17.87 tok/s` vs reference `24.44 tok/s` (`0.73x`).
- PP65536 has high decode variance (`21.72%` CV) because one run hit `34.63 tok/s`; inspect raw values before quoting only the mean.

## Proof markers

- Single server startup count: `1`
- `max_seq_len=262144`: `True`
- GPU KV cache size: `288576` tokens
- `Maximum concurrency for 262,144 tokens per request: 2.99x`
- DFlash direct draft-query marker present: `True`
- FlashQLA marker present: `True`
- Traceback count in final server log: `0`
- EngineDeadError count in final server log: `0`

## Key files

- N=30 summary: [`results/n30-summary.json`](results/n30-summary.json)
- Run driver: [`run_single_server_llamabenchy_allpp_v34_mlen262144_warm1_n30.sh`](run_single_server_llamabenchy_allpp_v34_mlen262144_warm1_n30.sh)
- Full artifact root: [`results/golden-depth-grid-single-server-llamabenchy-n30-20260513/`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/)
- Server proof excerpt: [`results/golden-depth-grid-single-server-llamabenchy-n30-20260513/server-candidate-llamabenchy-single-server-allpp-v34-mlen262144-n2-warm1-n30/server-proof-excerpt.log`](results/golden-depth-grid-single-server-llamabenchy-n30-20260513/server-candidate-llamabenchy-single-server-allpp-v34-mlen262144-n2-warm1-n30/server-proof-excerpt.log)

## Reproduction sketch

```bash
cd qwen36-aeon-combined-flashqla-dflash-spark
bash scripts/download_models.sh
# Build the exact image tag expected by the measured run driver.
docker build -t qwen36-fqla-baseline-dflash-spark:combined-20260507-threshold10 .

# GB10 host cleanup before long-context serving.
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo swapoff -a || true
sudo swapon -a || true

# Runs PP2048, PP16384, PP32768, PP65536, PP131072 on one server.
./run_single_server_llamabenchy_allpp_v34_mlen262144_warm1_n30.sh
```

The run driver writes a new `results/golden-depth-grid-single-server-llamabenchy-n30-YYYYMMDD/` artifact. Verify server logs for `max_seq_len=262144`, `DFlash direct draft-query attention v2 path active`, and `[flashqla-v2] active` before quoting results.

## Raw N=30 values

Raw per-run values are in the committed llama-benchy JSON files linked in the results table. Recompute with:

```bash
python3 - <<'PY'
import json, statistics as st, pathlib
root = pathlib.Path('results/golden-depth-grid-single-server-llamabenchy-n30-20260513')
for pp in [2048, 16384, 32768, 65536, 131072]:
    p = root / f'single-server-llamabenchy-v34-mlen262144-warm1-n30-pp{pp}' / f'measured-pp{pp}-tg128-c1-n30.json'
    b = json.loads(p.read_text())['benchmarks'][0]
    tg = b['tg_throughput']['values']
    pref = b['pp_throughput']['values']
    print(pp, 'prefill_mean', st.mean(pref), 'tg_mean', st.mean(tg), 'tg_median', st.median(tg), 'n', len(tg))
PY
```
