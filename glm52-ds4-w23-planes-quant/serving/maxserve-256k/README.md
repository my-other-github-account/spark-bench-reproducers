# MAXSERVE-256K one-Spark sizing receipt

Task: `t_627d955b`
Host: `spark-8` (one GB10; one vLLM EngineCore; no tensor/pipeline parallelism)
Verdict: **The 110.63 GB total-class R6 mixed artifact serves a full 262,144-token window and decodes normally. The larger W3v2 tier does not fit even at 32K.**

## Result table

Artifact GB uses exact expert-plane bytes plus the task-specified 7.55 GB FP8 non-expert payload. Both decimal GB and exact expert GiB are shown to avoid the campaign's GB/GiB ambiguity.

| artifact | artifact GB | context/request | prefill tok/s | decode tok/s | peak system used | fits |
|---|---:|---:|---:|---:|---:|:---:|
| W3v2 uniform + FP8 dense | 120.09 GB total est. (104.81 GiB expert planes) | 32,768 startup | — | — | 121.69 GiB; MemAvailable hit 0 | N |
| R6 96GiB mixed + FP8 dense | 110.63 GB total est. (96.00 GiB expert planes) | 32,768 startup + smoke | — | smoke passed | 120.77 GiB | Y |
| R6 96GiB mixed + FP8 dense | 110.63 GB total est. | 131,072 startup | — | — | 120.23 GiB | Y |
| R6 96GiB mixed + FP8 dense | 110.63 GB total est. | 261,888 prompt + 256 decode = 262,144 | **614.066** | **14.313 median** (3 runs) | **118.75 GiB**, 2.94 GiB min available | **Y** |
| same 262K deployment | 110.63 GB total est. | 8,192 prompt + 256 decode | 674.816 first run | **14.671 median** (3 runs) | 118.89 GiB | Y |

Observed one-Spark tier ceiling: **110.63 GB total-class fits**; **120.09 GB total-class fails**, so the empirical ceiling is bracketed in **[110.63, 120.09) GB decimal**. The largest artifact actually proven online at 256K is the 110.63 GB tier; this probe does not claim an unmeasured exact boundary inside that bracket.

## Full-window performance

- Real corpus: `$MISSION_ROOT/DS4_TEACHER/static/windows_ds4_eval.json`
- Corpus MD5: `1701920b4ba96dea0b18fe9df0151876`
- Prompt: 261,888 exact token IDs from 130 concatenated corpus windows
- Prompt-token SHA-256: `ff66d41d3899ba5b6b9120f7c249333a77d8a8d04fc845f9ca590d79aa730382`
- Output: 256 tokens per run, `ignore_eos=true`, temperature 0
- Cold prefix-cache-cleared TTFT: 426.482 s
- Cold prefill: **614.066 tok/s**
- Exact decode TPS runs: 14.332085, 14.292135, 14.313017
- Exact median decode: **14.313017 tok/s**
- Frame-based warm-32 median diagnostic: 14.227164 tok/s
- `<5 tok/s` alert: **NO** (minimum exact run 14.292135 tok/s)
- 8K exact decode TPS runs: 14.670662, 14.634683, 14.679841; median **14.670662 tok/s**

The primary exact decode rate uses server-reported `usage.completion_tokens`, not SSE-frame count. Run 1 emitted 256 tokens in 255 nonempty frames, proving why frame counting would understate throughput.

## Serve configuration

- vLLM: `vllm-0.24.0-53b723dd`
- Model: `$MODEL_ROOT`
- Expert planes: `$MISSION_ROOT/DS4_R6/planes_r6_96g` (103,079,412,731 bytes, 553 files)
- One Spark / TP1 / PP1; eager mode
- `--max-model-len 262144`
- `--kv-cache-dtype fp8`
- `--kv-cache-memory-bytes 2147483648` (2 GiB)
- `--block-size 256`
- `--max-num-batched-tokens 2048`
- `--max-num-seqs 1`
- `--gpu-memory-utilization 0.78` (explicit KV bytes control allocation)
- `--no-scheduler-reserve-full-isl`
- Chunked prefill: enabled
- Residency: all anonymous
- Systemd containment: `MemoryHigh=116G`, `MemoryMax=119G`, swap disabled
- Kernel path: mixed W2/W3/FP4 sentinel observed

Final startup receipt:
- Initial free memory: 116.09 GiB
- KV capacity: 303,775 tokens
- Maximum 262,144-token concurrency: 1.16x
- Peak system used: 127,505,637,376 bytes (118.75 GiB)
- Minimum MemAvailable: 3,157,532,672 bytes (2.94 GiB)
- Peak service cgroup memory: 107.47 GiB
- Swap used by service: 0
- Watchdog events: none
- Health remained HTTP 200 after the full-window request.

KV right-sizing mattered: a 4 GiB 262K launch initialized but was safety-stopped after MemAvailable stayed below 1 GiB. A 3 GiB launch passed with ~1.40 GiB minimum available. The final 2 GiB config still exposed 303,775 KV tokens (1.16x full-window concurrency) and increased measured minimum headroom to 2.94 GiB.

## Context ladder

- 32K: init + health + 16-token smoke completion passed; 114,577 KV tokens in the banked 3 GiB run.
- 128K: init + health passed; 426,345 KV tokens with 4 GiB KV.
- 262,144: init + health passed; final 303,775 KV tokens with 2 GiB KV; full 261,888+256 request passed.

## Capacity failure receipt

Uniform W3v2 loaded all 43 expert layers and hit the W3 on-path sentinel, then produced three kernel `NVRM NV_ERR_NO_MEMORY` events before KV/API health. MemAvailable reached zero. Exact W3 plane payload: 112,541,608,143 bytes (104.81 GiB). This is a real capacity failure, not a quality or kernel-routing failure.

## Evidence

Remote root: `$MISSION_ROOT/MAXSERVE_256K/`

Primary files:
- `out/FINAL_262K_SERVE_RECEIPT.json`
- `out/BENCH_261888_3RUNS.json`
- `out/BENCH_8K_3RUNS.json`
- `out/CONTEXT_RUNG_128K.json`
- `out/CONTEXT_RUNG_262K_FINAL.json`
- `out/W3V2_CAPACITY_ATTEMPT.json`
- `out/smoke_32k.json`
- `logs/serve_planes_r6_96g__mml262144_kv2147483648_anon_mbt2048.log`
- `logs/serve_moe_w3_planes_v2__mml32768_kv3221225472_anon_mbt2048.log`
- `src/launch_server.sh`, `src/start_rung.sh`, `src/memmon.py`, `src/bench_stream.py`, `src/seal_rung.py`
