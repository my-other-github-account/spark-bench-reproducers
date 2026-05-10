# PP2048 Root Cause / Blocker - 2026-05-10 Correction

The replacement grid was rerun on the correction host from the clean main worktree with exact requested shape and normal long-lived serving:

- PP rows: 2048, 16384, 32768, 65536, 131072
- TG128/C1/MBT2048, DFlash ON, FlashQLA ON, nspec15
- max_model_len=132000, shifted suffix block-table patch installed, direct attention/query-QKV/fused QK-RoPE/compact-delta no-sync/full CUDAGraph patch set installed
- temperature=0.6 payload patch only; top_p/min_tokens/ignore_eos are not forced
- prefix caching disabled; warmup JSON excluded from measured JSON

Measured replacement PP2048 row:

- raw JSON: `results/golden-depth-grid-20260510/mbt2048-dflash_on-fqla_on-nousage-threshold0-nspec15-temp06-mlen132000-shifted_suffix_block_table_fullgraph_compact_delta_nosync_direct_attention_kvupdate_query_qkv_fused_qk_rope-warm1-n3-pp2048/measured-pp2048-tg128-c1-n3.json`
- exact shape in JSON: prompt_size=2048, response_size=128, concurrency=1, runs=3
- pp_throughput mean=1934.7244989521257 tok/s, values=[1908.9747872004393, 1983.0377710883206, 1912.1609385676168]
- tg_throughput mean=7.732378910250621 tok/s, values=[7.748711293029551, 7.720508548666127, 7.727916889056185]

Concrete blocker for the PP2048 >3000 gate:

1. The exact required MBT2048/nspec15 shape forces vLLM to set `max_num_scheduled_tokens=2034` because speculative draft slots are reserved under MBT2048.
2. llama-benchy PP2048 chat requests arrive at the engine as `prompt=2049`, so the prefill cannot fit into one scheduled step. The server log records split prefill, including `prompt=2049 computed=0` followed by tail scheduling. This is not the historical one-shot PP2048 shape.
3. The active correction patch path extracts/logs DFlash longctx feature metadata and auxiliary hidden states on those scheduled chunks. The PP2048 and long-context rows show NaN aux hidden stats during chunked prefill, while output is preserved and the engine remains healthy.
4. Controls did not recover >3000 under the current correction path: TG32 exact was ~1912 tok/s, MBT4096/nspec15 was ~2269 tok/s, DFlash-OFF/FlashQLA-ON exact was ~2476 tok/s, threshold1024 exact was ~2094 tok/s, and historical-like MBT8192/nspec8/TG32/`generation-config vllm` was ~2054 tok/s in this worktree.

Therefore the old PP2048 >3000 evidence is not valid for this replacement grid. It came from a different/bypass regime and is retracted; this correction keeps the exact-shape measured row and records the blocker instead of marking PP2048 PASS.
