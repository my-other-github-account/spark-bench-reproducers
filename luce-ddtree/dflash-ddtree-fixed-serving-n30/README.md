# Lucebox Qwen3.6-27B DFlash fixed-serving N=30 grid

Created 20260514_164136 on spark-3. This reruns the 4-cell DFlash grid after fixing the OpenAI/SSE wrapper to stream exact `choices[0].token_ids` on every content-bearing chunk, so standard unpatched `llama-benchy` does not fall back to local per-delta BPE tokenization.

Grid:
- Sherlock / think ON
- Sherlock / think OFF
- Codegen / think ON
- Codegen / think OFF

Shape: `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30`, `--no-cache --no-adapt-prompt --latency-mode none --skip-coherence`.

Code corpus: `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py`.

Server config: Lucebox DFlash `test_dflash`, target `Qwen3.6-27B-Q4_K_M.gguf`, draft `models/draft`, budget=18, ddtree-temp=1.05, max_ctx=1024, ctk/ctv=f16, fa-window=2048, prefix/prefill caches disabled, `--ignore-eos-stop`, `--ddtree-no-chain-seed`.
