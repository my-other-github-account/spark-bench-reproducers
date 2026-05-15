# True-AR baseline for fixed-serving Luce DDTree N=30 grid

This reruns the same standard/unmodified llama-benchy shape as the fixed-serving DFlash/DDTree grid, but with speculative/DDTree flags removed from the server wrapper:

- pp=128
- tg=128
- depth=0
- concurrency=1
- runs=30
- response_size=128
- standard llama-benchy; no harness patching
- stream chunks include choices[0].token_ids; validation rejects benchy local-tokenization fallback

Generated: 2026-05-14T22:37:06-07:00 on spark-3
Completed 2026-05-14T22:51:06-07:00
