# Luce DDTree fixed-serving N=30 reproducer

This package now separates the Luce results into three tiers:

- **AR**: true non-speculative llama.cpp `llama-server` baseline. Timings are the **llama-benchy-reported** timings from a patched server that streams OpenAI `choices[0].token_ids`; they are not derived from `llama-server` logs.
- **+DF**: the previous `ar-baseline-fixed-serving-n30/` rows, relabeled as **DFlash-only/no-DDTree** (`ddtree=0`). `AR_BASELINE_INVALIDATION.md` is preserved because those rows are invalid as AR.
- **+DDT**: the existing DFlash + DDTree fixed-serving rows.

## Benchmark shape

- `llama-benchy 0.3.6`, `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30`.
- `N=30` means 30 independent single-concurrency runs, not concurrency 30.
- Headline numbers are warm medians: first pass dropped, N=29.
- Token accounting requires streamed `choices[0].token_ids`; logs are rejected if `No token_ids in response, using local tokenization` appears.

## Headline warm median TG throughput and ratios

| corpus | think | AR tok/s | +DF tok/s | +DDT tok/s | +DF/AR | +DDT/AR | +DDT/+DF |
|---|---:|---:|---:|---:|---:|---:|---:|
| sherlock | ON | 10.298 | 33.768 | 55.173 | 3.28x | 5.36x | 1.63x |
| sherlock | OFF | 10.694 | 16.771 | 25.109 | 1.57x | 2.35x | 1.50x |
| codegen | ON | 10.112 | 29.670 | 46.587 | 2.93x | 4.61x | 1.57x |
| codegen | OFF | 10.569 | 20.907 | 32.021 | 1.98x | 3.03x | 1.53x |

## Raw result locations

- `true-ar-openai-tokenids-n30/`: true AR `llama-server` + OpenAI `token_ids` receipts/logs/smoke files.
- `ar-baseline-fixed-serving-n30/`: DFlash-only/no-DDTree (`ddtree=0`) receipts. Preserved invalidation: not an AR baseline.
- `dflash-ddtree-fixed-serving-n30/`: DFlash + DDTree receipts.
- `combined_results.json`: machine-readable three-tier table and ratios.
- `AR_BASELINE_INVALIDATION.md`: historical invalidation note for the old AR interpretation.

## Validation

- True AR run source: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-true-ar-n30-20260515_182628` on `spark-3`.
- Required true-AR cells are present: `true-ar-sherlock-thinkON`, `true-ar-sherlock-thinkOFF`, `true-ar-codegen-thinkON`, `true-ar-codegen-thinkOFF`.
- Each true-AR stream smoke file contains OpenAI streaming `token_ids`.
- True-AR bench logs contain no `No token_ids in response, using local tokenization` fallback.
- +DF and +DDT package rows remain the prior fixed-serving receipts; +DF is intentionally relabeled from the invalid AR attempt, not rerun.

## Reproduce true AR rows

On `spark-3` with the patched llama.cpp server and Luce workspace present:

```bash
cd /home/user/work/dflash-lucebox-gb10-spark3
./run_true_ar_n30.sh
```

The runner smoke-tests streamed `token_ids` before accepting a cell and rejects llama-benchy local-tokenization fallback.
