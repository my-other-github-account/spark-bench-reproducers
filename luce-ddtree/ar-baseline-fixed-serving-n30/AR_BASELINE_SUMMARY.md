# True-AR baseline summary

Shape: standard llama-benchy `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30`. Warm metrics drop first pass.

- True AR: server wrapper command removes `--fast-rollback --ddtree --ddtree-budget --ddtree-temp`; no llama-benchy patching.
- Streaming token accounting: requires `choices[0].token_ids`; fallback must be false.

| Row | eligible | median warm TG tok/s | mean warm | std warm | warm_n | fallback |
|---|---:|---:|---:|---:|---:|---:|
| true-ar-sherlock-thinkON | True | **33.768** | 34.119 | 9.082 | 29 | False |
| true-ar-sherlock-thinkOFF | True | **16.771** | 16.475 | 2.984 | 29 | False |
| true-ar-codegen-thinkON | True | **29.670** | 31.663 | 7.433 | 29 | False |
| true-ar-codegen-thinkOFF | True | **20.907** | 21.306 | 4.002 | 29 | False |

## Raw receipts
- `true-ar-sherlock-thinkON`: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkON.json`; server `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkON_server.log`; bench `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkON_bench.log`; generated_tail=[128, 128, 128, 128, 128]
- `true-ar-sherlock-thinkOFF`: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkOFF.json`; server `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkOFF_server.log`; bench `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-sherlock-thinkOFF_bench.log`; generated_tail=[128, 128, 128, 128, 128]
- `true-ar-codegen-thinkON`: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkON.json`; server `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkON_server.log`; bench `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkON_bench.log`; generated_tail=[128, 128, 128, 128, 128]
- `true-ar-codegen-thinkOFF`: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkOFF.json`; server `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkOFF_server.log`; bench `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706/true-ar-codegen-thinkOFF_bench.log`; generated_tail=[128, 128, 128, 128, 128]
