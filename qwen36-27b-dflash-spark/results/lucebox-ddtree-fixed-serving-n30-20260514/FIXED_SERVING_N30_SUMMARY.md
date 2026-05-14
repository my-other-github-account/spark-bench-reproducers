# Fixed-serving DFlash N=30 grid summary

Standard unpatched `llama-benchy`; server streams `choices[0].token_ids` on every content-bearing SSE chunk; no per-delta BPE fallback accepted.

Shape: `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30 --no-cache --no-adapt-prompt --latency-mode none --skip-coherence`.

- **dflash-sherlock-thinkON**: warm median **55.173 tok/s**, warm mean 55.416, std 16.469, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON_server.log`
- **dflash-sherlock-thinkOFF**: warm median **25.109 tok/s**, warm mean 26.341, std 5.439, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF_server.log`
- **dflash-codegen-thinkON**: warm median **46.587 tok/s**, warm mean 49.209, std 14.646, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON_server.log`
- **dflash-codegen-thinkOFF**: warm median **32.021 tok/s**, warm mean 33.104, std 6.896, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF_server.log`

## Reproduction commands

Each cell has exact command receipts in `*_server_cmd.sh` and `*_bench_cmd.sh`. The benchmark command uses N=30 as serial runs: `--runs 30 --concurrency 1`.

Validation gates:
- `len(benchmarks[0].tg_throughput.values) == 30`
- `benchmarks[0].response_size == 128`
- `benchmarks[0].concurrency == 1`
- bench log contains no `No token_ids in response, using local tokenization`
- direct stream smoke has `noids == 0`
