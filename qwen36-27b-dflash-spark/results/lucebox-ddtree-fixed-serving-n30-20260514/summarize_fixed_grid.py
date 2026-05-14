#!/usr/bin/env python3
import json, sys, statistics as st, re, subprocess, pathlib
from pathlib import Path
D=Path(sys.argv[1])
cells=['dflash-sherlock-thinkON','dflash-sherlock-thinkOFF','dflash-codegen-thinkON','dflash-codegen-thinkOFF']
rows=[]
for tag in cells:
    p=D/f'{tag}.json'
    data=json.load(open(p))
    b=data['benchmarks'][0]
    vals=b['tg_throughput']['values']
    warm=vals[1:]
    tt=b.get('ttfr',{}).get('values') or []
    pp=b.get('pp_throughput',{})
    smoke=json.load(open(D/f'{tag}_tokenid_smoke.json'))
    blog=(D/f'{tag}_bench.log').read_text(errors='replace') if (D/f'{tag}_bench.log').exists() else ''
    slog=(D/f'{tag}_server.log').read_text(errors='replace') if (D/f'{tag}_server.log').exists() else ''
    counts=[]
    counts += [int(x) for x in re.findall(r'generated\s+(\d+)\s+tokens\s+in\s+[0-9.]+s\s*->', slog)]
    # Usually includes smoke + 30 measured. Keep measured-ish generated counts; exact wrapper counts are benchmark JSON response_size.
    dflash_tps=[float(x) for x in re.findall(r'generated\s+\d+\s+tokens\s+in\s+[0-9.]+s\s*->\s*([0-9.]+)\s*tok/s', slog)]
    fallback='No token_ids in response, using local tokenization' in blog
    row={
        'tag':tag,
        'corpus':'codegen' if 'codegen' in tag else 'sherlock',
        'think':'ON' if 'thinkON' in tag else 'OFF',
        'runs':len(vals),
        'warm_n':len(warm),
        'prompt_size':b.get('prompt_size'),
        'response_size':b.get('response_size'),
        'concurrency':b.get('concurrency'),
        'tg_values':vals,
        'tg_mean_all':st.mean(vals),
        'tg_median_all':st.median(vals),
        'tg_mean_warm':st.mean(warm),
        'tg_median_warm':st.median(warm),
        'tg_std_warm':st.pstdev(warm),
        'ttfr_median_warm_ms':st.median(tt[1:]) if len(tt)>1 else (st.median(tt) if tt else None),
        'pp_mean':pp.get('mean'),
        'fallback_detected':fallback,
        'tokenid_smoke':smoke,
        'server_dflash_generated_token_counts_tail':counts[-35:],
        'server_dflash_tps_tail':dflash_tps[-35:],
        'raw_json':str(p),
        'bench_log':str(D/f'{tag}_bench.log'),
        'server_log':str(D/f'{tag}_server.log'),
        'bench_cmd':str(D/f'{tag}_bench_cmd.sh'),
        'server_cmd':str(D/f'{tag}_server_cmd.sh'),
    }
    row['eligible_standard']= (row['runs']==30 and row['warm_n']==29 and row['prompt_size'] in (128,129,130,131,132,133,134,135,136,137,138,139,140,141,142) and row['response_size']==128 and row['concurrency']==1 and not fallback and smoke.get('noids')==0)
    rows.append(row)
summary={'receipt_dir':str(D),'rows':rows}
(D/'fixed_serving_grid_summary.json').write_text(json.dumps(summary,indent=2))
lines=[]
lines.append('# Fixed-serving DFlash N=30 grid summary')
lines.append('')
lines.append('Standard unpatched `llama-benchy`; server streams `choices[0].token_ids` on every content-bearing SSE chunk; no per-delta BPE fallback accepted.')
lines.append('')
lines.append('Shape: `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30 --no-cache --no-adapt-prompt --latency-mode none --skip-coherence`.')
lines.append('')
for r in rows:
    lines.append(f"- **{r['tag']}**: warm median **{r['tg_median_warm']:.3f} tok/s**, warm mean {r['tg_mean_warm']:.3f}, std {r['tg_std_warm']:.3f}, runs={r['runs']}, warm_n={r['warm_n']}, response_size={r['response_size']}, fallback={r['fallback_detected']}, tokenid_noids={r['tokenid_smoke'].get('noids')}, eligible={r['eligible_standard']}")
    lines.append(f"  - raw: `{r['raw_json']}`")
    lines.append(f"  - bench log: `{r['bench_log']}`")
    lines.append(f"  - server log: `{r['server_log']}`")
lines.append('')
lines.append('## Reproduction commands')
lines.append('')
lines.append('Each cell has exact command receipts in `*_server_cmd.sh` and `*_bench_cmd.sh`. The benchmark command uses N=30 as serial runs: `--runs 30 --concurrency 1`.')
lines.append('')
lines.append('Validation gates:')
lines.append('- `len(benchmarks[0].tg_throughput.values) == 30`')
lines.append('- `benchmarks[0].response_size == 128`')
lines.append('- `benchmarks[0].concurrency == 1`')
lines.append('- bench log contains no `No token_ids in response, using local tokenization`')
lines.append('- direct stream smoke has `noids == 0`')
(D/'FIXED_SERVING_N30_SUMMARY.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
