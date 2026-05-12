#!/usr/bin/env python3
import argparse, json, statistics, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
parser.add_argument('--model', default='qwen35-27b-axionml-nvfp4')
parser.add_argument('--tokenizer', default='/home/user/models/AxionML-Qwen3.5-27B-NVFP4')
parser.add_argument('--pp', type=int, default=2048)
parser.add_argument('--tg', type=int, default=32)
parser.add_argument('--warmup-runs', type=int, default=2)
parser.add_argument('--runs', type=int, default=4)
parser.add_argument('--disable-spec-decode', action='store_true')
parser.add_argument('--out', required=True)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
text = Path('/home/user/.cache/llama-benchy/cc6a0b5782734ee3b9069aa3b64cc62c.txt').read_text(errors='ignore')
ids = tok.encode(text, add_special_tokens=False)[:args.pp]
if len(ids) != args.pp:
    raise SystemExit(f'got {len(ids)} prompt tokens, need {args.pp}')
url = args.base_url.rstrip('/') + '/completions'

def one(i):
    payload = {
        'model': args.model,
        'prompt': ids,
        'max_tokens': args.tg,
        'temperature': 0,
        'stream': True,
        'ignore_eos': True,
    }
    if args.disable_spec_decode:
        payload['vllm_xargs'] = {'disable_spec_decode': 1}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    first = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            now = time.perf_counter()
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line or not line.startswith('data:'):
                continue
            body = line[5:].strip()
            if body == '[DONE]':
                break
            if first is None:
                first = now
            chunks += 1
    t1 = time.perf_counter()
    if first is None:
        first = t1
    ttft = first - t0
    total = t1 - t0
    return {'run': i, 'ttfr_s': ttft, 'total_s': total, 'chunks': chunks, 'pp_tps': args.pp / ttft, 'tg_tps': args.tg / max(total - ttft, 1e-9)}

warm = [one(i) for i in range(args.warmup_runs)]
vals = [one(i) for i in range(args.runs)]
pp = [v['pp_tps'] for v in vals]
tg = [v['tg_tps'] for v in vals]
tt = [v['ttfr_s'] * 1000 for v in vals]
res = {
    'version': 'codex-token-id-stream-v1',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime()),
    'latency_mode': 'stream_ttfr',
    'model': args.model,
    'request_format': 'openai_completions_prompt_token_ids_stream_true',
    'disable_spec_decode': args.disable_spec_decode,
    'warmup': warm,
    'benchmarks': [{
        'concurrency': 1,
        'prompt_size': args.pp,
        'response_size': args.tg,
        'pp_throughput': {'mean': statistics.mean(pp), 'std': statistics.pstdev(pp), 'values': pp},
        'tg_throughput': {'mean': statistics.mean(tg), 'std': statistics.pstdev(tg), 'values': tg},
        'ttfr': {'mean': statistics.mean(tt), 'std': statistics.pstdev(tt), 'values': tt},
        'raw_runs': vals,
    }],
}
Path(args.out).write_text(json.dumps(res, indent=2))
print(json.dumps({'out': args.out, 'pp_mean': res['benchmarks'][0]['pp_throughput']['mean'], 'pp_values': pp, 'ttfr_ms': tt}, indent=2))
