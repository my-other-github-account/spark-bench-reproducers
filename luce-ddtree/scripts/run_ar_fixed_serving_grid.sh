#!/usr/bin/env bash
set -euo pipefail
BASE=/home/user/work/dflash-lucebox-gb10-spark3
SRC=$BASE/llama-benchy-fixed-serving-grid-20260514_164136
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$BASE/llama-benchy-ar-baseline-fixed-serving-$STAMP
ln -sfn "$OUT" "$BASE/llama-benchy-ar-baseline-fixed-serving-latest"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/runner.log") 2>&1

PY=$BASE/venv/bin/python
BENCH=/home/user/venvs/vllm/bin/llama-benchy
MODEL=Qwen/Qwen3.6-27B
SERVED=luce-dflash
TARGET=$BASE/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_M.gguf
DRAFT=$BASE/lucebox-hub/dflash/models/draft
BIN=$BASE/lucebox-hub/dflash/build/test_dflash
CODE_URL=https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py
RUNS=30
PP=128
TG=128
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT

cat > "$OUT/README.md" <<EOF
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

Generated: $(date -Is) on $(hostname)
EOF

wait_ready() {
  local port="$1"
  for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

make_ar_server() {
  local src="$1" dst="$2"
  cp "$src" "$dst"
  "$PY" - "$dst" <<'PY'
from pathlib import Path
import sys, re
p=Path(sys.argv[1]); s=p.read_text()
# Patch qwen35 daemon command: remove speculative/DDTree flags for true autoregressive decode.
old='''        cmd = [bin_abs, str(target), str(draft), "--daemon",\n               "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",\n               f"--ddtree-temp={ddtree_temp}",\n               f"--max-ctx={max_ctx}",\n               f"--stream-fd={stream_fd_val}"]'''
new='''        cmd = [bin_abs, str(target), str(draft), "--daemon",\n               f"--max-ctx={max_ctx}",\n               f"--stream-fd={stream_fd_val}"]'''
if old not in s:
    raise SystemExit('AR patch target not found')
s=s.replace(old,new)
p.write_text(s)
PY
}

summarize_one() {
  local tag="$1" corpus="$2" think="$3" json_path="$4" bench_log="$5" server_log="$6"
  "$PY" - "$tag" "$corpus" "$think" "$json_path" "$bench_log" "$server_log" "$OUT/${tag}_summary.json" <<'PY'
import json,sys,statistics as st,re,pathlib
(tag,corpus,think,json_path,bench_log,server_log,out)=sys.argv[1:]
d=json.load(open(json_path))
bench=d['benchmarks'][0] if isinstance(d,dict) and 'benchmarks' in d else d[0]
vals=bench['tg_throughput']['values']
ttfr=bench.get('ttfr',{}).get('values',[])
pp=bench.get('pp_throughput',{})
blog=pathlib.Path(bench_log).read_text(errors='replace') if pathlib.Path(bench_log).exists() else ''
slog=pathlib.Path(server_log).read_text(errors='replace') if pathlib.Path(server_log).exists() else ''
counts=[]
for m in re.finditer(r'chat DONE .*?\bin=(\d+)\s+out=(\d+)\b', slog):
    inn,outc=map(int,m.groups())
    if outc==128 or inn>=120:
        counts.append(outc)
counts=counts[-len(vals):]
gen_counts=[int(x) for x in re.findall(r'\[dflash\] generated (\d+) tokens', slog)]
gen_counts=gen_counts[-len(vals):]
warm=vals[1:] if len(vals)>1 else vals
warm_ttfr=ttfr[1:] if len(ttfr)>1 else ttfr
summary={
 'tag':tag,'mode':'true-ar','corpus':corpus,'think':think,
 'runs':len(vals),'warm_n':len(warm),
 'prompt_size':bench.get('prompt_size'),'response_size':bench.get('response_size'),'concurrency':bench.get('concurrency'),
 'tg_values':vals,
 'tg_mean_all':st.mean(vals),'tg_median_all':st.median(vals),'tg_std_all':st.pstdev(vals) if len(vals)>1 else 0.0,
 'tg_mean_warm':st.mean(warm),'tg_median_warm':st.median(warm),'tg_std_warm':st.pstdev(warm) if len(warm)>1 else 0.0,
 'ttfr_median_warm_ms':(st.median(warm_ttfr) if warm_ttfr else None),
 'pp_mean':pp.get('mean'),
 'chat_done_counts_tail':counts,
 'dflash_generated_counts_tail':gen_counts,
 'fallback_detected':('No token_ids in response, using local tokenization' in blog),
 'eligible_standard': len(vals)==30 and bench.get('response_size')==128 and (bench.get('concurrency') in (1,None)) and not ('No token_ids in response, using local tokenization' in blog) and (not counts or all(c==128 for c in counts)) and (not gen_counts or all(c==128 for c in gen_counts)),
 'raw_json':json_path,'bench_log':bench_log,'server_log':server_log,
}
pathlib.Path(out).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2), flush=True)
PY
}

run_cell() {
  local corpus="$1" think="$2" port="$3"
  local tag="true-ar-${corpus}-${think}"
  local src="$SRC/server_dflash-${corpus}-${think}.py"
  local server_py="$OUT/server_${tag}.py"
  local server_log="$OUT/${tag}_server.log"
  local bench_log="$OUT/${tag}_bench.log"
  local json_path="$OUT/${tag}.json"
  local cmd_file="$OUT/${tag}_server_cmd.sh"
  local bench_cmd="$OUT/${tag}_bench_cmd.sh"
  cleanup || true
  make_ar_server "$src" "$server_py"
  echo "[$(date -Is)] START $tag port=$port"
  printf '%q ' PYTHONPATH="$BASE/lucebox-hub/dflash/scripts${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$server_py" --host 127.0.0.1 --port "$port" --target "$TARGET" --draft "$DRAFT" --bin "$BIN" --budget 18 --ddtree-temp 1.05 --max-ctx 1024 --ctk f16 --ctv f16 --fa-window 2048 --prefix-cache-slots 0 --prefill-cache-slots 0 --tokenizer "$MODEL" --verbose-daemon --ignore-eos-stop --ddtree-no-chain-seed > "$cmd_file"; echo >> "$cmd_file"
  PYTHONPATH="$BASE/lucebox-hub/dflash/scripts${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$server_py" --host 127.0.0.1 --port "$port" --target "$TARGET" --draft "$DRAFT" --bin "$BIN" --budget 18 --ddtree-temp 1.05 --max-ctx 1024 --ctk f16 --ctv f16 --fa-window 2048 --prefix-cache-slots 0 --prefill-cache-slots 0 --tokenizer "$MODEL" --verbose-daemon --ignore-eos-stop --ddtree-no-chain-seed > "$server_log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready "$port"; then echo "server failed to become ready for $tag"; tail -200 "$server_log"; return 1; fi
  args=("$BENCH" --base-url "http://127.0.0.1:$port/v1" --api-key dummy --model "$MODEL" --served-model-name "$SERVED" --tokenizer "$MODEL" --pp "$PP" --tg "$TG" --depth 0 --concurrency 1 --runs "$RUNS" --no-cache --no-adapt-prompt --latency-mode none --skip-coherence --save-result "$json_path" --format json)
  if [[ "$corpus" == codegen ]]; then args+=(--book-url "$CODE_URL"); fi
  printf '%q ' "${args[@]}" > "$bench_cmd"; echo >> "$bench_cmd"
  "${args[@]}" > "$bench_log" 2>&1
  summarize_one "$tag" "$corpus" "$think" "$json_path" "$bench_log" "$server_log" | tee "$OUT/${tag}_summary.log"
  cleanup || true
  sleep 3
}

# Stop stale benchmark-only processes from earlier failed/interactive runs.
pkill -f 'llama-benchy-best-n30-grid-v2-20260513_220310/server_dflash-sherlock-thinkOFF.py' || true
pkill -f 'test_dflash .*--max-ctx=139264.*--ddtree' || true
sleep 5

run_cell sherlock thinkON  8190
run_cell sherlock thinkOFF 8191
run_cell codegen  thinkON  8192
run_cell codegen  thinkOFF 8193

"$PY" - "$OUT" <<'PY' | tee "$OUT/AR_BASELINE_SUMMARY.md"
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
print('# True-AR baseline summary')
print('\nShape: standard llama-benchy `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30`. Warm metrics drop first pass.')
print('\n- True AR: server wrapper command removes `--fast-rollback --ddtree --ddtree-budget --ddtree-temp`; no llama-benchy patching.')
print('- Streaming token accounting: requires `choices[0].token_ids`; fallback must be false.')
print('\n| Row | eligible | median warm TG tok/s | mean warm | std warm | warm_n | fallback |')
print('|---|---:|---:|---:|---:|---:|---:|')
for tag in ['true-ar-sherlock-thinkON','true-ar-sherlock-thinkOFF','true-ar-codegen-thinkON','true-ar-codegen-thinkOFF']:
    p=out/(tag+'_summary.json')
    d=json.load(open(p))
    print(f"| {tag} | {d['eligible_standard']} | **{d['tg_median_warm']:.3f}** | {d['tg_mean_warm']:.3f} | {d['tg_std_warm']:.3f} | {d['warm_n']} | {d['fallback_detected']} |")
print('\n## Raw receipts')
for tag in ['true-ar-sherlock-thinkON','true-ar-sherlock-thinkOFF','true-ar-codegen-thinkON','true-ar-codegen-thinkOFF']:
    d=json.load(open(out/(tag+'_summary.json')))
    print(f"- `{tag}`: `{d['raw_json']}`; server `{d['server_log']}`; bench `{d['bench_log']}`; generated_tail={d['dflash_generated_counts_tail'][-5:]}")
PY

echo "Completed $(date -Is)" | tee -a "$OUT/README.md"
