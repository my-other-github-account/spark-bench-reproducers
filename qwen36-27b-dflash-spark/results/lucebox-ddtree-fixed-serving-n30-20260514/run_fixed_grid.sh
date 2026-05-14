#!/usr/bin/env bash
set -euo pipefail
DIR="${1:?receipt dir}"
cd "$DIR"
LOG="$DIR/runner.log"
exec > >(tee -a "$LOG") 2>&1

echo "[runner] start $(date -Is) dir=$DIR host=$(hostname)"
echo "[runner] unpatched llama-benchy path: $(command -v /home/user/venvs/vllm/bin/llama-benchy)"
/home/user/venvs/vllm/bin/python - <<'PY'
import inspect, llama_benchy.client as c
p=inspect.getsourcefile(c)
print('[runner] llama_benchy.client', p)
PY
# Stop interactive Hermes model service to avoid GPU contention; leave watchdogs paused.
systemctl --user stop qwen36-dflash-hermes-model.service || true
systemctl --user stop hermes-gateway-qwen36-spark3.service || true
sleep 2

cells=(dflash-sherlock-thinkON dflash-sherlock-thinkOFF dflash-codegen-thinkON dflash-codegen-thinkOFF)
for tag in "${cells[@]}"; do
  port=$(grep -oE -- '--port [0-9]+' "${tag}_server_cmd.sh" | awk '{print $2}')
  echo "[runner] === cell $tag port=$port start $(date -Is) ==="
  pkill -f "server_${tag}.py" || true
  pkill -f "127.0.0.1 --port ${port}" || true
  rm -f "${tag}.json" "${tag}_bench.log" "${tag}_server.log"
  bash "${tag}_server_cmd.sh" > "${tag}_server.log" 2>&1 &
  srv_pid=$!
  echo "$srv_pid" > "${tag}_server.pid"
  deadline=$((SECONDS+180))
  until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "[runner] server died for $tag"
      tail -200 "${tag}_server.log" || true
      exit 1
    fi
    if (( SECONDS > deadline )); then
      echo "[runner] timeout waiting for server $tag"
      tail -200 "${tag}_server.log" || true
      exit 1
    fi
    sleep 1
  done
  echo "[runner] server ready $tag $(date -Is)"
  # Data-plane smoke: confirm streamed content chunks include token_ids and no fallback-triggering content chunk lacks ids.
  /home/user/venvs/vllm/bin/python - "$port" "$DIR/${tag}_tokenid_smoke.json" <<'PY'
import json, sys, urllib.request
port=sys.argv[1]; out=sys.argv[2]
payload={"model":"luce-dflash","messages":[{"role":"user","content":"Reply with a short sentence about token ids."}],"max_tokens":16,"stream":True,"stream_options":{"include_usage":True}}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
content_chunks=noids=ids=0; usage=None; examples=[]
with urllib.request.urlopen(req, timeout=60) as r:
    for raw in r:
        line=raw.decode('utf-8','replace').strip()
        if not line.startswith('data: '): continue
        data=line[6:]
        if data=='[DONE]': break
        obj=json.loads(data)
        if obj.get('usage'): usage=obj['usage']
        for ch in obj.get('choices',[]):
            d=ch.get('delta') or {}
            txt=d.get('content') or d.get('reasoning_content') or ''
            if txt:
                content_chunks += 1
                tid=ch.get('token_ids')
                if not tid:
                    noids += 1
                    if len(examples)<3: examples.append(txt)
                else:
                    ids += len(tid)
res={"content_chunks":content_chunks,"noids":noids,"total_ids":ids,"usage":usage,"examples_missing":examples}
open(out,'w').write(json.dumps(res,indent=2))
print('[smoke]', json.dumps(res))
if noids:
    raise SystemExit('stream content chunk missing token_ids')
PY
  echo "[runner] bench $tag $(date -Is)"
  set +e
  bash "${tag}_bench_cmd.sh" > "${tag}_bench.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[runner] bench failed rc=$rc tag=$tag"
    tail -200 "${tag}_bench.log" || true
    kill -INT "$srv_pid" || true; sleep 3; kill "$srv_pid" || true
    exit $rc
  fi
  if grep -q 'No token_ids in response, using local tokenization' "${tag}_bench.log"; then
    echo "[runner] INVALID: llama-benchy fallback detected for $tag"
    exit 2
  fi
  kill -INT "$srv_pid" || true
  sleep 3
  kill "$srv_pid" 2>/dev/null || true
  wait "$srv_pid" 2>/dev/null || true
  echo "[runner] === cell $tag done $(date -Is) ==="
done

/home/user/venvs/vllm/bin/python "$DIR/summarize_fixed_grid.py" "$DIR"
echo "[runner] done $(date -Is)"
# Restart interactive Hermes services after benchmark grid.
systemctl --user start qwen36-dflash-hermes-model.service || true
sleep 2
systemctl --user start hermes-gateway-qwen36-spark3.service || true
