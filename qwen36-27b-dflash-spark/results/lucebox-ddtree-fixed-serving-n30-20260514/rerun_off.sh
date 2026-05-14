#!/usr/bin/env bash
set -euo pipefail
DIR="${1:?receipt dir}"
cd "$DIR"
LOG="$DIR/rerun_off.log"
exec > >(tee -a "$LOG") 2>&1
cells=(dflash-sherlock-thinkOFF dflash-codegen-thinkOFF)
echo "[rerun-off] start $(date -Is)"
systemctl --user stop qwen36-dflash-hermes-model.service || true
systemctl --user stop hermes-gateway-qwen36-spark3.service || true
for tag in "${cells[@]}"; do
  port=$(grep -oE -- '--port [0-9]+' "${tag}_server_cmd.sh" | awk '{print $2}')
  echo "[rerun-off] === $tag port=$port ==="
  rm -f "${tag}.json" "${tag}_bench.log" "${tag}_server.log" "${tag}_tokenid_smoke.json"
  bash "${tag}_server_cmd.sh" > "${tag}_server.log" 2>&1 & srv_pid=$!
  deadline=$((SECONDS+180))
  until curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; do
    kill -0 "$srv_pid" 2>/dev/null || { echo server died; tail -100 "${tag}_server.log"; exit 1; }
    ((SECONDS<deadline)) || { echo timeout; tail -100 "${tag}_server.log"; exit 1; }
    sleep 1
  done
  /home/user/venvs/vllm/bin/python - "$port" "$DIR/${tag}_tokenid_smoke.json" <<'PY'
import json, sys, urllib.request
port=sys.argv[1]; out=sys.argv[2]
payload={"model":"luce-dflash","messages":[{"role":"user","content":"Reply with a short sentence about token ids."}],"max_tokens":16,"stream":True,"stream_options":{"include_usage":True}}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
content_chunks=noids=ids=0; usage=None
with urllib.request.urlopen(req, timeout=60) as r:
  for raw in r:
    line=raw.decode().strip()
    if not line.startswith('data: '): continue
    data=line[6:]
    if data=='[DONE]': break
    obj=json.loads(data)
    if obj.get('usage'): usage=obj['usage']
    for ch in obj.get('choices',[]):
      d=ch.get('delta') or {}; txt=d.get('content') or d.get('reasoning_content') or ''
      if txt:
        content_chunks+=1; tid=ch.get('token_ids')
        if not tid: noids+=1
        else: ids+=len(tid)
res={"content_chunks":content_chunks,"noids":noids,"total_ids":ids,"usage":usage}
open(out,'w').write(json.dumps(res,indent=2)); print('[smoke]',res)
assert noids==0
PY
  bash "${tag}_bench_cmd.sh" > "${tag}_bench.log" 2>&1
  if grep -q 'No token_ids in response, using local tokenization' "${tag}_bench.log"; then echo fallback; exit 2; fi
  kill -INT "$srv_pid" || true; sleep 3; kill "$srv_pid" 2>/dev/null || true; wait "$srv_pid" 2>/dev/null || true
  echo "[rerun-off] done $tag $(date -Is)"
done
/home/user/venvs/vllm/bin/python "$DIR/summarize_fixed_grid.py" "$DIR"
echo "[rerun-off] done all $(date -Is)"
systemctl --user start qwen36-dflash-hermes-model.service || true
sleep 2
systemctl --user start hermes-gateway-qwen36-spark3.service || true
