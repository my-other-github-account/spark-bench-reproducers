#!/usr/bin/env bash
set -euo pipefail
MISSION=${SPARK_HOME}/missions/P530_PREFILL_t_099a5835_s8
RUN="$MISSION/run"
OUT="$MISSION/receipts/final_gate"
LOG="$MISSION/logs/final_gate.log"
PIDFILE="$RUN/final_gate.pid"
if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "refuse duplicate live pid=$(cat "$PIDFILE")" >&2
  exit 2
fi
curl -fsS http://127.0.0.1:8130/health > "$MISSION/receipts/HEALTH.before_final_gate.json"
setsid nohup env PYTHONPATH=${SPARK_HOME}/venvs/vllm/lib/python3.12/site-packages /usr/bin/python3 -S "$MISSION/code/run_prefill_ladder.py" \
  --base http://127.0.0.1:8130 \
  --tokenizer-json ${SPARK_HOME}/models/hf/DeepSeek-V4-Flash/tokenizer.json \
  --out "$OUT" --targets 2048,8192 --rows 3 --decode-tokens 128 \
  --task task-redacted --variant rung1_dequant_dense --timeout 3600 \
  >"$LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
ps -o pgid= -p "$pid" | tr -d ' ' > "$RUN/final_gate.pgid"
ps -o sid= -p "$pid" | tr -d ' ' > "$RUN/final_gate.sid"
printf '{"task":"task-redacted","host":"spark-8","phase":"final_gate_detached","pid":%s,"log":"%s"}\n' "$pid" "$LOG" > "$MISSION/PROGRESS.json"
printf 'pid=%s pgid=%s sid=%s log=%s\n' "$pid" "$(cat "$RUN/final_gate.pgid")" "$(cat "$RUN/final_gate.sid")" "$LOG"
