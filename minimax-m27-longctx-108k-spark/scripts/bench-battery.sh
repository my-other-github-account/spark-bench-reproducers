#!/usr/bin/env bash
# bench-battery.sh — Five-phase comprehensive validation battery.
# Submission-grade: tells you whether the recipe is back-to-back stable AND
# meets headline throughput AND has a working prefix cache AND no slow leak.
#
# Must be run from the HOST (not inside the container). It will exec into the
# running server container (default name: mm-srv-longctx-108k) for each cell.
#
# Phases (single boot, ~75 min total wall):
#   A — Stability     : 5× back-to-back unique 100K prompts (n=1 each)
#                       PASS = R5/R1 decode ≥ 0.9 (server eval-time as truth)
#   B — Prefix cache  : same 100K prompt sent 3× back-to-back via curl
#                       PASS = send 2 wall ≤ 5% of send 1 wall (≥20× speedup)
#   C — d=0 post-stress: 3× d=0 tg128 immediately after Phase B
#   D — d=0 leak check : 60s settle, then 3× more d=0 tg128 — compare to C
#                       PASS = D mean within ±5% of C mean (no slow leak)
#   E — Headliner grid : 6 cells of d=0 pp=128 tg=128 n=5 each
#                       (the real submission number)
#
# Usage:
#   ./scripts/bench-battery.sh [container-name] [results-dir]
# e.g.
#   ./scripts/bench-battery.sh mm-srv-longctx-108k ./results
set -uo pipefail

NAME="${1:-mm-srv-longctx-108k}"
RESULTS_HOST_DIR="${2:-$(pwd)/results}"
PORT=18080

mkdir -p "$RESULTS_HOST_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$RESULTS_HOST_DIR/battery-${TS}.log"
SUMMARY="$RESULTS_HOST_DIR/battery-${TS}.summary.json"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
log_mem() { free -h | head -3 | tee -a "$LOG"; }
alive() {
  if ! sudo docker ps -q -f name="$NAME" | grep -q .; then
    log "!!! container '$NAME' DIED at: $1"
    sudo docker ps -a --filter "name=$NAME" --format '{{.Names}} {{.Status}}' | tee -a "$LOG"
    sudo dmesg -T 2>/dev/null | tail -30 | grep -iE "oom|kill|memory" | tail -8 | tee -a "$LOG"
    return 1
  fi
  return 0
}

# Verify server is up
if ! curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  log "ERROR: no llama-server on 127.0.0.1:${PORT}. Boot first via launch_server.sh"
  exit 1
fi

log "=== BATTERY START ($NAME) ==="
log "Results: $RESULTS_HOST_DIR"
log_mem

# Inner: run a tg cell inside the container. The container must already have
# $RESULTS_HOST_DIR bind-mounted at /repro/results (this is the convention
# from `docker run -v $(pwd)/results:/repro/results`).
exec_tg() {
  local depth="$1" runs="$2" tg="${3:-128}" tag="$4"
  local out_in="/repro/results/${tag}.json"
  sudo docker exec -i \
    -e OUT="$out_in" -e PP=128 \
    "$NAME" \
    bash /repro/scripts/bench-tg.sh "$depth" "$runs" "$tg" 2>&1 | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

# ===== PHASE A: 5× back-to-back unique 100K =====
log ""
log "=== PHASE A: 5× back-to-back unique 100K (stability) ==="
PASSED_A=0
DEPTH=100100
for i in 1 2 3 4 5; do
  log ""
  log "--- [Phase-A run $i/5] $(date -Is) ---"
  exec_tg "$DEPTH" 1 128 "phaseA-r${i}-d${DEPTH}" || true
  alive "Phase-A-r$i" || break
  PASSED_A=$((PASSED_A+1))
  log_mem
done
log ""
log "Phase A: $PASSED_A/5 runs survived"

# ===== PHASE B: prefix-cache validation =====
log ""
log "=== PHASE B: same-prompt 3× prefix-cache test ==="
PROMPT_FILE="$RESULTS_HOST_DIR/.prefix-test-prompt.txt"
if [[ ! -s "$PROMPT_FILE" ]]; then
  log "Fetching prefix-test prompt (sherlock, 400KB)..."
  curl -fsS https://www.gutenberg.org/files/1661/1661-0.txt | head -c 400000 > "$PROMPT_FILE"
fi
PAYLOAD="$RESULTS_HOST_DIR/.battery-payload.json"
jq -n --rawfile p "$PROMPT_FILE" '{model:"MiniMax-M2.7-UD-IQ4_XS", messages:[{role:"user", content:$p}], max_tokens:8, temperature:0}' > "$PAYLOAD"

PHASE_B_TIMES=()
for i in 1 2 3; do
  log ""
  log "--- [Phase-B send $i/3] $(date -Is) ---"
  T0=$(date +%s.%N)
  curl -fsS -m 1500 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$PAYLOAD" \
    -o "$RESULTS_HOST_DIR/.phaseB-r${i}.json" 2>>"$LOG" || log "  send $i failed"
  T1=$(date +%s.%N)
  ELAPSED=$(echo "$T1 - $T0" | bc)
  log "  send $i full roundtrip: ${ELAPSED}s"
  PHASE_B_TIMES+=("$ELAPSED")
  alive "Phase-B-send$i" || break
  sleep 5
done
log "Phase B times: ${PHASE_B_TIMES[*]}"

# ===== PHASE C: d=0 post-stress =====
log ""
log "=== PHASE C: d=0 tg128 (3 runs, post-stress) ==="
for i in 1 2 3; do
  log ""
  log "--- [Phase-C run $i/3] d=0 tg128 ---"
  exec_tg 0 1 128 "phaseC-r${i}-d0" || true
  alive "Phase-C-r$i" || break
done

# ===== PHASE D: d=0 leak check =====
log ""
log "=== PHASE D: 60s settle + 3× d=0 tg128 (leak check) ==="
sleep 60
log_mem
for i in 1 2 3; do
  log ""
  log "--- [Phase-D run $i/3] d=0 tg128 (post-settle) ---"
  exec_tg 0 1 128 "phaseD-r${i}-d0" || true
  alive "Phase-D-r$i" || break
done

# ===== PHASE E: headliner grid (d=0 n=5 × 1 cell × 1 mode for now) =====
log ""
log "=== PHASE E: headliner d=0 pp=128 tg=128 n=5 ==="
exec_tg 0 5 128 "phaseE-d0-tg128-n5" || true
alive "Phase-E" || true

# ===== Summary =====
log ""
log "=== BATTERY DONE ==="
log "Aggregating server eval-times..."
SERVER_LOG_TAIL=$(sudo docker logs "$NAME" 2>&1 | tail -2000)
echo "$SERVER_LOG_TAIL" > "$RESULTS_HOST_DIR/battery-${TS}.server-tail.log"

python3 - "$RESULTS_HOST_DIR" "$TS" "$PASSED_A" "${PHASE_B_TIMES[*]}" <<'PY' | tee -a "$LOG"
import glob, json, os, statistics as st, sys
results_dir, ts, passed_a, phase_b_times = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
phase_b = [float(x) for x in phase_b_times.split() if x]

def load(prefix):
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")))
    out = []
    for f in files:
        try:
            d = json.load(open(f))
            b = d["benchmarks"][0]
            tg = b["tg_throughput"]["values"]
            pp = b["pp_throughput"]
            ttfr = b.get("e2e_ttft", b.get("ttfr"))["values"]
            out.append({
                "file": os.path.basename(f),
                "tg_median": st.median(tg) if tg else None,
                "tg_mean": b["tg_throughput"]["mean"],
                "tg_n": len(tg),
                "pp_mean": pp["mean"],
                "ttfr_mean_ms": st.mean(ttfr) if ttfr else None,
                "depth": b.get("depth", 0),
            })
        except Exception as e:
            out.append({"file": os.path.basename(f), "error": str(e)})
    return out

phase_a = load("phaseA-")
phase_c = load("phaseC-")
phase_d = load("phaseD-")
phase_e = load("phaseE-")

# Phase A: R5/R1 decode ratio
def cell_tg(rec):
    return rec.get("tg_median") or rec.get("tg_mean")

pa_tgs = [cell_tg(r) for r in phase_a if cell_tg(r)]
pa_ratio = (pa_tgs[-1] / pa_tgs[0]) if len(pa_tgs) >= 2 and pa_tgs[0] else None
pa_pass = (pa_ratio is not None and pa_ratio >= 0.9 and passed_a >= 5)

# Phase B: send2/send1 wall ratio
pb_speedup = (phase_b[0] / phase_b[1]) if len(phase_b) >= 2 and phase_b[1] > 0 else None
pb_pass = (pb_speedup is not None and pb_speedup >= 20)

# Phase C/D: tg/s mean comparison
pc_mean = st.mean([cell_tg(r) for r in phase_c if cell_tg(r)]) if phase_c else None
pd_mean = st.mean([cell_tg(r) for r in phase_d if cell_tg(r)]) if phase_d else None
pd_drift = (pd_mean / pc_mean - 1) * 100 if pc_mean and pd_mean else None
pcd_pass = (pd_drift is not None and abs(pd_drift) <= 5)

# Phase E: headliner
pe_tg = cell_tg(phase_e[0]) if phase_e else None
pe_pp = phase_e[0].get("pp_mean") if phase_e else None
pe_ttfr = phase_e[0].get("ttfr_mean_ms") if phase_e else None

summary = {
  "ts": ts, "container": "mm-srv-longctx-108k",
  "phase_a": {
    "passed": passed_a, "total": 5,
    "tg_per_run": pa_tgs,
    "r5_over_r1": pa_ratio,
    "result": "PASS" if pa_pass else "FAIL",
  },
  "phase_b": {
    "wall_seconds": phase_b,
    "send1_over_send2": pb_speedup,
    "result": "PASS" if pb_pass else "FAIL",
  },
  "phase_c_d_leak": {
    "phaseC_mean_tg": pc_mean, "phaseD_mean_tg": pd_mean,
    "drift_pct": pd_drift,
    "result": "PASS" if pcd_pass else "FAIL",
  },
  "phase_e_headline": {
    "depth": 0, "tg": 128, "pp": 128, "n": (phase_e[0].get("tg_n") if phase_e else None),
    "tg_median_warm": pe_tg, "pp_mean": pe_pp, "ttfr_mean_ms": pe_ttfr,
  },
}
out_path = os.path.join(results_dir, f"battery-{ts}.summary.json")
json.dump(summary, open(out_path, "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\nSummary: {out_path}")
PY

log ""
log "Summary JSON: $SUMMARY"
log "Log: $LOG"
log "=== END ==="
