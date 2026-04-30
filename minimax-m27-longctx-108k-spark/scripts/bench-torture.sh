#!/usr/bin/env bash
# bench-torture.sh — extended Phase A. Runs N back-to-back unique 100K prompts
# (default N=10) and reports decode rate per run + R_N/R_1 ratio. Use after
# bench-battery.sh passes to confirm stability holds beyond 5 runs.
#
# Usage (from host):
#   ./scripts/bench-torture.sh [container-name] [results-dir] [num-runs]
set -uo pipefail

NAME="${1:-mm-srv-longctx-108k}"
RESULTS_HOST_DIR="${2:-$(pwd)/results}"
N="${3:-10}"
PORT=18080
DEPTH=100100

mkdir -p "$RESULTS_HOST_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$RESULTS_HOST_DIR/torture-${TS}.log"
SUMMARY="$RESULTS_HOST_DIR/torture-${TS}.summary.json"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
log_mem() { free -h | head -3 | tee -a "$LOG"; }
alive() {
  if ! sudo docker ps -q -f name="$NAME" | grep -q .; then
    log "!!! container '$NAME' DIED at: $1"
    return 1
  fi
  return 0
}

if ! curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  log "ERROR: no llama-server on :${PORT}. Boot first."
  exit 1
fi

log "=== TORTURE START ($NAME) — $N back-to-back unique 100K runs ==="
log_mem

PASSED=0
for i in $(seq 1 "$N"); do
  log ""
  log "--- [Torture run $i/$N] $(date -Is) ---"
  sudo docker exec -i \
    -e OUT="/repro/results/torture-r${i}-d${DEPTH}.json" -e PP=128 \
    "$NAME" \
    bash /repro/scripts/bench-tg.sh "$DEPTH" 1 128 2>&1 | tee -a "$LOG"
  alive "Torture-r$i" || break
  PASSED=$((PASSED+1))
  log_mem
done

log ""
log "=== TORTURE DONE: $PASSED/$N runs survived ==="

python3 - "$RESULTS_HOST_DIR" "$TS" "$PASSED" "$N" <<'PY' | tee -a "$LOG"
import glob, json, os, statistics as st, sys
results_dir, ts, passed, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(glob.glob(os.path.join(results_dir, f"torture-r*-d100100.json")))
runs = []
for f in files:
    try:
        b = json.load(open(f))["benchmarks"][0]
        tg = b["tg_throughput"]["values"]
        pp = b["pp_throughput"]
        ttfr = b.get("e2e_ttft", b.get("ttfr"))["values"]
        runs.append({
          "file": os.path.basename(f),
          "tg_median": st.median(tg) if tg else None,
          "pp_mean": pp["mean"],
          "ttfr_mean_ms": st.mean(ttfr) if ttfr else None,
        })
    except Exception as e:
        runs.append({"file": os.path.basename(f), "error": str(e)})

tgs = [r.get("tg_median") for r in runs if r.get("tg_median")]
ratio = (tgs[-1] / tgs[0]) if len(tgs) >= 2 and tgs[0] else None
verdict = "PASS" if (passed >= n and ratio is not None and ratio >= 0.9) else "FAIL"
out = {
  "ts": ts, "passed": passed, "total": n,
  "tg_per_run": tgs,
  "r_last_over_r_first": ratio,
  "tg_min": min(tgs) if tgs else None,
  "tg_max": max(tgs) if tgs else None,
  "tg_median_overall": st.median(tgs) if tgs else None,
  "verdict": verdict,
  "per_run": runs,
}
json.dump(out, open(os.path.join(results_dir, f"torture-{ts}.summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
PY

log ""
log "Summary: $SUMMARY"
log "Log: $LOG"
