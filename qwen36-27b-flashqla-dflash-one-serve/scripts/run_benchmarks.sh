#!/usr/bin/env bash
# Run BOTH gates on one server session. Mirrors the successful 2026-05-11 12:25 PT run.
# Output: results/codex_qwen36_one_serve_repro_<TS>_{pp_N10,tg_4prompt,summary}.json
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
mkdir -p "$RESULTS_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-codex_qwen36_one_serve_repro_${TS}}"

SERVER_LOG="$RESULTS_DIR/${TAG}_server.log"
PP_JSON="$RESULTS_DIR/${TAG}_pp_N10.json"
TG_JSON="$RESULTS_DIR/${TAG}_tg_4prompt.json"
SUMMARY_JSON="$RESULTS_DIR/${TAG}_summary.json"

# --- Start server ---
echo "=== Starting one-serve composition ==="
"$REPO_ROOT/scripts/launch_one_serve.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  server PID=$SERVER_PID"
echo "  log: $SERVER_LOG"

cleanup() {
  echo "=== Stopping server PID $SERVER_PID ==="
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for /health
echo "=== Waiting for server /health ==="
for i in $(seq 1 600); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "  ready after ${i}s"
    break
  fi
  sleep 1
done

# Verify boot markers (FlashQLA + DFlash + GDN qkvz + cutlass)
echo "=== Boot markers ==="
grep -F "[flashqla-patch] active"        "$SERVER_LOG" || echo "  WARN: flashqla active marker missing"
grep -F "DFlashDraftModel"               "$SERVER_LOG" | head -1 || echo "  WARN: DFlash arch marker missing"
grep -F "auxiliary layers from speculative" "$SERVER_LOG" | head -1 || echo "  WARN: aux layers marker missing"
grep -iE "Cutlass.*NVFP4|NVFP4.*Cutlass" "$SERVER_LOG" | head -1 || echo "  WARN: cutlass NVFP4 marker missing"
grep -F "CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4" "$SERVER_LOG" | head -1 || echo "  WARN: codex GDN qkv/z marker missing"

# --- Gate 1: PP2048/TG32/C1 N=10 (spec decode disabled at request level) ---
echo ""
echo "=== Gate 1: PP2048/TG32/C1 N=10 ==="
# token_id_stream_bench.py disables spec at the request body level so the PP measurement
# is not contaminated by DFlash overhead — but the server itself still has spec wired.
python3 "$REPO_ROOT/benchmarks/token_id_stream_bench.py" \
  --host 127.0.0.1 --port 8000 \
  --model qwen36-27b-unsloth-one-serve \
  --pp 2048 --tg 32 --concurrency 1 --warmup 8 --runs 10 \
  --disable-spec-decode \
  --output "$PP_JSON"

# --- Gate 2: AEON 4-prompt TG (with spec decode active) ---
echo ""
echo "=== Gate 2: AEON natural 4-prompt TG ==="
for class in code reasoning dialogue prose; do
  python3 "$REPO_ROOT/benchmarks/aeon_bench_natural.py" \
    --host 127.0.0.1 --port 8000 \
    --model qwen36-27b-unsloth-one-serve \
    --prompt "$class" \
    --concurrencies 1 --requests-per-level 16 \
    --max-tokens 512 --warmup 3 \
    --output "$RESULTS_DIR/${TAG}_${class}.csv"
done

# --- Aggregate summary ---
echo ""
echo "=== Building summary ==="
python3 - <<PY
import json, csv, hashlib, pathlib, statistics

results = pathlib.Path("$RESULTS_DIR")
tag = "$TAG"
server_log = results / f"{tag}_server.log"

# PP
pp = json.load(open(results / f"{tag}_pp_N10.json"))
pp_mean = pp.get("pp_throughput", {}).get("mean") or statistics.mean(pp.get("pp_throughput", {}).get("values", [0]))

# Per-prompt TG
per_prompt = {}
for cls in ["code", "reasoning", "dialogue", "prose"]:
    csv_path = results / f"{tag}_{cls}.csv"
    with open(csv_path) as f:
        row = next(csv.DictReader(f))
        per_prompt[cls] = {
            "output_tps_aggregate": float(row["output_tps_aggregate"]),
            "median_tokens_per_req_tps": float(row["median_tokens_per_req_tps"]),
            "ttft_ms_p50": float(row["ttft_ms_p50"]),
            "tpot_ms_p50": float(row["tpot_ms_p50"]),
            "n_requests": int(row["n_requests"]),
            "total_output_tokens": int(row["total_output_tokens"]),
            "csv": str(csv_path),
        }

tg_avg = sum(p["output_tps_aggregate"] for p in per_prompt.values()) / 4.0

# Server log sha256
sha = hashlib.sha256(open(server_log, "rb").read()).hexdigest() if server_log.exists() else None

# Boot markers (re-grep)
log = server_log.read_text() if server_log.exists() else ""
markers = {
    "flashqla_patch_active": "[flashqla-patch] active" in log,
    "dflash_arch_resolved":  "DFlashDraftModel" in log,
    "auxiliary_layers":      "auxiliary layers from speculative" in log,
    "nvfp4_backend_cutlass": "Cutlass" in log and "NVFP4" in log,
    "gdn_qkvz_fp4_active":   "CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4" in log,
    "prefix_disabled":       "enable_prefix_caching=False" in log or "enable_prefix_caching': False" in log,
}

summary = {
    "schema": "codex_qwen36_one_serve_repro_v1",
    "timestamp": tag.split("_repro_")[-1],
    "body_model": "$REPO_ROOT (set MODEL_DIR via env)",
    "drafter":    "$REPO_ROOT (set DRAFTER_DIR via env)",
    "pp_mean": pp_mean,
    "tg_avg_output_tps_aggregate": tg_avg,
    "per_prompt": per_prompt,
    "boot_markers": markers,
    "gates": {
        "pp_ge_3000": pp_mean >= 3000.0,
        "tg_avg_ge_30": tg_avg >= 30.0,
        "both": pp_mean >= 3000.0 and tg_avg >= 30.0,
    },
    "server_log_sha256": sha,
    "server_log": str(server_log),
    "pp_receipt": str(results / f"{tag}_pp_N10.json"),
    "tg_csvs": {cls: str(results / f"{tag}_{cls}.csv") for cls in ["code","reasoning","dialogue","prose"]},
}
out = results / f"{tag}_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"Summary: {out}")
print(f"  PP mean = {pp_mean:.2f}  (gate 3000)")
print(f"  TG avg  = {tg_avg:.2f}  (gate 30)")
print(f"  BOTH PASS: {summary['gates']['both']}")
PY
