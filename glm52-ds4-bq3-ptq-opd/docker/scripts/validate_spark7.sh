#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-banana_smasher-dsv4-mixed-tier:sm121}
PACK=${2:?usage: validate_spark7.sh IMAGE PACK OUTPUT_DIR}
OUT=${3:?usage: validate_spark7.sh IMAGE PACK OUTPUT_DIR}
PACK=$(cd "$PACK" && pwd)
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd)
[[ -f "$PACK/MANIFEST.json" ]] || { echo "missing pack manifest" >&2; exit 2; }

image_id=$(sudo -n docker image inspect "$IMAGE" --format '{{.Id}}')
runtime_compiler_absence=$(sudo -n docker run --rm --entrypoint /bin/sh "$IMAGE" -c '
  set -eu
  for compiler in cc gcc g++ clang; do
    if command -v "$compiler" >/dev/null 2>&1; then
      echo "unexpected runtime compiler: $compiler" >&2
      exit 1
    fi
  done
  echo PASS
')
[[ "$runtime_compiler_absence" == "PASS" ]] || exit 2

cleanup() {
  for n in banana_smasher-cold-1 banana_smasher-cold-2; do
    sudo -n docker rm -f "$n" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for run in 1 2; do
  name="banana_smasher-cold-$run"
  run_out="$OUT/run-$run"
  sudo -n rm -rf "$run_out"
  mkdir -p "$run_out"
  chmod 0777 "$run_out"
  sudo -n docker rm -f "$name" >/dev/null 2>&1 || true
  cid=$(sudo -n docker run -d --name "$name" --gpus all \
    --ulimit core=0 \
    -e BANANA_SMASHER_START_TIMEOUT=60 \
    -e BANANA_SMASHER_PACK_HASH_WORKERS=32 \
    -e BANANA_SMASHER_DISTRIBUTED_BACKEND=gloo \
    -e CUDA_MODULE_LOADING=LAZY \
    -e CUDA_LAUNCH_BLOCKING=1 \
    -e PYTHONFAULTHANDLER=1 \
    -v "$PACK:/model:ro" \
    -v "$run_out:/run/banana_smasher" \
    -p 8000:8000 \
    "$IMAGE" /model)

  deadline=$((SECONDS + 900))
  registration_deadline=$((SECONDS + 30))
  while [[ ! -s "$run_out/receipts/STARTUP_SMOKE.json" ]]; do
    state=$(sudo -n docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || true)
    if [[ -z "$state" ]]; then
      (( SECONDS < registration_deadline )) || {
        echo "container registration timeout: $cid" >&2
        exit 4
      }
      sleep 1
      continue
    fi
    if [[ "$state" != running ]]; then
      sudo -n docker logs "$cid" >&2 || true
      exit 4
    fi
    (( SECONDS < deadline )) || {
      sudo -n docker logs "$cid" >&2 || true
      echo "startup timeout" >&2
      exit 5
    }
    sleep 2
  done

  sudo -n docker exec "$name" /opt/vllm-runtime/bin/python \
    /opt/banana_smasher/runtime/run_prefill_ladder.py \
    --base http://127.0.0.1:8000 \
    --tokenizer-json /model/tokenizer/tokenizer.json \
    --out "/run/banana_smasher/ladder" \
    --targets 2048,8192 --rows 1 --decode-tokens 128 \
    --task public-validation --variant packaged-container

  sudo -n docker stop -t 30 "$name" >/dev/null
  sudo -n docker rm "$name" >/dev/null
  sudo -n chmod -R a+rX "$run_out"
done

cache_sha=$(sudo -n docker run --rm --entrypoint /opt/vllm-runtime/bin/python "$IMAGE" -c \
  'import hashlib;print(hashlib.sha256(open("/opt/banana_smasher/triton-cache/CACHE_MANIFEST.json","rb").read()).hexdigest())')
pack_sha=$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$PACK/MANIFEST.json")

python3 - "$OUT" "$image_id" "$cache_sha" "$pack_sha" "$runtime_compiler_absence" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
image_id, cache_sha, pack_sha, runtime_compiler_absence = sys.argv[2:]
if pack_sha != "58466bce399bc1536fc19de235445e5e697053ce57a1ee8c07a8c6b7b1684242":
    raise SystemExit(f"pack manifest identity mismatch: {pack_sha}")
if runtime_compiler_absence != "PASS":
    raise SystemExit(f"runtime compiler absence failed: {runtime_compiler_absence}")
sealed = {
    2048: {"ttft_seconds": 1.793210939002165, "prefill_tok_s": 1142.085381845603, "decode_tok_s": 16.95},
    8192: {"ttft_seconds": 3.7798590419988614, "prefill_tok_s": 2167.276585972347, "decode_tok_s": 16.95},
}
runs = []
for idx in (1, 2):
    root = out / f"run-{idx}"
    smoke = json.loads((root / "receipts/STARTUP_SMOKE.json").read_text())
    pack_validation = json.loads((root / "receipts/PACK_VALIDATION.json").read_text())
    ladder = json.loads((root / "ladder/MIXED_PREFILL_LADDER_RESULT.json").read_text())
    rows = {int(row["prompt_tokens"]): row for row in ladder["rows"]}
    if set(rows) != {2048, 8192}:
        raise SystemExit(f"run {idx}: wrong ladder targets {sorted(rows)}")
    if smoke["bind_seconds"] >= 60:
        raise SystemExit(f"run {idx}: /v1/models bind deadline missed: {smoke}")
    if smoke["first_token_seconds_from_container_start"] >= 60:
        raise SystemExit(f"run {idx}: first token deadline missed: {smoke}")
    if smoke["resident_product_bytes"] != 101_346_700_411:
        raise SystemExit(f"run {idx}: resident bytes mismatch")
    envelope = pack_validation.get("resident_envelope", {})
    pack_gates = {
        "status_pass": pack_validation.get("status") == "PASS",
        "manifest_sha_exact": pack_validation.get("manifest_sha256") == "58466bce399bc1536fc19de235445e5e697053ce57a1ee8c07a8c6b7b1684242",
        "bytes_exact": envelope.get("bytes") == 101_346_700_411,
        "files_exact": envelope.get("files") == 1_645,
        "pack_inventory_exact": envelope.get("inventory_sha256") == "38a1b3eefaef21b6bc3368be2d4e6480a3c30d64a1e19d5239a9ac89b87233f9",
        "sealed_source_inventory_exact": envelope.get("sealed_source_inventory_sha256") == "cb00fc4e783ab97018bbe0642556820596a7846816fb0bcc55bd9f27b223b3bd",
    }
    if not all(pack_gates.values()):
        raise SystemExit(f"run {idx}: fail-closed pack gates failed: {pack_gates}")
    runs.append({
        "run": idx,
        "pack_validation": pack_gates,
        "startup": {
            "bind_seconds": smoke["bind_seconds"],
            "warmup_smoke_seconds_after_bind": (
                smoke["smoke_response_seconds_from_container_start"] - smoke["bind_seconds"]
            ),
            **{key: smoke[key] for key in (
                "first_token_seconds_from_container_start",
                "smoke_response_seconds_from_container_start",
                "smoke_request_seconds",
                "prefill_tok_s",
                "decode_tok_s",
                "resident_product_bytes",
            )},
        },
        "prefill": {
            str(target): {
                "ttft_seconds": rows[target]["client_ttft_seconds_median"],
                "prefill_tok_s": rows[target]["prefill_tok_s_median"],
                "decode_tok_s": rows[target]["decode_tok_s_median"],
            }
            for target in (2048, 8192)
        },
    })

comparisons = {}
for target in (2048, 8192):
    cross_restart = abs(
        runs[0]["prefill"][str(target)]["prefill_tok_s"]
        - runs[1]["prefill"][str(target)]["prefill_tok_s"]
    ) / max(
        abs(runs[0]["prefill"][str(target)]["prefill_tok_s"]),
        abs(runs[1]["prefill"][str(target)]["prefill_tok_s"]),
    )
    by_run = []
    for run in runs:
        actual = run["prefill"][str(target)]
        relative_errors = {
            "ttft": abs(actual["ttft_seconds"] - sealed[target]["ttft_seconds"]) / sealed[target]["ttft_seconds"],
            "prefill": abs(actual["prefill_tok_s"] - sealed[target]["prefill_tok_s"]) / sealed[target]["prefill_tok_s"],
            "decode": abs(actual["decode_tok_s"] - sealed[target]["decode_tok_s"]) / sealed[target]["decode_tok_s"],
        }
        by_run.append({
            "run": run["run"],
            "relative_errors": relative_errors,
            "within_20_percent": all(value <= 0.20 for value in relative_errors.values()),
        })
    comparisons[str(target)] = {
        "sealed_reference": sealed[target],
        "cross_restart_prefill_relative_drift": cross_restart,
        "cross_restart_within_20_percent": cross_restart <= 0.20,
        "runs": by_run,
    }
if not all(
    row["cross_restart_within_20_percent"]
    and all(run["within_20_percent"] for run in row["runs"])
    for row in comparisons.values()
):
    raise SystemExit(f"sealed throughput/TTFT gate failed: {comparisons}")

receipt = {
    "schema": "banana_smasher-deploy-validation-v1",
    "status": "PASS",
    "validation_host": "single-sm121-spark",
    "container_cold_starts": 2,
    "image_id": image_id,
    "runtime_compiler_absence": runtime_compiler_absence,
    "kernel_cache_manifest_sha256": cache_sha,
    "pack_manifest_sha256": pack_sha,
    "runs": runs,
    "acceptance_vs_sealed": comparisons,
    "quality_claim": False,
}
(out / "deploy_validation.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY
