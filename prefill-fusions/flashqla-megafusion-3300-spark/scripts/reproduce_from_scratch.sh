#!/usr/bin/env bash
# Build base, apply alias+kpack2 patch, launch server, run the strict API N=30 bench, verify.
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
BASE_RECIPE_DIR="${BASE_RECIPE_DIR:-$REPO_ROOT/vllm-prefill-flashqla-hkv-spark}"
BASE_IMAGE="${BASE_IMAGE:-vllm-prefill-flashqla-hkv-spark:spark1-current}"
OUT_IMAGE="${OUT_IMAGE:-vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2}"
CONTAINER="${CONTAINER:-vllm-prefill-flashqla-alias-kpack2-api-n30}"
MODELS_DIR="${MODELS_DIR:-/home/user/models}"
RUN_QUALITY_CANARY="${RUN_QUALITY_CANARY:-1}"

if [ ! -d "$BASE_RECIPE_DIR" ]; then
  echo "ERROR: missing base recipe dir: $BASE_RECIPE_DIR" >&2
  echo "Run from a full spark-bench-reproducers clone." >&2
  exit 2
fi

if [ ! -s "$MODELS_DIR/AxionML-Qwen3.5-27B-NVFP4/config.json" ]; then
  echo "model not found under $MODELS_DIR; running scripts/download_model.sh"
  MODELS_DIR="$MODELS_DIR" bash "$THIS_DIR/scripts/download_model.sh"
fi

echo "[1/6] build base image: $BASE_IMAGE"
docker build -t "$BASE_IMAGE" "$BASE_RECIPE_DIR"

echo "[2/6] patch/commit PASS image: $OUT_IMAGE"
BASE_IMAGE="$BASE_IMAGE" OUT_IMAGE="$OUT_IMAGE" bash "$THIS_DIR/scripts/apply_alias_kpack2_image_changes.sh"

echo "[3/6] launch PASS server: $CONTAINER"
CONTAINER="$CONTAINER" IMAGE="$OUT_IMAGE" MODELS_DIR="$MODELS_DIR" REPRO_DIR="$THIS_DIR" bash "$THIS_DIR/scripts/launch_pass_server.sh"

echo "[4/6] wait for /v1/models"
CONTAINER="$CONTAINER" bash "$THIS_DIR/scripts/wait_for_server.sh"

echo "[5/6] run strict API-mode PP2048/TG32/C1/N=30 bench"
CONTAINER="$CONTAINER" OUT=/repro/results/result-repro-alias-kpack2-api-n30.json bash "$THIS_DIR/scripts/bench_reproduce_api_n30.sh"

echo "[6/6] verify artifact contract"
python3 "$THIS_DIR/scripts/verify_artifact.py" "$THIS_DIR/results/result-repro-alias-kpack2-api-n30.json"

if [ "$RUN_QUALITY_CANARY" = "1" ]; then
  python3 "$THIS_DIR/scripts/quality_canary.py" --base-url http://127.0.0.1:8000/v1 --model qwen35-27b-axionml-nvfp4
fi

echo "done"
echo "server container is still running: $CONTAINER"
echo "stop with: docker rm -f $CONTAINER"
