#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_ROOT:=$HOME/models}"
mkdir -p "$MODEL_ROOT"
cat <<EOF
This recipe expects the already-licensed model assets at:
  target:  $MODEL_ROOT/spark6-Qwen3.5-27B-NVFP4
  drafter: $PWD/work/qwen35-dflash  (or set DRAFT_MODEL=...)

If you have Hugging Face access and know the exact repos for your mirror, download them with hf CLI, e.g.:
  hf download <target-repo> --local-dir "$MODEL_ROOT/spark6-Qwen3.5-27B-NVFP4"
  hf download <drafter-repo> --local-dir "$PWD/work/qwen35-dflash"

The original measured run used /home/banana_bae/models/spark6-Qwen3.5-27B-NVFP4 and
/home/banana_bae/atlas-dflash-spark1/drafters/qwen35-dflash on Spark-1.
EOF
