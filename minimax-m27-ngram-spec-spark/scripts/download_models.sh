#!/usr/bin/env bash
# Download MiniMax-M2.7-UD-IQ4_XS GGUF (101 GB) + tokenizer files into ./models.
# Run on the host BEFORE `docker run`; mount ~/models into the container.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
mkdir -p "$MODELS_DIR"

if ! command -v hf >/dev/null 2>&1; then
  echo "[download] installing huggingface_hub CLI..."
  pip3 install --user --quiet "huggingface_hub[cli]"
  export PATH="$HOME/.local/bin:$PATH"
fi

# 1. GGUF shards (101 GB)
GGUF_DIR="$MODELS_DIR/MiniMax-M2.7-GGUF"
if [[ ! -f "$GGUF_DIR/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00004-of-00004.gguf" ]]; then
  echo "[download] unsloth/MiniMax-M2.7-GGUF UD-IQ4_XS (~101 GB)..."
  hf download unsloth/MiniMax-M2.7-GGUF \
    --include "UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-*.gguf" \
    --local-dir "$GGUF_DIR"
else
  echo "[download] GGUF already present, skipping"
fi

# 2. Tokenizer (only the files llama-benchy needs)
TOK_DIR="$MODELS_DIR/MiniMax-M2.7-tokenizer"
if [[ ! -f "$TOK_DIR/tokenizer.json" ]]; then
  echo "[download] MiniMaxAI/MiniMax-M2.7 tokenizer files..."
  hf download MiniMaxAI/MiniMax-M2.7 \
    --include "tokenizer.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
    --local-dir "$TOK_DIR"
else
  echo "[download] tokenizer already present, skipping"
fi

echo "[download] done. $MODELS_DIR contents:"
du -sh "$GGUF_DIR" "$TOK_DIR"
