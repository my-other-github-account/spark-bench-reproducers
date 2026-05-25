#!/usr/bin/env bash
set -euo pipefail
: "${WORKDIR:=$PWD/work}"
ATLAS_URL="${ATLAS_URL:-https://github.com/Avarok-Cybersecurity/atlas.git}"
ATLAS_BASE="${ATLAS_BASE:-87b7bb3279c33b130cf98ced6f13d79d5cde013c}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
if [ ! -d atlas/.git ]; then
  git clone "$ATLAS_URL" atlas
fi
cd atlas
git fetch origin
# Hard reset is intentional: this script creates a clean repro checkout.
git reset --hard "$ATLAS_BASE"
git clean -xfd
git am ../../patches/0001-atlas-nvfp4-dflash-all-quant-pass-state.patch
cargo build --release -p spark-server --bin spark
printf 'Atlas patched HEAD: '; git rev-parse HEAD
printf 'Spark binary: '; realpath target/release/spark
