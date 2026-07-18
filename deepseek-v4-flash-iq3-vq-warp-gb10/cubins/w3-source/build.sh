#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/out}"
CUBIT_REF="${CUBIT_REF:-c139df8b34f1dcab607f8ccb685fdea948f3ae4d}"
WORK="${WORK:-$ROOT/.build}"
mkdir -p "$OUT" "$WORK"
command -v python3 >/dev/null
command -v sha256sum >/dev/null

if [[ -z "${CUBIT_BIN:-}" ]]; then
  command -v git >/dev/null
  command -v cargo >/dev/null
  if [[ ! -d "$WORK/cubit/.git" ]]; then
    git clone https://github.com/kacper-daftcode/cubit.git "$WORK/cubit"
  fi
  git -C "$WORK/cubit" fetch --depth 1 origin "$CUBIT_REF"
  git -C "$WORK/cubit" checkout --detach "$CUBIT_REF"
  cargo build --release --manifest-path "$WORK/cubit/Cargo.toml"
  CUBIT_BIN="$WORK/cubit/target/release/cubit"
fi

for variant in base mc4 mc4afrag; do
  case "$variant" in
    base) mc=1; afrag=0; runtime_name="moe_w3_mm" ;;
    mc4) mc=4; afrag=0; runtime_name="moe_w3_mm_mc4" ;;
    mc4afrag) mc=4; afrag=1; runtime_name="moe_w3_mm_mc4afrag" ;;
  esac
  for k in 2048 4096; do
    sass="$OUT/moe_w3_mm_e43_${variant}_k${k}.sass"
    MOEW3_MC="$mc" MOEW3_AFRAG="$afrag" \
      MOEW3_LUT_LO=0xb6bfc6cd MOEW3_LUT_HI=0x4d463c21 \
      python3 "$ROOT/gen_moe_w3.py" "$sass" "$k"
    cmp "$sass" "$ROOT/reference-sass/$(basename "$sass")"
    if [[ -d "$WORK/cubit" ]]; then
      (cd "$WORK/cubit" && "$CUBIT_BIN" asm "$sass" \
        -o "$OUT/${runtime_name}_k${k}.cubin" \
        --kernel moe_w3_mm --mercury-stub "$ROOT/qmma_e4m3.merc.stub")
    else
      "$CUBIT_BIN" asm "$sass" -o "$OUT/${runtime_name}_k${k}.cubin" \
        --kernel moe_w3_mm --mercury-stub "$ROOT/qmma_e4m3.merc.stub"
    fi
  done
done
sha256sum "$OUT"/*.cubin
