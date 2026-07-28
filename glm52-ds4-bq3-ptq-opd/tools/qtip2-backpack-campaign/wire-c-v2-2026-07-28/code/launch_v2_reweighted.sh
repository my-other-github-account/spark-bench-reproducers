#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SOLVER_ROOT="$PACKAGE_ROOT/solver_root"
INPUTS="$SOLVER_ROOT/inputs/wire_c_v2"
PYTHON="${PYTHON:-$SOURCE_ROOT/python}"
OUT="$PACKAGE_ROOT/outputs/V2_REWEIGHTED_PRE_V3"
export P693_HINT_ASSIGNMENT="${P693_HINT_ASSIGNMENT:-$INPUTS/P891_V2_TERMINAL_ASSIGNMENT.json}"
EXTRA=()

if [[ "${1:-}" == "--dry-run" ]]; then
  EXTRA+=("--dry-run")
  OUT="$PACKAGE_ROOT/outputs/V2_REWEIGHTED_PRE_V3_DRY_RUN"
  shift
fi

if [[ ! -x "$PYTHON" ]]; then
  printf 'ERROR: interpreter is not executable: %s\n' "$PYTHON" >&2
  printf 'For the Mac metadata-only proof, use: PYTHON=/path/to/python ./launch_wire_c_v2.sh --dry-run\n' >&2
  exit 2
fi

mkdir -p "$OUT"
exec "$PYTHON" "$SOLVER_ROOT/code/solve_p924_reweighted.py" \
  --current-menu "$INPUTS/CURRENT_MENU_SNAPSHOT_REWEIGHTED.json" \
  --q2-anchor "$INPUTS/P880_QTIP2_ASSEALED_BALANCED64_V1.json" \
  --q3-anchor "$INPUTS/P819_QTIP3_UNIFORM_BALANCED64_P875F1.json" \
  --retrodiction "$INPUTS/WIRE_B_RETRODICTION.json" \
  --operator-baseline-override "$INPUTS/OPERATOR_BASELINE_OVERRIDE.json" \
  --corrected-grid-receipt "$INPUTS/CORRECTED_VERTICAL_GRID_MANIFEST.json" \
  --v2-config "$INPUTS/WIRE_C_V2_REWEIGHTED_PRE_V3_CONFIG.json" \
  --first-feasible "$INPUTS/FIRST_FEASIBLE_PREVIEW.json" \
  --p887-receipt "$INPUTS/P887_QTIP3_LATE_L030_L042_BALANCED64.json" \
  --source-root "$INPUTS" \
  --out-dir "$OUT" \
  --time-limit-seconds 3600 \
  --threads 16 \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
