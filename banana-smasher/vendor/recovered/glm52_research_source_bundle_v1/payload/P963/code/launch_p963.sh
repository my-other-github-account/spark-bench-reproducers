#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPARK_HOME}/missions/P963_ACCEL_TRUE_C_t_d78a699d_s1
cd "$ROOT"
export P770_ROOT="$ROOT"
export P770_HOST=spark-1
export P885_ACTIVE_MANIFEST="$ROOT/inputs/WIRE_C_TRUE_C_ACTIVE_OVERLAY.json"
export P885_ACTIVE_ASSIGNMENT="$ROOT/inputs/ASSIGNMENT_WIRE_C_V2.json"
export P885_OVERLAY_ADAPTER="$ROOT/code/p963_true_c_overlay_adapter.py"
export P885_RUN_ID=P963_ACCEL_EXACT_P951_BALANCED64_V3_MB2
export P963_MB=2
exec ${SPARK_HOME}/humming_env/bin/python3 -u "$ROOT/code/p963_true_c_accel.py" --anchor qtip3 --measurement balanced64
