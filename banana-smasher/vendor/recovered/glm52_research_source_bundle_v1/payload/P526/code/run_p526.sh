#!/usr/bin/env bash
set -euo pipefail
M=${SPARK_HOME}/missions/P526_QTIP_MBATCH_t_88eede57_s4
exec ${SPARK_HOME}/humming_env/bin/python -u "$M/code/bench_qtip_mbatched.py"
