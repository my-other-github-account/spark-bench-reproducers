#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/missions/SERVED_AB"
exec python3 "$HOME/missions/SERVED_AB/run_s8_tps.py"
