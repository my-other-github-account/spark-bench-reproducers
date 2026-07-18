#!/usr/bin/env bash
set -euo pipefail
CLOCK_MHZ=${CLOCK_MHZ:-3003}
sudo nvidia-smi -lgc "$CLOCK_MHZ,$CLOCK_MHZ"
echo "Clock lock requested; verify under workload:"
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader
