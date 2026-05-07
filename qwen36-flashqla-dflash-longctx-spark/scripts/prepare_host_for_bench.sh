#!/usr/bin/env bash
# Reset host-side memory state before the GB10 unified-memory benchmark.
set -euo pipefail

sudo docker rm -f vllm-prefill-flashqla-hkv 2>/dev/null || true
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo swapoff -a
sudo swapon -a

free -h
swapon --show || true
nvidia-smi --query-gpu=temperature.gpu,pstate,clocks.sm,utilization.gpu --format=csv,noheader,nounits
