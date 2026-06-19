#!/usr/bin/env bash
# cache_reaper.sh — drop CLEAN page cache every few seconds while weights load serially.
# This preserves free physical-RAM margin through the NVFP4 post-load finalize allocation
# (the non-swappable GPU-pool chunk that OOMs a worker on GB10 unified memory). It self-stops
# when :8000 binds. Started in the background by launch_node.sh; runs on the host.
set -u
for _ in $(seq 1 600); do
  if ss -ltn 2>/dev/null | grep -q ':8000 '; then exit 0; fi
  sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 3
done
