#!/usr/bin/env bash
# cache_reaper_aggressive.sh — drop ALL caches (=3: pagecache+dentries+inodes) every 1s while
# weights stream over NFS, to keep the 119GiB worker nodes from accumulating NFS read-ahead /
# slab that squeezes the GLM-5.2 indexer out-of-pool finalize. Self-stops when :8000 binds.
# Replaces the gentle =1/3s reaper for the OS-memory-axis 5.2 attempt.
set -u
for _ in $(seq 1 900); do
  if ss -ltn 2>/dev/null | grep -q ':8000 '; then exit 0; fi
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 1
done
