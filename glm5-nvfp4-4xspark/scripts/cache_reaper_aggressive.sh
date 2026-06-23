#!/usr/bin/env bash
# cache_reaper_aggressive.sh — drop ALL caches (=3: pagecache+dentries+inodes) every 1s while
# weights stream over NFS, to keep the 119GiB worker nodes from accumulating NFS read-ahead /
# slab that squeezes the GLM-5.2 indexer out-of-pool finalize. Self-stops when :8000 binds.
# Replaces the gentle =1/3s reaper for the OS-memory-axis 5.2 attempt.
set -u
# RUN #63 FIX: the old 900-iteration (15-min) cap EXITED the reaper BEFORE the slow thin-node
# NFS load finished (head load alone = 17 min; thin nodes finish later) — the exact bug RUN #62
# found (`glm52-reaper` went `inactive` mid-load → thin nodes lost protection → NFS-readahead
# wedge). Extend to 5400s (90 min) so it covers the full multi-node NFS load + barrier. Still
# self-stops the instant :8000 binds (serving started → no more reaping needed).
for _ in $(seq 1 5400); do
  if ss -ltn 2>/dev/null | grep -q ':8000 '; then exit 0; fi
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 1
done
