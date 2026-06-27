#!/usr/bin/env bash
# purge_node.sh — run on EACH node BEFORE launching the serve container. Maximizes free unified RAM:
# the ~104.7 GiB weight load + FP4 finalize spike + cold first forward all compete for the same
# 128 GB LPDDR5X pool, and every reclaimed GiB is headroom the TP collective needs. Idempotent.
set -u
# 1) tear down our serve container + any OTHER container (the fleet is dedicated to this run)
docker rm -f vllm_node 2>/dev/null || true
for c in $(docker ps --format '{{.Names}}' 2>/dev/null | grep -vx 'vllm_node'); do
  echo "  purge: stopping other container '$c'"; docker stop "$c" 2>/dev/null || true
done
# 2) kill leftover host-side inference procs (patterns never match this script)
for pat in "vllm serve" "EngineCore" "Worker_TP" "pt_main_thread" "from multiprocessing.spawn"; do
  pkill -9 -f "$pat" 2>/dev/null || true
done
# 3) clear shared memory (stale TP shm broadcast blocks + orphaned SysV IPC)
rm -f /dev/shm/* 2>/dev/null || true; ipcrm -a 2>/dev/null || true
# 4) drop ALL reclaimable caches twice, then compact for big contiguous allocs
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sudo sh -c 'echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true
# 5) SWAP STAYS FULLY ON as a burst cushion (do NOT swapoff — removing it at the allocation burst
#    risks OOM). Just make the kernel willing to use it.
sudo swapon -a 2>/dev/null || true
sudo sh -c 'echo 100 > /proc/sys/vm/swappiness' 2>/dev/null || true
AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "  purge complete: avail=${AVAILM}MiB swappiness=$(cat /proc/sys/vm/swappiness 2>/dev/null) swapused=$(free -m | awk '/Swap:/{print $3}')MiB"
if [ "${AVAILM:-0}" -lt 100000 ]; then echo "  WARNING: <100G free after purge — reboot the node before launching"; fi
