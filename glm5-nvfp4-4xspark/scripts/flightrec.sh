#!/usr/bin/env bash
# flightrec.sh — append 1s mem snapshots to a PERSISTENT CSV (~/) that survives a node wedge,
# and dump the in-container vllm serve-log tail to ~/glm_flightrec_serve.log each tick so the
# loader trace survives even when the container is later power-cycled away.
# Usage: bash flightrec.sh   (run via systemd-run --user --collect)
set -u
OUT="${OUT:-$HOME/glm_flightrec.csv}"
LOGOUT="${LOGOUT:-$HOME/glm_flightrec_serve.log}"
echo "epoch,avail_mib,buffcache_mib,vllm_log_lines" > "$OUT"
for _ in $(seq 1 900); do
  read -r avail bc <<EOF
$(free -m | awk '/Mem:/{print $7, $6}')
EOF
  ll=$(docker exec vllm_node sh -c 'wc -l < /tmp/vllm_serve.log' 2>/dev/null || echo 0)
  printf '%s,%s,%s,%s\n' "$(date +%s)" "${avail:-0}" "${bc:-0}" "${ll:-0}" >> "$OUT"
  # persist the serve-log tail (loader stage lines) outside the container
  docker exec vllm_node sh -c 'tail -40 /tmp/vllm_serve.log' > "$LOGOUT" 2>/dev/null || true
  sync "$OUT" 2>/dev/null || true
  if ss -ltn 2>/dev/null | grep -q ':8000 '; then echo "BOUND" >> "$OUT"; exit 0; fi
  sleep 1
done
