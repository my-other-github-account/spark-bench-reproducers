#!/usr/bin/env bash
# t_cf38c8c9 SPEC UPDATE — s8 chain: extract 94G arm then 96G arm.
# CPU/IO only; coexists with the UD-IQ GPU job. Restartable (per-layer skip).
set -uo pipefail
cd ~/missions/DS4_R6
PY=python3
for ARM in 94G 96G; do
  case $ARM in
    94G) MD5=808c863b867bfc6ed869af5472044ae1;;
    96G) MD5=596001afa9c4660f4fa72e26b174bba9;;
  esac
  OUT=planes_r6_${ARM,,}
  echo "=== $(date -Is) extract $ARM -> $OUT"
  $PY r6_extract_arm.py R6_MANIFEST_${ARM}.json "$MD5" "$OUT" \
      >> logs/extract_${ARM}.log 2>&1 \
      || { echo "EXTRACT_${ARM}_FAIL"; exit 1; }
  touch EXTRACT_${ARM}_DONE
  echo "=== $(date -Is) $ARM done"
done
echo CHAIN_2ARM_DONE
