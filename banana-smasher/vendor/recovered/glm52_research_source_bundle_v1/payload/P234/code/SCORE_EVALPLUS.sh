#!/usr/bin/env bash
set -euo pipefail
umask 077
MISSION=${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8
CODE="$MISSION/code"; OUT="$MISSION/results/evalplus"; DATASET="$MISSION/assets/data/HumanEvalPlus-v0.1.10.jsonl"
python3 - <<'PY'
import json,pathlib
c=json.loads(pathlib.Path('${SPARK_HOME}/HOST_CLAIM.json').read_text());assert c.get('owner')=='task-redacted' and c.get('mission')=='${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8',c
s=json.loads(pathlib.Path('${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8/run/GENERATION_STATUS.json').read_text());assert s.get('state')=='sealed' and s.get('sealed')==164,s
PY
IMAGE_ID="$(sudo -n docker image inspect evalplus:26d6d00 --format '{{.Id}}')"
test "$IMAGE_ID" = "sha256:ce82d4f2e99754feb576991dec8d558096cbcb43644b53faf941324d77981c95"
mkdir -p "$OUT"
python3 "$CODE/prepare_evalplus.py" | tee "$OUT/prepare.log"
sudo -n docker run --rm --network none --cpus 8 --memory 16g --pids-limit 512 \
 -v "$OUT:/work" -v "$DATASET:/work/HumanEvalPlus-v0.1.10.jsonl:ro" \
 -v "$CODE/sanitize_evalplus.py:/sanitize_evalplus.py:ro" \
 -e HUMANEVAL_OVERRIDE_PATH=/work/HumanEvalPlus-v0.1.10.jsonl \
 evalplus:26d6d00 python /sanitize_evalplus.py | tee "$OUT/sanitize.log"
set +e
sudo -n docker run --rm --network none --cpus 16 --memory 32g --pids-limit 1024 \
 -v "$OUT:/work" -v "$DATASET:/work/HumanEvalPlus-v0.1.10.jsonl:ro" \
 -e HUMANEVAL_OVERRIDE_PATH=/work/HumanEvalPlus-v0.1.10.jsonl \
 evalplus:26d6d00 evalplus.evaluate humaneval --samples /work/samples.jsonl --parallel 16 --i-just-wanna-run --test-details >"$OUT/evalplus.log" 2>&1
RC=$?
set -e
if [[ $RC -ne 0 ]]; then tail -n 100 "$OUT/evalplus.log" >&2; exit "$RC"; fi
sudo -n chown -R "$(id -u):$(id -g)" "$OUT"
python3 "$CODE/seal_evalplus.py" | tee "$OUT/seal.log"
