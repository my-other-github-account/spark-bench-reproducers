#!/usr/bin/env bash
set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-banana_smasher-serve:golden}"
REGISTRY="${REGISTRY:-localhost:5050}"
REMOTE_IMAGE="${REGISTRY%/}/banana_smasher-serve:golden"
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/receipts}"
mkdir -p "$OUT"

docker image inspect "$SOURCE_IMAGE" >/dev/null
curl -fsS "http://${REGISTRY}/v2/" >/dev/null
docker tag "$SOURCE_IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE" | tee "$OUT/LOCAL_REGISTRY_PUSH.log"
docker image inspect "$REMOTE_IMAGE" > "$OUT/LOCAL_REGISTRY_IMAGE_INSPECT.json"
python3 - "$OUT/LOCAL_REGISTRY_IMAGE_INSPECT.json" "$OUT/LOCAL_REGISTRY_RECEIPT.json" "$REMOTE_IMAGE" <<'PY'
import hashlib,json,sys,time
source,out,image=sys.argv[1:]
inspect=json.load(open(source))[0]
digests=inspect.get("RepoDigests") or []
if not digests:
    raise SystemExit("registry push produced no RepoDigest")
receipt={
 "schema":"banana_smasher-golden-local-registry-v1","status":"PASS","created_unix":time.time(),
 "image":image,"image_id":inspect["Id"],"repo_digests":digests,"size":inspect["Size"],
 "inspect_sha256":hashlib.sha256(open(source,"rb").read()).hexdigest(),
 "truth_label":"PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
}
open(out,"w").write(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
PY
