#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-build-8}"
SOURCE_IMAGE="${SOURCE_IMAGE:-genesis-serve:golden}"
PORT="${PORT:-5050}"
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/receipts}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PORT must be an integer in 1..65535" >&2
  exit 2
fi
REMOTE_REPO="localhost:${PORT}/genesis-serve"
REMOTE_IMAGE="${REMOTE_REPO}:golden"
mkdir -p "$OUT"
curl -fsS "http://127.0.0.1:${PORT}/v2/" >/dev/null

# Pass all dynamic values as positional argv to a fixed stdin script. No value is
# interpolated into remote shell syntax.
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -R "127.0.0.1:${PORT}:127.0.0.1:${PORT}" "$HOST" \
  sh -s -- "$SOURCE_IMAGE" "$REMOTE_IMAGE" "$REMOTE_REPO" \
  > "$OUT/SSH_TUNNEL_REGISTRY_PUSH_AND_INSPECT.log" <<'REMOTE'
set -eu
source_image=$1
remote_image=$2
remote_repo=$3
sudo -n docker image inspect "$source_image" >/dev/null
source_id=$(sudo -n docker image inspect "$source_image" --format '{{.Id}}')
sudo -n docker tag "$source_image" "$remote_image"
sudo -n docker push "$remote_image" >&2
pushed_id=$(sudo -n docker image inspect "$remote_image" --format '{{.Id}}')
[ "$source_id" = "$pushed_id" ] || {
  echo "source/tag image ID mismatch: $source_id != $pushed_id" >&2
  exit 4
}
sudo -n docker image inspect "$remote_image"
REMOTE

digest="$(curl -fsSI -H 'Accept: application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json' \
  "http://127.0.0.1:${PORT}/v2/genesis-serve/manifests/golden" | tr -d '\r' | awk -F': ' 'tolower($1)=="docker-content-digest"{print $2}')"
[[ "$digest" == sha256:* ]] || { echo "registry returned no immutable digest" >&2; exit 3; }
python3 - "$OUT/SSH_TUNNEL_REGISTRY_PUSH_AND_INSPECT.log" "$OUT/LOCAL_REGISTRY_RECEIPT.json" "$HOST" "$REMOTE_IMAGE" "$REMOTE_REPO" "$digest" <<'PY'
import hashlib,json,sys,time
log,out,host,image,repo,digest=sys.argv[1:]
text=open(log,errors='ignore').read()
start=text.rfind('\n[')
if start < 0:
    start=text.find('[')
if start < 0:
    raise SystemExit('remote inspect JSON missing')
inspect=json.loads(text[start+1 if text[start:start+2]=='\n[' else start:])[0]
expected=f'{repo}@{digest}'
repo_digests=inspect.get('RepoDigests') or []
if expected not in repo_digests:
    raise SystemExit(f'remote inspect is not bound to fetched registry digest: {expected} not in {repo_digests}')
receipt={
 'schema':'genesis-golden-local-registry-via-ssh-v2','status':'PASS','created_unix':time.time(),
 'source_host':host,'registry_image':image,'manifest_digest':digest,
 'repo_digest':expected,'image_id':inspect['Id'],'size':inspect['Size'],
 'remote_repo_digests':repo_digests,
 'push_inspect_log_sha256':hashlib.sha256(open(log,'rb').read()).hexdigest(),
 'truth_label':'PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C',
}
open(out,'w').write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
PY
