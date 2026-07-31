#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "usage: ./build.sh" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="genesis-serve:golden"
SEAL="ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5"
RECEIPT="$ROOT/BUILD_RECEIPT.json"
LOG="$ROOT/BUILD.log"

command -v docker >/dev/null
test -f "$ROOT/Dockerfile"
test -f "$ROOT/SOURCE_MANIFEST.json"

python3 - "$ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
for relative, row in manifest["files"].items():
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"sealed source missing/non-regular: {relative}")
    data = path.read_bytes()
    if len(data) != row["bytes"]:
        raise SystemExit(f"sealed source byte drift: {relative}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != row["sha256"]:
        raise SystemExit(f"sealed source SHA drift: {relative}: {actual}")
print(f"SOURCE_MANIFEST_PASS files={len(manifest['files'])}")
PY

DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --provenance=false \
  --progress=plain \
  --tag "$IMAGE" \
  --file "$ROOT/Dockerfile" \
  "$ROOT" 2>&1 | tee "$LOG"

docker image inspect "$IMAGE" > "$ROOT/IMAGE_INSPECT.json"
python3 - "$ROOT" "$IMAGE" "$SEAL" "$RECEIPT" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
image, seal, output = sys.argv[2], sys.argv[3], Path(sys.argv[4])
inspect_path = root / "IMAGE_INSPECT.json"
inspect = json.loads(inspect_path.read_text())[0]
config = inspect["Config"]
labels = config.get("Labels") or {}
if labels.get("io.genesis.parent-hand.ladder-seal.sha256") != seal:
    raise SystemExit("built image ladder-seal label mismatch")
if config.get("Entrypoint") not in (None, []):
    raise SystemExit(f"stock contract requires null entrypoint: {config.get('Entrypoint')}")
if config.get("Cmd") != ["vllm", "serve", "/model"]:
    raise SystemExit(f"stock command mismatch: {config.get('Cmd')}")
image_id = subprocess.check_output(
    ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
).strip()
source_manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
receipt = {
    "schema": "banana-smasher-build-receipt-v1",
    "status": "PASS_STATIC_BUILD_RUNTIME_HEALTH_PENDING",
    "created_unix": time.time(),
    "image": image,
    "image_id": image_id,
    "parent_hand_ladder_seal_sha256": seal,
    "p1321_ladder_seal_sha256": labels.get("io.genesis.p1321.ladder-seal.sha256"),
    "cmd": config.get("Cmd"),
    "entrypoint": config.get("Entrypoint"),
    "healthcheck": config.get("Healthcheck"),
    "labels": labels,
    "source_manifest_sha256": hashlib.sha256(
        (root / "SOURCE_MANIFEST.json").read_bytes()
    ).hexdigest(),
    "source_manifest": source_manifest,
    "dockerfile_sha256": hashlib.sha256((root / "Dockerfile").read_bytes()).hexdigest(),
    "build_log_sha256": hashlib.sha256((root / "BUILD.log").read_bytes()).hexdigest(),
    "image_inspect_sha256": hashlib.sha256(inspect_path.read_bytes()).hexdigest(),
    "runtime_gate": {
        "receipt": "/tmp/GOLDEN_PERF_HEALTH.json",
        "bars": {"c1_tok_s": 13.0, "c4_tok_s": 27.0, "c4_gt_c2": True},
        "degraded_is_unhealthy": True,
    },
}
output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({
    "status": receipt["status"],
    "image": image,
    "image_id": image_id,
    "receipt": str(output),
}, sort_keys=True))
PY
