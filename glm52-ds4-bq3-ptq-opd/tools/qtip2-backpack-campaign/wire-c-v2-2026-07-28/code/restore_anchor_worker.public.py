#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, subprocess, time, uuid
from pathlib import Path

TASK = "PUBLIC_TASK"
K1_TASK = "PUBLIC_TASK"
ROOT = Path("$SOURCE_ROOT/P880_QTIP2_ASSEALED_PUBLIC_TASK_s3")
K1 = Path("$SOURCE_ROOT/P852_K1_PUBLIC_TASK_s3")
CLAIM = Path("$SOURCE_ROOT/HOST_CLAIM.json")
SRC = Path("/dev/shm/P852_K1_PUBLIC_TASK_s3/output/L017/units/L017")
BANK = K1 / "paused_bank/P880_L017_448"
MANIFEST = ROOT / "receipts/K1_BANK_L017_448_MANIFEST.json"
TARGET_Q2_CLAIM_SHA = "614fbedd03a725819cadc2dd057dcc78a4db222392cdf7de20d74f9d65805fea"
TARGET_K1_CLAIM_SHA = "6ee6a5362e13903f36dfd5498b514a0fae891e2e4e9a3b233944e5231e06d1cf"
TARGET_MANIFEST_SHA = "042074d24639ae53dface6d05a9843de706822ed5adf0967984c52235ef223e4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic(path: Path, obj: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with tmp.open("wb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(dfd)
    finally: os.close(dfd)
    json.loads(path.read_text())
    return raw


def matching_processes():
    lines = subprocess.run(["ps", "-eo", "pid,ppid,state,args"], text=True, capture_output=True, check=True).stdout.splitlines()
    keys = ["P880_QTIP2_ASSEALED", "p852_s3_controller.py", "p852_k1_build_layer.py"]
    return [x for x in lines if any(k in x for k in keys) and str(os.getpid()) not in x]


def gpu_apps():
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    return [x for x in p.stdout.splitlines() if x.strip()]

if sha(MANIFEST) != TARGET_MANIFEST_SHA:
    raise RuntimeError("bank manifest SHA drift")
manifest = json.loads(MANIFEST.read_text())
if manifest.get("status") != "PASS" or manifest.get("valid_banked_units") != 448 or len(manifest.get("rows", [])) != 448:
    raise RuntimeError("bank manifest closure drift")
if matching_processes():
    raise RuntimeError({"unexpected_owned_processes": matching_processes()})
if gpu_apps():
    raise RuntimeError({"gpu_not_empty": gpu_apps()})

# Rehash every banked pair and the active restart namespace against the sealed manifest.
for row in manifest["rows"]:
    stem = row["identity"]
    for base in (BANK, SRC):
        pt = base / f"{stem}.pt"
        done = base / f"{stem}.DONE.json"
        if not pt.is_file() or not done.is_file():
            raise RuntimeError(f"missing restart pair {base} {stem}")
        if pt.stat().st_size != row["pt_bytes"] or done.stat().st_size != row["done_bytes"]:
            raise RuntimeError(f"restart size drift {base} {stem}")
        if sha(pt) != row["pt_sha256"] or sha(done) != row["done_sha256"]:
            raise RuntimeError(f"restart SHA drift {base} {stem}")

raw = CLAIM.read_bytes()
if hashlib.sha256(raw).hexdigest() != TARGET_Q2_CLAIM_SHA:
    raise RuntimeError("Q2 claim exact-CAS preimage drift")
current = json.loads(raw)
if current.get("owner") != TASK or current.get("status") != "CLAIMED":
    raise RuntimeError("Q2 claim owner/status drift")
previous = current.get("previous_claim")
restored_raw = (json.dumps(previous, indent=2, sort_keys=True) + "\n").encode()
if hashlib.sha256(restored_raw).hexdigest() != TARGET_K1_CLAIM_SHA:
    raise RuntimeError("reconstructed K1 exact preimage mismatch")
if CLAIM.read_bytes() != raw:
    raise RuntimeError("claim changed before restore CAS")
tmp = CLAIM.with_name(f".{CLAIM.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
with tmp.open("wb") as f:
    f.write(restored_raw); f.flush(); os.fsync(f.fileno())
os.replace(tmp, CLAIM)
dfd = os.open(CLAIM.parent, os.O_RDONLY)
try: os.fsync(dfd)
finally: os.close(dfd)
readback = CLAIM.read_bytes()
if readback != restored_raw or hashlib.sha256(readback).hexdigest() != TARGET_K1_CLAIM_SHA:
    raise RuntimeError("K1 claim exact readback failure")
restored = json.loads(readback)
if restored.get("owner") != K1_TASK or restored.get("task") != K1_TASK:
    raise RuntimeError("K1 restored binding drift")

log = K1 / "logs/P852_K1_RESUMED_AFTER_P880.log"
log.parent.mkdir(parents=True, exist_ok=True)
cmd = ["$SOURCE_ROOT/python", "-u", str(K1 / "code/p852_s3_controller.py")]
with log.open("ab", buffering=0) as h:
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=h, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
time.sleep(3)
if proc.poll() is not None:
    raise RuntimeError(f"K1 controller exited immediately rc={proc.returncode}")
psline = subprocess.run(["ps", "-o", "pid=,ppid=,state=,args=", "-p", str(proc.pid)], text=True, capture_output=True, check=True).stdout.strip()
if "p852_s3_controller.py" not in psline:
    raise RuntimeError("K1 controller cmdline verification failed")
launch = {
    "schema": "p880-k1-exact-restore-launch-v1", "status": "PASS_LAUNCHED_PENDING_MOTION",
    "task": TASK, "restored_task": K1_TASK, "host": "compute-node", "checkpoint": 448,
    "bank_manifest_path": str(MANIFEST), "bank_manifest_sha256": TARGET_MANIFEST_SHA,
    "bank_and_live_pairs_rehashed": 448, "bank_and_live_files_rehashed": 1792,
    "claim_preimage_q2_sha256": TARGET_Q2_CLAIM_SHA, "claim_restored_exact_sha256": TARGET_K1_CLAIM_SHA,
    "controller_pid": proc.pid, "controller_pgid": os.getpgid(proc.pid), "controller_sid": os.getsid(proc.pid),
    "controller_cmd": cmd, "controller_ps": psline, "log": str(log), "gpu_apps_at_launch": gpu_apps(),
    "created_unix": time.time(),
}
atomic(ROOT / "receipts/K1_RESTORED_AFTER_P880_LAUNCH.json", launch)
atomic(K1 / "receipts/RESUMED_AFTER_P880_LAUNCH.json", launch)
(K1 / "run/CONTROLLER_RESUMED_AFTER_P880.pid").write_text(str(proc.pid) + "\n")
print(json.dumps(launch, sort_keys=True))
