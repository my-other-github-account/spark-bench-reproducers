#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, shutil, time, uuid
from pathlib import Path

TASK = "PUBLIC_TASK"
ROOT = Path("$SOURCE_ROOT/P880_QTIP2_ASSEALED_PUBLIC_TASK_s3")
K1 = Path("$SOURCE_ROOT/P852_K1_PUBLIC_TASK_s3")
CLAIM = Path("$SOURCE_ROOT/HOST_CLAIM.json")
SRC = Path("/dev/shm/P852_K1_PUBLIC_TASK_s3/output/L017/units/L017")
BANK = K1 / "paused_bank/P880_L017_448"


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)

prep=json.loads((ROOT/"receipts/P880_Q2_PREPARED.json").read_text())
if prep.get("status") != "PASS_READY_TO_CLAIM": raise RuntimeError("P880 prep gate absent")
raw=CLAIM.read_bytes(); pre_sha=hashlib.sha256(raw).hexdigest(); current=json.loads(raw)
if current.get("owner")!="PUBLIC_TASK" or current.get("status") not in ("CLAIMED","ACTIVE"):
    raise RuntimeError(f"unexpected claim owner/status {current.get('owner')} {current.get('status')}")
if pre_sha != "6ee6a5362e13903f36dfd5498b514a0fae891e2e4e9a3b233944e5231e06d1cf":
    raise RuntimeError(f"K1 restored claim drift {pre_sha}")
# K1 is already stopped; adopt, don't rebuild, its newest exact 64 boundary.
ckpt=K1/"receipts/L017_CHECKPOINT_448.json"; c=json.loads(ckpt.read_text())
if c.get("status")!="PASS" or c.get("completed_units")!=448 or c.get("layer")!=17:
    raise RuntimeError("K1 L017 checkpoint 448 drift")
rows=[]
for expert in range(224):
    for projection in ("fused13","down"):
        stem=f"L017_E{expert:03d}_{projection}_L16_K1_V2"
        pt=SRC/f"{stem}.pt"; done=SRC/f"{stem}.DONE.json"
        if not pt.is_file() or not done.is_file(): raise RuntimeError(f"missing checkpoint pair {stem}")
        rows.append({"identity":stem,"pt_sha256":sha(pt),"pt_bytes":pt.stat().st_size,"done_sha256":sha(done),"done_bytes":done.stat().st_size})
if len(rows)!=448: raise RuntimeError("bank row count")
rows_sha=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest()
if BANK.exists(): shutil.rmtree(BANK)
BANK.mkdir(parents=True)
for row in rows:
    stem=row["identity"]
    shutil.copy2(SRC/f"{stem}.pt", BANK/f"{stem}.pt")
    shutil.copy2(SRC/f"{stem}.DONE.json", BANK/f"{stem}.DONE.json")
# Readback the bank before claim mutation.
for row in rows:
    stem=row["identity"]
    if sha(BANK/f"{stem}.pt")!=row["pt_sha256"] or sha(BANK/f"{stem}.DONE.json")!=row["done_sha256"]:
        raise RuntimeError(f"bank readback drift {stem}")
manifest={"schema":"p880-k1-bank-v2","status":"PASS","task":TASK,"layer":17,"valid_banked_units":448,"rows_sha256":rows_sha,"rows":rows,"checkpoint_path":str(ckpt),"checkpoint_sha256":sha(ckpt),"created_unix":time.time()}
manifest_path=ROOT/"receipts/K1_BANK_L017_448_MANIFEST.json"; atomic(manifest_path,manifest)
# Exact compare immediately before atomic replacement.
if CLAIM.read_bytes()!=raw: raise RuntimeError("claim changed before exact-CAS takeover")
now=time.time()
new={
 "schema":"spark-host-claim-v2","status":"CLAIMED","state":"CLAIMED","host":"compute-node",
 "owner":TASK,"task":TASK,"task_id":TASK,"mission":str(ROOT),"mission_root":str(ROOT),
 "claim_nonce":uuid.uuid4().hex,"claimed_unix":now,"lease_until_unix":now+8*3600,
 "exact_cas_from_sha256":pre_sha,"previous_claim":current,
 "scope":{"anchor":"qtip2","coverage":"PARTIAL_VERTICAL_39_OF_40","layers":[x for x in range(3,43) if x!=13],"windows":64,"geometry":{"K":2,"L":16,"V":2,"tlut_bits":9}},
 "scratch":"/dev/shm/P880_QTIP2_ASSEALED_PUBLIC_TASK_s3","no_services":True,"no_tailscale":True,"no_autoresume":True,
 "launch_policy":"task-owned nohup+setsid only; no services/systemd/systemd-run/tmux/Tailscale",
}
tmp=CLAIM.with_name(f".{CLAIM.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
tmp.write_text(json.dumps(new,indent=2,sort_keys=True)+"\n")
os.replace(tmp,CLAIM)
post_raw=CLAIM.read_bytes(); post=json.loads(post_raw); post_sha=hashlib.sha256(post_raw).hexdigest()
if post.get("owner")!=TASK or post.get("exact_cas_from_sha256")!=pre_sha: raise RuntimeError("claim readback drift")
receipt={"schema":"p880-k1-stopped-adopt-takeover-v1","status":"PASS","task":TASK,"k1_processes_before":[],"adopted_exact_checkpoint":448,"checkpoint_path":str(ckpt),"checkpoint_sha256":sha(ckpt),"bank_path":str(BANK),"bank_manifest_path":str(manifest_path),"bank_manifest_sha256":sha(manifest_path),"claim_preimage_sha256":pre_sha,"claim_postimage_sha256":post_sha,"claim_nonce":post["claim_nonce"],"created_unix":time.time()}
atomic(ROOT/"receipts/K1_ADOPTED_448_AND_Q2_CLAIMED.json",receipt)
print(json.dumps(receipt,sort_keys=True))
