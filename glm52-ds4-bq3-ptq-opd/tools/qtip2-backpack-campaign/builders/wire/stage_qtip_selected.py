#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOSTS = {
    "compute-node-3": "fleet-user@203.0.113.3",
    "compute-node-4": "fleet-user@203.0.113.4",
    "compute-node-5-work": "fleet-user@203.0.113.7",
}
ROOTS = {
    "s6": Path("$HOME/run-bundles/P640_BANANA_SMASHER_QTIP2_WIRE_PUBLIC_TASK_s6"),
    "s8": Path("$HOME/run-bundles/P640_BANANA_SMASHER_QTIP2_WIRE_PUBLIC_TASK_s8"),
}
EXPECTED_ASSIGNMENT = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"

def actual_source(row: dict) -> str:
    """Resolve campaign-prestaged copies without mutating source hosts."""
    basename = row["basename"]
    source_host = row["stage_source_host"]
    if source_host == "compute-node-3":
        return f"/dev/shm/P622_QTIP2_ALL16/L{int(row['layer']):03d}/{basename}"
    if source_host == "compute-node-4":
        return "$HOME/run-bundles/P604_QTIP2_DAMAGE_ANCHORS_SHARDB_PUBLIC_TASK_s4/staged/src_192_168_200_7$HOME/run-bundles/P534_QTIP2_REP16_PUBLIC_TASK_s5w/artifacts/" + basename
    return row["source_artifact"]

def sha(path: Path, chunk: int = 16 << 20) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(chunk),b""): h.update(b)
    return h.hexdigest()

def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,path)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",choices=sorted(ROOTS),required=True); ap.add_argument("--streams",type=int,default=8); a=ap.parse_args()
    root=ROOTS[a.shard]; source=root/"inputs/QTIP_SELECTED_EXPECTED.json"; expected=json.loads(source.read_text())
    if expected.get("assignment_sha256")!=EXPECTED_ASSIGNMENT or expected.get("selected_count")!=406: raise RuntimeError("expected manifest identity drift")
    rows=[r for r in expected["rows"] if r["stage_shard"]==a.shard]
    final=root/"inputs/qtip_selected"; final.mkdir(parents=True,exist_ok=True)
    started=time.time(); missing=[]; sealed=[]
    for r in rows:
        dst=final/r["basename"]
        if dst.is_file() and dst.stat().st_size==int(r["artifact_bytes"]) and sha(dst)==r["artifact_sha256"]:
            sealed.append(r); continue
        if dst.exists(): dst.rename(dst.with_name(dst.name+f".bad.{int(time.time())}"))
        missing.append(r)
    for stale in (root/"run").glob(f"qtip_stage_{a.shard}.*"):
        shutil.rmtree(stale,ignore_errors=True)
    scratch=root/"run"/f"qtip_stage_{a.shard}.{os.getpid()}"; scratch.mkdir(parents=True,exist_ok=True)
    groups={}
    for r in missing: groups.setdefault(r["stage_source_host"],[]).append(r)
    transfers=[]
    for source_host, group in sorted(groups.items()):
        remote=HOSTS[source_host]; chunks=[group[i::a.streams] for i in range(a.streams)]; chunks=[c for c in chunks if c]
        source_started=time.time(); jobs=[]
        for i,chunk in enumerate(chunks):
            listp=scratch/f"{source_host}.stream{i}.files"; listp.write_text("".join(actual_source(r).lstrip("/")+"\n" for r in chunk))
            dest=scratch/f"{source_host}.stream{i}"; dest.mkdir(parents=True,exist_ok=True)
            log=scratch/f"{source_host}.stream{i}.log"; sink=log.open("wb")
            cmd=["rsync","-a","--whole-file","--no-compress","--timeout=180","--files-from",str(listp),"-e","ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4",f"{remote}:/",str(dest)+"/"]
            proc=subprocess.Popen(cmd,stdout=sink,stderr=subprocess.STDOUT); jobs.append((proc,sink,log,dest,chunk))
        for proc,sink,log,dest,chunk in jobs:
            rc=proc.wait(); sink.close()
            if rc: raise RuntimeError(f"rsync {source_host} rc={rc}: {log.read_text(errors='replace')[-4000:]}")
            for r in chunk:
                src=dest/actual_source(r).lstrip("/"); dst=final/r["basename"]
                if not src.is_file() or src.stat().st_size!=int(r["artifact_bytes"]) or sha(src)!=r["artifact_sha256"]: raise RuntimeError(f"staged artifact drift {src}")
                os.replace(src,dst); sealed.append(r)
        wall=time.time()-source_started; payload=sum(int(r["artifact_bytes"]) for r in group)
        transfers.append({"source_host":source_host,"source_qsfp":remote,"streams":len(chunks),"files":len(group),"bytes":payload,"wall_seconds":wall,"aggregate_GBps":payload/wall/1e9 if wall else None})
    shutil.rmtree(scratch,ignore_errors=True)
    if len(sealed)!=len(rows): raise RuntimeError("sealed row count drift")
    keys={(int(r["layer"]),int(r["expert"]),r["projection"]) for r in rows}
    final_rows=[]
    for r in sorted(rows,key=lambda x:(int(x["layer"]),int(x["expert"]),x["projection"])):
        p=final/r["basename"]
        if p.stat().st_size!=int(r["artifact_bytes"]) or sha(p)!=r["artifact_sha256"]: raise RuntimeError(f"final hash drift {p}")
        final_rows.append({"layer":int(r["layer"]),"expert":int(r["expert"]),"projection":r["projection"],"path":str(p),"bytes":p.stat().st_size,"sha256":r["artifact_sha256"],"logical_bytes":int(r["logical_bytes"])})
    result={"schema":"p640-final406-qtip-stage-v1","status":"PASS","task":"PUBLIC_TASK","shard":a.shard,"assignment_sha256":EXPECTED_ASSIGNMENT,"selected_total":406,"shard_units":len(final_rows),"layers":sorted({x[0] for x in keys}),"physical_bytes":sum(r["bytes"] for r in final_rows),"logical_bytes":sum(r["logical_bytes"] for r in final_rows),"rows":final_rows,"transfers":transfers,"elapsed_seconds":time.time()-started,"completed_unix":time.time()}
    receipt=root/"receipts"/f"QTIP_SELECTED_FINAL406_{a.shard.upper()}_DONE.json"; atomic_json(receipt,result)
    print(json.dumps({"status":"PASS","receipt":str(receipt),"sha256":sha(receipt),"units":len(final_rows),"physical_bytes":result["physical_bytes"],"transfers":transfers},sort_keys=True),flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
