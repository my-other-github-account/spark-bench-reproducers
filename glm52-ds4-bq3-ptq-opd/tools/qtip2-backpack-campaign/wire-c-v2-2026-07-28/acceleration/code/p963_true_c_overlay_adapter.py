#!/usr/bin/env python3
"""Exact sparse TRUE-C f521-T adapter for cumulative/full BALANCED64 rails.

Consumes a task-local active-overlay manifest assembled only from witnessed complete
layers. Inactive layers remain byte-bound to the immutable Genesis base. Artifact
payloads stream layer-at-a-time from spark-local peers; NAS is never consulted.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
from typing import Any, Mapping

TASK = "PUBLIC_TASK"
MEASUREMENT_TASK = "PUBLIC_TASK"
ROOT = Path(os.environ["P770_ROOT"])
ACTIVE_MANIFEST = Path(os.environ["P885_ACTIVE_MANIFEST"])
ACTIVE_ASSIGNMENT = Path(os.environ["P885_ACTIVE_ASSIGNMENT"])
BASE_ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
MODEL = Path("/PUBLIC_SOURCE_ROOT/models/hf/DeepSeek-V4-Flash")
HOST_ADDR = {"compute-node-a":"PUBLIC_NODE_A_ADDRESS", "compute-node-c":"PUBLIC_NODE_C_ADDRESS", "compute-node-f":"PUBLIC_NODE_F_ADDRESS", "compute-node-g":"PUBLIC_NODE_H_ADDRESS", "compute-node-h":"PUBLIC_NODE_I_ADDRESS"}
FINAL_ASSIGNMENT_SHA = "f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39"
FINAL_MAP_SHA = "786b01a3f8c0197407e0025c80ca92c29b347a9c18de4b1ca48b7cf52ae08df6"
F949_ASSIGNMENT_SHA = "f949b01a29049b03c9dcba6fb1c9df775d414427aeb9f475cd176503f8ddd654"
F949_MAP_SHA = "6afeeac7ff6e3510a04c55c688f95c11cda78e5e96d78b7c864fe1ec852ea8a4"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
BASE_WIRE_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
PLAN_SHA = "8c92ce62167db7980fde20b8e32cecc6934a816bb2a4b65dd78e99ecbf8f29c4"
PLAN_ROWS_SHA = "985ca4030c3381a1242bb6bc75763b034110f7404b0ddbbf714a2c170fcfb766"
FINAL_RESULT_SHA = "6e866b468716758ee00abe8277f0b9789372d24df478a1dcbe56925ee64deea0"
P653_ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
VQ_BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def canonical(v: object) -> str:
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def flatten(path: Path) -> dict[tuple[int,int,str],str]:
    doc=json.loads(path.read_text())["assignment"]; out={}
    for l in range(43):
        for e in range(256):
            for p in ("fused13","down"): out[(l,e,p)]=str(doc[str(l)][str(e)][p])
    if len(out)!=22016: raise RuntimeError(f"assignment surface drift {path}")
    return out

def ident(r: Mapping[str,Any]) -> tuple[int,int,str]:
    return int(r["layer"]),int(r["expert"]),str(r["projection"])

def preflight_manifest() -> dict[str,Any]:
    raw=ACTIVE_MANIFEST.read_bytes(); doc=json.loads(raw)
    if doc.get("schema")!="p885-wire-c-active-overlay-v1" or doc.get("status")!="PASS_EXACT_ACTIVE_LAYERS":
        raise RuntimeError("active overlay schema/status drift")
    exact={
      "task_id":TASK,"wire_label":"WIRE_C_TRUE_C_f521_T","definitive":True,"locally_altered":True,"stale":False,
      "measurement_suite":"BALANCED64_V1","pack_fraction":1.0,
      "final_assignment_sha256":FINAL_ASSIGNMENT_SHA,"final_assignment_map_sha256":FINAL_MAP_SHA,
      "base_assignment_sha256":BASE_ASSIGNMENT_SHA,"base_wire_manifest_sha256":BASE_WIRE_SHA,
      "plan_sha256":PLAN_SHA,"plan_rows_sha256":PLAN_ROWS_SHA,"result_sha256":FINAL_RESULT_SHA,
      "repair_and_dosing_blocked":True,
    }
    drift={k:{"expected":v,"observed":doc.get(k)} for k,v in exact.items() if doc.get(k)!=v}
    if drift: raise RuntimeError(f"active overlay binding drift {drift}")
    if not isinstance(doc.get("p925_build_identity_sha256"),str) or len(doc["p925_build_identity_sha256"])!=64:
        raise RuntimeError("TRUE-C build identity pin missing")
    # Runtime-local mirrors are bound by the sealed hashes. The immutable terminal
    # overlay retains its original provenance paths, but those mutable path strings
    # are never used as runtime authority on this independent host.
    plan_path=ROOT / "inputs/BUILD_PLAN_WIRE_C_V2_PREVIEW.json"
    final_path=ACTIVE_ASSIGNMENT
    if sha(plan_path)!=PLAN_SHA or sha(final_path)!=FINAL_ASSIGNMENT_SHA or sha(BASE_ASSIGNMENT)!=BASE_ASSIGNMENT_SHA:
        raise RuntimeError("plan/final/base file pin drift")
    if sha(ACTIVE_ASSIGNMENT)!=doc.get("active_assignment_sha256"):
        raise RuntimeError("active assignment SHA drift")
    layers=[int(x) for x in doc.get("active_layers",[])]
    if layers!=sorted(set(layers)) or any(x<0 or x>42 for x in layers): raise RuntimeError("active layer list drift")
    plan=json.loads(plan_path.read_text()); plan_rows=sorted(plan["rows"],key=ident)
    if canonical(plan_rows)!=PLAN_ROWS_SHA: raise RuntimeError("plan rows canonical drift")
    expected=[r for r in plan_rows if int(r["layer"]) in set(layers)]
    rows=doc.get("rows")
    if not isinstance(rows,list) or len(rows)!=len(expected): raise RuntimeError("active row count drift")
    if canonical(rows)!=doc.get("active_rows_sha256"): raise RuntimeError("active rows canonical drift")
    normalized=[]
    for got,want in zip(rows,expected):
        if ident(got)!=ident(want) or got.get("old")!=want.get("old") or got.get("new")!=want.get("new"):
            raise RuntimeError(f"active semantic row drift {ident(got)}")
        method=str(got.get("effective_method"))
        if method=="current_qtip_inventory": kind="qtip2_exact" if got["new"]=="qtip2_2.0117" else "qtip3_exact"
        elif method in ("fresh_canonical_vq_build","reuse_sealed_p653","fresh_canonical_vq_refit_p925"): kind="genesis_vq"
        elif method=="native_checkpoint_reference": kind="native_mxfp4"
        else: raise RuntimeError(f"unsupported effective method {method}")
        row=dict(got); row["kind"]=kind; row["old_tier"]=got["old"]; row["new_tier"]=got["new"]
        if kind!="native_mxfp4":
            for k in ("artifact","artifact_bytes","artifact_sha256","source_host"):
                if not row.get(k): raise RuntimeError(f"missing {k} {ident(row)}")
            if row.get("pack_fraction")!=1.0: raise RuntimeError(f"pack drift {ident(row)}")
        if kind=="genesis_vq":
            for k in ("codebook","codebook_sha256"):
                if not row.get(k): raise RuntimeError(f"missing {k} {ident(row)}")
        normalized.append(row)
    base,final,active=flatten(BASE_ASSIGNMENT),flatten(final_path),flatten(ACTIVE_ASSIGNMENT)
    active_set=set(layers)
    for k in base:
        want=final[k] if k[0] in active_set else base[k]
        if active[k]!=want: raise RuntimeError(f"active assignment semantic drift {k}")
    if {k for k in active if active[k]!=base[k]}!={ident(r) for r in normalized}:
        raise RuntimeError("active assignment/row identity drift")
    by_layer={l:[r for r in normalized if int(r["layer"])==l] for l in range(43)}
    kinds=Counter(r["kind"] for r in normalized)
    return {"rows":normalized,"by_layer":by_layer,"manifest_rows":[{"name":"P925_TRUE_C_ACTIVE_OVERLAY","path":str(ACTIVE_MANIFEST),"sha256":hashlib.sha256(raw).hexdigest(),"rows":len(normalized)}],"changed_cells":len(normalized),"unchanged_cells":22016-len(normalized),"qtip2_cells":kinds["qtip2_exact"],"qtip3_cells":kinds["qtip3_exact"],"vq_cells":kinds["genesis_vq"],"native_cells":kinds["native_mxfp4"],"identity_set_sha256":canonical(sorted(ident(r) for r in normalized)),"coverage_layers":layers,"inventory_sha256":hashlib.sha256(raw).hexdigest(),"assignment_sha256":sha(ACTIVE_ASSIGNMENT),"final_assignment_sha256":FINAL_ASSIGNMENT_SHA,"active_rows_sha256":doc["active_rows_sha256"],"target_contract_sha256":doc["p925_target_contract_sha256"],"build_identity_sha256":doc["p925_build_identity_sha256"],"pack_fraction":1.0,"source_physical_manifest":doc.get("source_physical_manifest"),"compatibility_binding":doc.get("compatibility_binding"),"codebook_deviation_disclosure":doc.get("codebook_deviation_disclosure")}

def _copy(spec: Mapping[str,Any], dst: Path) -> None:
    host=str(spec["host"]); source=str(spec["source"])
    dst.parent.mkdir(parents=True,exist_ok=True)
    if host==os.environ.get("P770_HOST", "compute-node-a"): shutil.copyfile(source,dst)
    elif host in HOST_ADDR:
        subprocess.run(["rsync","-a","--partial","--timeout=180","-e","ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8",f"{HOST_ADDR[host]}:{source}",str(dst)],check=True,capture_output=True,text=True)
    else: raise RuntimeError(f"unsupported source host {host}")

def stage_layer(layer:int, rows:list[dict[str,Any]], cache:Path):
    specs={}
    for r in rows:
        if r["kind"]=="native_mxfp4": continue
        for role in ("artifact","codebook"):
            src=r.get(role)
            if not src: continue
            expected_sha=str(r[f"{role}_sha256"])
            s={"host":r["source_host"],"source":src,"rel":f"{role}s/by_sha/{expected_sha}/{Path(src).name}","bytes":int(r.get(f"{role}_bytes") or (Path(src).stat().st_size if r["source_host"]==os.environ.get("P770_HOST", "compute-node-a") else 0)),"sha256":expected_sha}
            prior=specs.setdefault((s["host"],s["source"]),s)
            if prior!=s: raise RuntimeError(f"conflicting source {s['source']}")
            r[f"{role}_stage_rel"]=s["rel"]
    if not specs: return None,{"layer":layer,"files":0,"bytes":0,"elapsed_seconds":0.0,"stage_retired":True}
    stage=cache/f"wire_c_layer_{layer:03d}"; partial=cache/f".{stage.name}.{uuid.uuid4().hex}.partial"
    if stage.exists() or partial.exists(): raise RuntimeError(f"once-only stage exists L{layer:03d}")
    partial.mkdir(parents=True); ordered=list(specs.values()); started=time.time()
    try:
        # One SSH/rsync session per peer, not one session per artifact.  Preserve
        # absolute source paths in a raw tree, then hardlink into the expected-SHA
        # namespace so basename collisions remain impossible without doubling bytes.
        by_host={}
        for s in ordered: by_host.setdefault(s["host"],[]).append(s)
        raw_root=partial/"_peer_raw"; raw_root.mkdir()
        for host,host_specs in by_host.items():
            if host==os.environ.get("P770_HOST", "compute-node-a"):
                for s in host_specs:
                    src=Path(s["source"]); raw=raw_root/str(src).lstrip("/")
                    raw.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,raw)
            else:
                if host not in HOST_ADDR: raise RuntimeError(f"unsupported source host {host}")
                files_from=partial/f".files_from_{host}.{os.getpid()}"
                files_from.write_bytes(b"\0".join(str(s["source"]).lstrip("/").encode() for s in host_specs)+b"\0")
                subprocess.run(["rsync","-a","-R","--from0",f"--files-from={files_from}","--partial","--timeout=180","-e","ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=8",f"{HOST_ADDR[host]}:/",str(raw_root)],check=True,capture_output=True,text=True)
                files_from.unlink()
        for s in ordered:
            raw=raw_root/str(s["source"]).lstrip("/"); dst=partial/s["rel"]
            dst.parent.mkdir(parents=True,exist_ok=True); os.link(raw,dst)
        shutil.rmtree(raw_root)
        def verify(s):
            p=partial/s["rel"]
            if not p.is_file() or (s["bytes"] and p.stat().st_size!=s["bytes"]) or sha(p)!=s["sha256"]:
                raise RuntimeError(f"staged identity drift {s['source']}")
        with ThreadPoolExecutor(max_workers=min(8,len(ordered))) as ex: list(ex.map(verify,ordered))
        os.replace(partial,stage)
    except Exception:
        shutil.rmtree(partial,ignore_errors=True); raise
    total=sum((stage/s["rel"]).stat().st_size for s in ordered)
    return stage,{"layer":layer,"files":len(ordered),"bytes":total,"elapsed_seconds":time.time()-started,"transport":"batched one-session-per-peer QSFP rsync; expected-SHA namespace; no NAS"}

def install_stream_source(p651:Any, base:Any, manifest:Mapping[str,Any], cache:Path, mode:str, qtip_source:Path,qtip_kernel:Path,qtip_tlut:Path):
    legacy=load_module("p885_legacy_mixed_adapter",ROOT/"code/p760_overlay_adapter.py")
    by_layer=manifest["by_layer"]; stage_rows=[]; applied={}; holder={}
    overlay_cache=Path("/dev/shm/P963_TRUE_C_OVERLAY_PUBLIC_TASK")/os.environ["P885_RUN_ID"]
    overlay_cache.mkdir(parents=True,exist_ok=True)
    class WireCSource(base.GenesisTierSource):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self._overlay_stage_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="p963-overlay-stage")
            self._overlay_stage_futures={}
        def _overlay_future(self,layer:int):
            future=self._overlay_stage_futures.get(layer)
            if future is None:
                future=self._overlay_stage_executor.submit(stage_layer,layer,by_layer[layer],overlay_cache)
                self._overlay_stage_futures[layer]=future
            return future
        def _stage_remote(self,layer:int,row:dict)->Path: return base.GenesisTierSource._stage_remote(self,layer,row)
        def fill_layer(self,layer:int,gate_up:Any,down:Any,*extra:Any)->None:
            rows=by_layer[layer]
            stage_future=self._overlay_future(layer) if rows else None
            full_keys={ident(r) for r in rows}
            exact_full=full_keys=={(layer,e,p) for e in range(256) for p in ("fused13","down")}
            layer_started=time.time()
            if not exact_full:
                super().fill_layer(layer,gate_up,down,*extra)
            if not rows:
                stage_rows.append({"layer":layer,"changed_cells":0,"files":0,"bytes":0,"elapsed_seconds":0.0,"stage_retired":True}); return
            stage,transfer=stage_future.result()
            self._overlay_stage_futures.pop(layer,None)
            if layer+1<43 and by_layer[layer+1]: self._overlay_future(layer+1)
            if any(r["kind"] in ("qtip2_exact","qtip3_exact") for r in rows) and "qtip" not in holder:
                holder["qtip"]=legacy.QtipDecoder(p651,qtip_source,qtip_kernel,qtip_tlut)
            try:
                for r in rows:
                    dst=gate_up[r["expert"]] if r["projection"]=="fused13" else down[r["expert"]]
                    if r["kind"] in ("qtip2_exact","qtip3_exact"):
                        info=holder["qtip"].decode(p651,stage/r["artifact_stage_rel"],r,dst)
                    elif r["kind"]=="native_mxfp4": info=legacy._native_tensor(p651,self,r,dst)
                    else:
                        payload=p651.torch.load(stage/r["artifact_stage_rel"],map_location="cpu",mmap=True,weights_only=True); meta=payload.get("meta") or {}
                        allowed=(meta.get("schema")=="wire-c-preview-canonical-vq-cell-v1" and meta.get("assignment_sha256")==FINAL_ASSIGNMENT_SHA and meta.get("builder_sha256")==VQ_BUILDER_SHA) or (meta.get("schema")=="p640-genesis-vq-overlay-cell-v1" and meta.get("assignment_sha256")==P653_ASSIGNMENT_SHA and meta.get("canonical_builder_sha256")==VQ_BUILDER_SHA) or (meta.get("schema") in ("p892-wire-c-vq-overlay-cell-v1","p892-wire-c-vq-cell-v1") and meta.get("task_id") in ("PUBLIC_TASK","PUBLIC_TASK","PUBLIC_TASK","PUBLIC_TASK","PUBLIC_TASK") and (meta.get("assignment_sha256"),meta.get("assignment_map_sha256")) in ((F949_ASSIGNMENT_SHA,F949_MAP_SHA),(FINAL_ASSIGNMENT_SHA,FINAL_MAP_SHA)) and meta.get("canonical_builder_sha256")==VQ_BUILDER_SHA and meta.get("pack_fraction",1.0)==1.0) or (meta.get("schema")=="p925-true-c-refit-vq-cell-v2" and meta.get("task_id")==TASK and meta.get("assignment_sha256")==FINAL_ASSIGNMENT_SHA and meta.get("assignment_map_sha256")==FINAL_MAP_SHA and meta.get("builder_sha256")==VQ_BUILDER_SHA and meta.get("target_contract_sha256")==manifest["target_contract_sha256"] and meta.get("build_identity_sha256")==manifest["build_identity_sha256"])
                        if not allowed or int(meta.get("layer",-1))!=r["layer"] or int(meta.get("expert",-1))!=r["expert"] or meta.get("projection")!=r["projection"] or meta.get("tier")!=r["new_tier"] or meta.get("codebook_sha256")!=r["codebook_sha256"] or meta.get("fp16_codebook_replay_exact") is not True:
                            raise RuntimeError(f"VQ metadata drift {ident(r)}")
                        d,k=int(meta["d"]),int(meta["k"]); cbpath=stage/r["codebook_stage_rel"]
                        cb=p651.torch.from_file(str(cbpath),dtype=p651.torch.float16,size=k*d).reshape(k,d).clone(); codes,scales=payload["codes"].unsqueeze(0),payload["scales"].unsqueeze(0)
                        if not bool(p651.torch.isfinite(cb).all()) or int(codes.min())<0 or int(codes.max())>=k: raise RuntimeError(f"VQ numerical drift {ident(r)}")
                        base.GenesisTierSource._launch_vq(codes,scales,cb,[r["expert"]],gate_up if r["projection"]=="fused13" else down,d); info={"d":d,"k":k,"finite":True,"fp16_codebook_replay_exact":True}; del payload,cb,codes,scales
                    key=ident(r)
                    if key in applied: raise RuntimeError(f"duplicate apply {key}")
                    applied[key]={"layer":r["layer"],"expert":r["expert"],"projection":r["projection"],"old_tier":r["old_tier"],"new_tier":r["new_tier"],"kind":r["kind"],"artifact_sha256":r.get("artifact_sha256"),"decode":info}
                p651.torch.cuda.synchronize()
            finally:
                gc.collect(); shutil.rmtree(stage,ignore_errors=True) if stage is not None else None
            if exact_full:
                base.GenesisTierSource.complete_overlay_full_layer(self,layer,time.time()-layer_started)
            transfer.update({"schema":"p963-accelerated-true-c-layer-consumption-v1","status":"PASS","task_id":MEASUREMENT_TASK,"canonical_task_id":TASK,"authority_binding":"expected SHA namespaced stage; mutable basenames cannot collide","mode":mode,"changed_cells":len(rows),"kinds":dict(Counter(r["kind"] for r in rows)),"base_skipped_exact_full_overlay":exact_full,"stage_retired":stage is None or not stage.exists()}); stage_rows.append(transfer)
            p651.atomic_json(ROOT/f"run/{os.environ['P885_RUN_ID']}/LAYER_{layer:03d}_OVERLAY.json",transfer)
        def finish(self):
            self._overlay_stage_executor.shutdown(wait=True,cancel_futures=False)
            shutil.rmtree(overlay_cache,ignore_errors=True)
            super().finish()
    return WireCSource,{"stage_rows":stage_rows,"applied":applied,"decoder_holder":holder}
