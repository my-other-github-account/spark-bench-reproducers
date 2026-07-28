#!/usr/bin/env python3
"""Exact sparse P637/Wire-A adapter for the P921 BALANCED64 rail.

The immutable Genesis pack supplies all 20,605 unchanged projection cells.  The
1,411 changed cells are consumed from the already-sealed, independently rehashed
P653 sparse-overlay manifest.  Every changed payload and VQ codebook is local to
spark-6 before the GPU walk; the per-layer cache is retired after consumption.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, uuid
from pathlib import Path
from typing import Any, Mapping

TASK = "task-redacted"
ROOT = Path(os.environ["P770_ROOT"])
ACTIVE_MANIFEST = Path(os.environ["P885_ACTIVE_MANIFEST"])
ACTIVE_ASSIGNMENT = Path(os.environ["P885_ACTIVE_ASSIGNMENT"])
BASE_ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
MODEL = Path("${SPARK_HOME}/models/hf/DeepSeek-V4-Flash")
HOST_ADDR = {"spark-3":"${QSFP_HOST}", "spark-6":"${QSFP_HOST}", "spark-7":"${QSFP_HOST}"}
FINAL_ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
FINAL_MAP_SHA = "36d0841986d5781186f766b3815e4b3c6332eece2090d3e6d73e7e3ffa33dc07"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
BASE_WIRE_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
P653_ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
P653_MANIFEST_SHA = "e03bc8919d51bbf1a9cf1f54f342e9f43dea625839ad8aad23578f7b8f9d98fa"
P653_OVERLAY_ROWS_SHA = "23b4211622efcd70f31e5c60d12a720c6741a57f2d4703cae6a8956ee944af15"
P653_SOURCE_ROOT = Path("${SPARK_HOME}/missions/P640_GENESIS_QTIP2_WIRE_t_e6f5ee14_s6/WIRE_STREAM_IN/P647_RESPENT")
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
    observed_manifest_sha=hashlib.sha256(raw).hexdigest()
    if observed_manifest_sha!=P653_MANIFEST_SHA:
        raise RuntimeError(f"P653 manifest SHA drift {observed_manifest_sha}")
    if doc.get("schema")!="p653-exact-sparse-overlay-assembled-wire-v1" or doc.get("status")!="PASS_EXACT_ASSEMBLED_LOGICAL_WIRE":
        raise RuntimeError("P653 manifest schema/status drift")
    exact={
      "assignment_map_sha256":FINAL_MAP_SHA,
      "overlay_rows_sha256":P653_OVERLAY_ROWS_SHA,
      "exact_wire_bytes":101346521679,
      "canonical_builder_sha256":VQ_BUILDER_SHA,
    }
    drift={k:{"expected":v,"observed":doc.get(k)} for k,v in exact.items() if doc.get(k)!=v}
    if drift: raise RuntimeError(f"P653 scalar binding drift {drift}")
    if doc.get("assignment",{}).get("sha256")!=FINAL_ASSIGNMENT_SHA or doc.get("base_assignment",{}).get("sha256")!=BASE_ASSIGNMENT_SHA or doc.get("base_wire",{}).get("manifest_sha256")!=BASE_WIRE_SHA:
        raise RuntimeError("P653 assignment/base authority drift")
    coverage=doc.get("assignment_coverage") or {}
    required_coverage={
      "changed_cells":1411,"unchanged_cells":20605,"qtip2_exact_rep16_cells":406,
      "vq_sparse_overlay_cells":1005,"computed_diff_equals_canonical_build_plan":True,
      "all_changed_artifacts_size_sha_verified":True,"all_remote_payload_sha_verified_after_spark3_stage":True,
    }
    coverage_drift={k:{"expected":v,"observed":coverage.get(k)} for k,v in required_coverage.items() if coverage.get(k)!=v}
    if coverage_drift or coverage.get("missing_changed_cells") or coverage.get("extra_changed_cells") or coverage.get("duplicate_changed_cells"):
        raise RuntimeError(f"P653 assignment coverage drift {coverage_drift}")
    contamination=doc.get("contamination_gates") or {}
    expected_contamination={"ASSIGNMENT_WITH_payload_absent":True,"old_assignment_map_26d0cd3b_absent":True,"raw_VQ3_K4096_payload_absent":True,"superseded_full_tier_outputs_absent":True}
    contamination_drift={k:{"expected":v,"observed":contamination.get(k)} for k,v in expected_contamination.items() if contamination.get(k)!=v}
    if contamination_drift:
        raise RuntimeError(f"P653 contamination gate drift {contamination_drift}")
    if sha(ACTIVE_ASSIGNMENT)!=FINAL_ASSIGNMENT_SHA or sha(BASE_ASSIGNMENT)!=BASE_ASSIGNMENT_SHA:
        raise RuntimeError("P637 final/base assignment file pin drift")

    base,final=flatten(BASE_ASSIGNMENT),flatten(ACTIVE_ASSIGNMENT)
    expected_diff={k for k in base if base[k]!=final[k]}
    rows=doc.get("overlay_rows")
    if not isinstance(rows,list) or len(rows)!=1411 or len(expected_diff)!=1411:
        raise RuntimeError(f"P653 row/diff count drift rows={len(rows) if isinstance(rows,list) else None} diff={len(expected_diff)}")
    normalized=[]; seen=set()
    for got in rows:
        key=ident(got)
        if key in seen: raise RuntimeError(f"duplicate P653 identity {key}")
        seen.add(key)
        if got.get("key")!=f"{key[0]}:{key[1]}:{key[2]}" or key not in expected_diff:
            raise RuntimeError(f"P653 identity/key drift {key}")
        if got.get("old")!=base[key] or got.get("new")!=final[key]:
            raise RuntimeError(f"P653 assignment semantic drift {key}")
        artifact=got.get("artifact") or {}; apath=Path(str(artifact.get("consumer_source_path_spark6", "")))
        try: apath.relative_to(P653_SOURCE_ROOT)
        except ValueError: raise RuntimeError(f"P653 artifact not under sealed spark-6 source root {key}: {apath}")
        row={"layer":key[0],"expert":key[1],"projection":key[2],"old":got["old"],"new":got["new"],
             "old_tier":got["old"],"new_tier":got["new"],"pack_fraction":1.0,"source_host":"spark-6",
             "artifact":str(apath),"artifact_bytes":int(artifact.get("bytes") or 0),"artifact_sha256":artifact.get("sha256")}
        if not row["artifact_bytes"] or not row["artifact_sha256"]:
            raise RuntimeError(f"P653 artifact authority missing {key}")
        if got["new"]=="qtip2_2.0117":
            row.update(kind="qtip2_exact",effective_method="current_qtip_inventory")
            if got.get("codebook") is not None: raise RuntimeError(f"QTIP row unexpectedly has VQ codebook {key}")
        else:
            cb=got.get("codebook") or {}; cbpath=Path(str(cb.get("consumer_source_path_spark6", "")))
            try: cbpath.relative_to(P653_SOURCE_ROOT)
            except ValueError: raise RuntimeError(f"P653 codebook not under sealed spark-6 source root {key}: {cbpath}")
            row.update(kind="genesis_vq",effective_method="reuse_sealed_p653",codebook=str(cbpath),
                       codebook_bytes=int(cb.get("bytes") or 0),codebook_sha256=cb.get("sha256"))
            if not row["codebook_bytes"] or not row["codebook_sha256"]:
                raise RuntimeError(f"P653 VQ codebook authority missing {key}")
        normalized.append(row)
    if seen!=expected_diff:
        missing=sorted(expected_diff-seen); extra=sorted(seen-expected_diff)
        raise RuntimeError(f"P653 exact assignment closure drift missing={missing[:5]} extra={extra[:5]}")
    normalized.sort(key=ident)
    kinds=Counter(r["kind"] for r in normalized)
    if kinds!={"qtip2_exact":406,"genesis_vq":1005}:
        raise RuntimeError(f"P653 kind count drift {kinds}")
    by_layer={l:[r for r in normalized if int(r["layer"])==l] for l in range(43)}
    normalized_sha=canonical(normalized)
    return {"rows":normalized,"by_layer":by_layer,
      "manifest_rows":[{"name":"P653_EXACT_ASSEMBLED_WIRE_MANIFEST","path":str(ACTIVE_MANIFEST),"sha256":observed_manifest_sha,"rows":len(normalized),"source_overlay_rows_sha256":P653_OVERLAY_ROWS_SHA}],
      "changed_cells":1411,"unchanged_cells":20605,"qtip2_cells":406,"qtip3_cells":0,"vq_cells":1005,"native_cells":0,
      "identity_set_sha256":canonical(sorted(ident(r) for r in normalized)),"coverage_layers":list(range(43)),
      "inventory_sha256":observed_manifest_sha,"assignment_sha256":FINAL_ASSIGNMENT_SHA,"final_assignment_sha256":FINAL_ASSIGNMENT_SHA,
      "active_rows_sha256":normalized_sha,"source_overlay_rows_sha256":P653_OVERLAY_ROWS_SHA,
      "source_physical_manifest":{"path":str(ACTIVE_MANIFEST),"sha256":observed_manifest_sha,"overlay_rows_sha256":P653_OVERLAY_ROWS_SHA},
      "compatibility_binding":None,"codebook_deviation_disclosure":None,"pack_fraction":1.0,"exact_wire_bytes":101346521679}

def _copy(spec: Mapping[str,Any], dst: Path) -> None:
    host=str(spec["host"]); source=str(spec["source"])
    dst.parent.mkdir(parents=True,exist_ok=True)
    if host=="spark-6": shutil.copyfile(source,dst)
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
            s={"host":r["source_host"],"source":src,"rel":f"{role}s/{r[f'{role}_sha256']}/{Path(src).name}","bytes":int(r.get(f"{role}_bytes") or Path(src).stat().st_size if r["source_host"]=="spark-6" else r.get(f"{role}_bytes") or 0),"sha256":r[f"{role}_sha256"]}
            prior=specs.setdefault(s["rel"],s)
            if prior["sha256"]!=s["sha256"] or prior["bytes"]!=s["bytes"]: raise RuntimeError(f"conflicting staged destination {s['rel']}")
            r[f"{role}_stage_rel"]=s["rel"]
    if not specs: return None,{"layer":layer,"files":0,"bytes":0,"elapsed_seconds":0.0,"stage_retired":True}
    stage=cache/f"wire_c_layer_{layer:03d}"; partial=cache/f".{stage.name}.{uuid.uuid4().hex}.partial"
    if stage.exists() or partial.exists(): raise RuntimeError(f"once-only stage exists L{layer:03d}")
    partial.mkdir(parents=True); ordered=list(specs.values()); started=time.time()
    try:
        with ThreadPoolExecutor(max_workers=min(4,len(ordered))) as ex: list(ex.map(lambda s:_copy(s,partial/s["rel"]),ordered))
        for s in ordered:
            p=partial/s["rel"]
            if not p.is_file() or (s["bytes"] and p.stat().st_size!=s["bytes"]) or sha(p)!=s["sha256"]:
                raise RuntimeError(f"staged identity drift {s['source']}")
        os.replace(partial,stage)
    except Exception:
        shutil.rmtree(partial,ignore_errors=True); raise
    total=sum((stage/s["rel"]).stat().st_size for s in ordered)
    return stage,{"layer":layer,"files":len(ordered),"bytes":total,"elapsed_seconds":time.time()-started,"transport":"spark-6 local copy; no network/NAS"}

def install_stream_source(p651:Any, base:Any, manifest:Mapping[str,Any], cache:Path, mode:str, qtip_source:Path,qtip_kernel:Path,qtip_tlut:Path):
    legacy=load_module("p885_legacy_mixed_adapter",ROOT/"code/p760_overlay_adapter.py")
    by_layer=manifest["by_layer"]; stage_rows=[]; applied={}; holder={}
    class WireCSource(base.GenesisTierSource):
        def _stage_remote(self,layer:int,row:dict)->Path: return base.GenesisTierSource._stage_remote(self,layer,row)
        def fill_layer(self,layer:int,gate_up:Any,down:Any,*extra:Any)->None:
            super().fill_layer(layer,gate_up,down,*extra); rows=by_layer[layer]
            if not rows:
                stage_rows.append({"layer":layer,"changed_cells":0,"files":0,"bytes":0,"elapsed_seconds":0.0,"stage_retired":True}); return
            stage,transfer=stage_layer(layer,rows,cache)
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
                        allowed=(meta.get("schema")=="wire-c-preview-canonical-vq-cell-v1" and meta.get("assignment_sha256")==FINAL_ASSIGNMENT_SHA and meta.get("builder_sha256")==VQ_BUILDER_SHA) or (meta.get("schema")=="p640-genesis-vq-overlay-cell-v1" and meta.get("assignment_sha256")==P653_ASSIGNMENT_SHA and meta.get("canonical_builder_sha256")==VQ_BUILDER_SHA) or (meta.get("schema") in ("p892-wire-c-vq-overlay-cell-v1","p892-wire-c-vq-cell-v1") and meta.get("task_id") in ("task-redacted","task-redacted","task-redacted","task-redacted","task-redacted") and (meta.get("assignment_sha256"),meta.get("assignment_map_sha256")) in ((F949_ASSIGNMENT_SHA,F949_MAP_SHA),(FINAL_ASSIGNMENT_SHA,FINAL_MAP_SHA)) and meta.get("canonical_builder_sha256")==VQ_BUILDER_SHA and meta.get("pack_fraction",1.0)==1.0) or (meta.get("schema")=="p760-genesis-vq-overlay-cell-v1" and meta.get("task_id")=="task-redacted" and meta.get("assignment_sha256")=="d791614b1f9c2a3ceeac1635e1dfd63f17f135fb31f18a9e08c38c034f4a4935" and meta.get("builder_sha256")==VQ_BUILDER_SHA) or (meta.get("schema")=="p897-wire-c-fortress-vq-cell-v1" and meta.get("task_id")=="task-redacted" and meta.get("final_assignment_sha256")==FINAL_ASSIGNMENT_SHA and meta.get("final_assignment_map_sha256")==FINAL_MAP_SHA and meta.get("canonical_builder_sha256")==VQ_BUILDER_SHA and meta.get("pack_fraction")==1.0)
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
            transfer.update({"schema":"p921-wire-a-layer-consumption-v1","status":"PASS","task_id":TASK,"mode":mode,"changed_cells":len(rows),"kinds":dict(Counter(r["kind"] for r in rows)),"stage_retired":stage is None or not stage.exists()}); stage_rows.append(transfer)
            p651.atomic_json(ROOT/f"run/{os.environ['P885_RUN_ID']}/LAYER_{layer:03d}_OVERLAY.json",transfer)
    return WireCSource,{"stage_rows":stage_rows,"applied":applied,"decoder_holder":holder}
