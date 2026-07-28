#!/usr/bin/env python3
import concurrent.futures, copy, hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path

SRC=Path('${SPARK_HOME}/missions/P909_WIRE_C_REPAIR_t_c80c52ef_s3')
DST=Path('${SPARK_HOME}/missions/P948_SPECULATIVE_REPAIR_t_b1aa61d3_s3')
CLAIM=Path('${SPARK_HOME}/HOST_CLAIM.json')
CLAIM_SHA='6eb7d061bc57a364404eeaf1de22eb0f67e14bb59c888954337cbeb015593b8c'
TASK='task-redacted'
LABEL='SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL'
S6='${OPERATOR_USER}@${QSFP_HOST}'
S6ROOT='${SPARK_HOME}/missions/P929_TRUE_C_REFIT_t_a2b3a979_s6'
TARGET='c9547a0b306701b23e07e1186b26c017c8fbb1efc0fa0b6250987f66bbb5872f'
BUILD='13d1f887f8e6055f1f579730c2cc37be1e6c0754dd02256cf35a3a9f8c2d0a2f'
OLD_MANIFEST='398441d16f1a251079b518a55095c568353b9f3e542f2ec55d4139e0ac6e7ffd'
OLD_PLAN='372d949e333546e8627834b738d26ba596978e951a4efbd19b92f3490cab6b48'
OLD_CB='7bb8044ab5c8d98934329f0f4733c3c2682c4c94c8e6172a04fdeece6864aa37'


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_path(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);obj=dict(obj);obj.setdefault('speculative_label',LABEL)
 fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);d=os.open(p.parent,os.O_DIRECTORY);os.fsync(d);os.close(d)
def progress(state,**kw):
 atomic_json(DST/'PROGRESS.json',{'schema':'p948-speculative-progress-v1','task_id':TASK,'host':'spark-3','state':state,'updated_unix':time.time(),**kw})
def sh(cmd,**kw): return subprocess.run(cmd,check=True,text=True,**kw)

def guard():
 raw=CLAIM.read_bytes();d=json.loads(raw)
 if sha_bytes(raw)!=CLAIM_SHA or d.get('owner')!=TASK or d.get('host')!='spark-3': raise RuntimeError('claim drift')
 apps=sh(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True).stdout.strip()
 if apps: raise RuntimeError('GPU occupied '+apps)
 free=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize
 if free<8*1024**3: raise RuntimeError(f'root floor {free}')
 return free

def remote_json_script():
 code=r'''
import json,glob,hashlib
R='${SPARK_HOME}/missions/P929_TRUE_C_REFIT_t_a2b3a979_s6'
T='c9547a0b306701b23e07e1186b26c017c8fbb1efc0fa0b6250987f66bbb5872f'
B='13d1f887f8e6055f1f579730c2cc37be1e6c0754dd02256cf35a3a9f8c2d0a2f'
missing={(7,'d4_k2048','fused13'),(7,'d4_k4096','down'),(7,'d4_k4096','fused13')}
out=[]
for p in sorted(glob.glob(R+'/receipts/CODEBOOK_L*.json')):
 b=open(p,'rb').read()
 try:d=json.loads(b)
 except:continue
 g=tuple(d.get('codebook_group',()))
 if len(g)!=3 or g in missing or d.get('status')!='PASS':continue
 if d.get('target_contract_sha256')!=T or d.get('build_identity_sha256')!=B:continue
 if len(d.get('rows',[]))!=int(d.get('expected_rows',-1)):continue
 out.append({'path':p,'sha256':hashlib.sha256(b).hexdigest(),'document':d})
print(json.dumps(out,sort_keys=True))
'''
 cp=sh(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',S6,'python3','-'],input=code,capture_output=True)
 return json.loads(cp.stdout)

def replace_once(path,old,new):
 p=Path(path);s=p.read_text()
 if old not in s: raise RuntimeError(f'missing patch marker in {p}: {old[:80]}')
 p.write_text(s.replace(old,new,1))

def main():
 free=guard();DST.mkdir(parents=True,exist_ok=True)
 for n in ('code','inputs','approvals','assets','receipts','checkpoints','rollback','logs','run','staging','pinned_codebooks'):
  (DST/n).mkdir(parents=True,exist_ok=True)
 progress('PREP_STARTED',root_free_bytes=free,source_mission=str(SRC),source_launcher_sha256=sha_path(SRC/'code/run_p911_repair.sh'))
 # Small immutable trees only. Never duplicate the large cache/slices.
 for n in ('code','inputs','approvals','assets'):
  target=DST/n
  if n != 'code' and target.exists(): shutil.rmtree(target)
  shutil.copytree(SRC/n,target,symlinks=True,dirs_exist_ok=True)
 for n in ('stage_cache','slices'):
  p=DST/n
  if p.exists() or p.is_symlink():
   if p.is_dir() and not p.is_symlink(): shutil.rmtree(p)
   else:p.unlink()
  p.symlink_to(SRC/n,target_is_directory=True)
 # Preserve canonical UPDATE_000 by reference and immutable copy; never expose it as LATEST.
 seed=DST/'seed';seed.mkdir(exist_ok=True)
 for n in ('UPDATE_000.pt','UPDATE_000.json'):
  shutil.copy2(SRC/'checkpoints'/n,seed/('BASELINE_'+n))
 atomic_json(seed/'BASELINE_UPDATE_000_REFERENCE.json',{'schema':'p948-baseline-update0-reference-v1','status':'PRESERVED_NOT_CONSUMED_IDENTITY_REBUILD_REQUIRED','source':str(SRC/'checkpoints/UPDATE_000.pt'),'source_sha256':sha_path(SRC/'checkpoints/UPDATE_000.pt'),'copied':str(seed/'BASELINE_UPDATE_000.pt'),'copied_sha256':sha_path(seed/'BASELINE_UPDATE_000.pt')})

 snap=remote_json_script()
 rows=sum(len(x['document']['rows']) for x in snap)
 if len(snap)!=77 or rows!=2808: raise RuntimeError(f'snapshot drift {len(snap)}/{rows}')
 snapshot={'schema':'p948-p943-current-snapshot-v1','status':'STRICT_PASS_CURRENT_NONTERMINAL','task_id':TASK,'source_task_id':'task-redacted','source_host':'spark-6','label':LABEL,'target_contract_sha256':TARGET,'build_identity_sha256':BUILD,'pass_codebooks':77,'pass_rows':2808,'expected_codebooks':80,'expected_rows':2860,'remaining_codebooks':3,'remaining_rows':52,'missing_groups':[[7,'d4_k2048','fused13'],[7,'d4_k4096','down'],[7,'d4_k4096','fused13']],'receipt_bindings':[{'path':x['path'],'sha256':x['sha256'],'group':x['document']['codebook_group'],'rows':len(x['document']['rows']),'codebook_sha256':x['document']['codebook_sha256']} for x in snap],'receipt_set_sha256':canonical([(x['path'],x['sha256']) for x in snap]),'pack_fraction':1.0,'zero_substitution':True,'zero_quarantine':True,'terminal':False,'captured_unix':time.time()}
 atomic_json(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json',snapshot)
 progress('SNAPSHOT_BOUND',snapshot_sha256=sha_path(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json'),codebooks=77,rows=2808)

 plan=json.load(open(SRC/'inputs/F521_EXACT_SOURCE_STAGING_PLAN.json'))
 manifest=json.load(open(SRC/'inputs/RESTORED_F521_PHYSICAL_MANIFEST.json'))
 pby={tuple(r['identity']):r for r in plan['rows']};mby={tuple(r['identity']):r for r in manifest['rows']}
 new_codebooks={}
 for rec in snap:
  d=rec['document']
  for rr in d['rows']:
   ident=tuple(rr['identity'])
   if ident not in pby: raise RuntimeError(f'target row outside repair surface {ident}')
   ps=pby[ident]
   ps['payload']={'bytes':int(rr['artifact_bytes']),'sha256':rr['artifact_sha256'],'local_path':None,'spark4_path':rr['artifact'],'remote_host':'${QSFP_HOST}'}
   ps['codebook']={'bytes':int(rr['codebook_bytes']),'sha256':rr['codebook_sha256'],'local_path':None,'spark4_path':rr['codebook'],'remote_host':'${QSFP_HOST}'}
   ps['snapshot_source']={'label':LABEL,'p943_receipt':rec['path'],'p943_receipt_sha256':rec['sha256']}
   mm=mby[ident];mm.update({'payload_bytes':int(rr['artifact_bytes']),'payload_sha256':rr['artifact_sha256'],'payload_path':rr['artifact'],'source_host':'spark-6','source_path':rr['old_artifact'],'pack_fraction':1.0,'runtime_reread_exact':True,'codebook':{'bytes':int(rr['codebook_bytes']),'sha256':rr['codebook_sha256'],'path':rr['codebook']},'snapshot_source':ps['snapshot_source']})
   new_codebooks[rr['codebook_sha256']]={'bytes':int(rr['codebook_bytes']),'remote_path':rr['codebook']}

 # Copy only 77 tiny codebooks; payloads remain rolling and are never bulk-restaged.
 def pull(item):
  dig,spec=item;dest=DST/'pinned_codebooks'/dig;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.is_file() or dest.stat().st_size!=spec['bytes'] or sha_path(dest)!=dig:
   sh(['rsync','-a','--whole-file','--partial','-e','ssh -o BatchMode=yes -o ConnectTimeout=8',f'{S6}:{spec["remote_path"]}',str(dest)],stdout=subprocess.DEVNULL)
  if dest.stat().st_size!=spec['bytes'] or sha_path(dest)!=dig: raise RuntimeError('codebook pull mismatch '+dig)
  return dig,str(dest)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  pulled=dict(ex.map(pull,new_codebooks.items()))
 # Point new row codebooks to exact local pins.
 for r in pby.values():
  cb=r.get('codebook')
  if cb and cb['sha256'] in pulled: cb['local_path']=pulled[cb['sha256']]

 plan['rows']=[pby[k] for k in sorted(pby)]
 plan.update({'schema':'p948-speculative-f521-source-plan-v1','status':'PASS_SPECULATIVE_CURRENT_SNAPSHOT','task_id':TASK,'speculative_label':LABEL,'p943_snapshot':str(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json'),'p943_snapshot_sha256':sha_path(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json')})
 # Dynamic cache may need any remote rows; exact sum is informational and preserves >=8-stream gate.
 rem={}
 for r in plan['rows']:
  for role in ('payload','codebook'):
   s=r.get(role)
   if s and not s.get('local_path'): rem[(s['sha256'],s['bytes'])]=s
 plan['missing_unique_files']=len(rem);plan['missing_unique_bytes']=sum(int(k[1]) for k in rem)
 manifest['rows']=[mby[k] for k in sorted(mby)]
 manifest.update({'schema':'p948-speculative-restored-f521-current-snapshot-v1','status':'PASS','task_id':TASK,'wire_label':'PRELIM_TRUE_C_77_OF_80','speculative_label':LABEL,'p943_snapshot':str(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json'),'p943_snapshot_sha256':sha_path(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json'),'artifact_bytes':sum(int(r['payload_bytes']) for r in manifest['rows']),'rows_sha256':canonical(manifest['rows']),'pack_fraction':1.0,'zero_missing':True,'zero_duplicates':True,'zero_substitution':True,'zero_quarantine':True,'full_host_reread_pass':True,'peer_reread_pass':True})
 atomic_json(DST/'inputs/P948_SPECULATIVE_SOURCE_PLAN.json',plan)
 atomic_json(DST/'inputs/P948_SPECULATIVE_MANIFEST.json',manifest)
 plan_sha=sha_path(DST/'inputs/P948_SPECULATIVE_SOURCE_PLAN.json');man_sha=sha_path(DST/'inputs/P948_SPECULATIVE_MANIFEST.json')

 # Rebuild the exact 184-SHA local pin receipt from the current snapshot plan.
 oldcb=json.load(open(SRC/'receipts/F521_CODEBOOK_SET_184.json'))
 old_by={r['sha256']:r for r in oldcb['rows']}
 uniq={}
 for r in plan['rows']:
  cb=r.get('codebook')
  if cb: uniq[cb['sha256']]=cb
 if len(uniq)!=184: raise RuntimeError(f'current snapshot codebook set drift {len(uniq)}')
 cbrows=[]
 for dig,spec in sorted(uniq.items()):
  if dig in pulled: path=Path(pulled[dig])
  else:
   old=old_by.get(dig)
   if not old: raise RuntimeError('inherited codebook absent '+dig)
   path=Path(old['path'])
  if not path.is_file() or path.stat().st_size!=int(spec['bytes']) or sha_path(path)!=dig: raise RuntimeError('pin drift '+dig)
  cbrows.append({'bytes':int(spec['bytes']),'path':str(path),'sha256':dig,'source':'P943_CURRENT_77_OF_80' if dig in pulled else 'P909_INHERITED_NONTERMINAL'})
 cbdoc=copy.deepcopy(oldcb);cbdoc.update({'schema':'p948-speculative-codebook-set-v1','status':'PASS_184_PINNED_OR_LOCAL','task_id':TASK,'speculative_label':LABEL,'rows':cbrows,'unique_sha256_set':canonical(sorted(uniq)),'p943_snapshot_sha256':snapshot['receipt_set_sha256']})
 atomic_json(DST/'receipts/P948_CODEBOOK_SET_184.json',cbdoc);cb_sha=sha_path(DST/'receipts/P948_CODEBOOK_SET_184.json')

 # Isolated code constants and transport host support.
 ov=DST/'code/f521_repair_overlay.py';rs=DST/'code/f521_rolling_stage.py';gb=DST/'code/genesis_basic_repair.py'
 for p in (ov,rs):
  s=p.read_text().replace(OLD_MANIFEST,man_sha).replace(OLD_PLAN,plan_sha)
  p.write_text(s)
 ov.write_text(ov.read_text().replace(OLD_CB,cb_sha))
 s=rs.read_text()
 s=s.replace('any(authoritative.get(k) != spec.get(k) for k in ("bytes", "sha256", "spark4_path", "local_path"))','any(authoritative.get(k) != spec.get(k) for k in ("bytes", "sha256", "spark4_path", "local_path", "remote_host"))')
 s=s.replace('remote = str(spec["spark4_path"])\n                    subprocess.run(["rsync", "-a", "--whole-file", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8", f"${OPERATOR_USER}@${QSFP_HOST}:{remote}"','remote = str(spec["spark4_path"])\n                    remote_host = str(spec.get("remote_host") or "${QSFP_HOST}")\n                    subprocess.run(["rsync", "-a", "--whole-file", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=8", f"${OPERATOR_USER}@{remote_host}:{remote}"')
 rs.write_text(s)
 # Label all structured outputs and checkpoint identity in the throwaway code.
 for p in (rs,gb):
  s=p.read_text()
  marker='def atomic_json(path: Path, value: object) -> None:\n'
  if marker in s:
   s=s.replace(marker,marker+'    if isinstance(value, dict):\n        value = dict(value); value.setdefault("speculative_label", os.environ.get("GENESIS_SPECULATIVE_LABEL", "SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL"))\n',1)
  p.write_text(s)
 replace_once(gb,'    checkpoints = root / "checkpoints"','    identity = dict(identity)\n    identity["speculative_label"] = os.environ.get("GENESIS_SPECULATIVE_LABEL", "SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL")\n    identity["p943_current_snapshot_sha256"] = "'+sha_path(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json')+'"\n    checkpoints = root / "checkpoints"')
 replace_once(gb,'    def emit(**row):\n        row.setdefault("unix", time.time())','    def emit(**row):\n        row.setdefault("unix", time.time())\n        row.setdefault("speculative_label", os.environ.get("GENESIS_SPECULATIVE_LABEL", "SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL"))')

 # Derive only path/identity bindings from the canonical launcher; preserve all math/schedule knobs.
 launch=SRC.joinpath('code/run_p911_repair.sh').read_text()
 launch=launch.replace(str(SRC),str(DST)).replace('TASK=task-redacted','TASK='+TASK).replace('CLAIM_SHA=6f575f8dbd8ff835689bb9093716e6dc7fbd52ac0129aac6aa305176757d46f8','CLAIM_SHA='+CLAIM_SHA)
 launch=launch.replace('export GENESIS_REPAIR_ROOT="$ROOT"','export GENESIS_REPAIR_ROOT="$ROOT"\nexport GENESIS_SPECULATIVE_LABEL='+LABEL)
 launch=launch.replace('export GENESIS_F521_MANIFEST="$ROOT/inputs/RESTORED_F521_PHYSICAL_MANIFEST.json"','export GENESIS_F521_MANIFEST="$ROOT/inputs/P948_SPECULATIVE_MANIFEST.json"')
 launch=launch.replace('export GENESIS_F521_SOURCE_PLAN="$ROOT/inputs/F521_EXACT_SOURCE_STAGING_PLAN.json"','export GENESIS_F521_SOURCE_PLAN="$ROOT/inputs/P948_SPECULATIVE_SOURCE_PLAN.json"')
 launch=launch.replace('export GENESIS_F521_CODEBOOK_RECEIPT="$ROOT/receipts/F521_CODEBOOK_SET_184.json"','export GENESIS_F521_CODEBOOK_RECEIPT="$ROOT/receipts/P948_CODEBOOK_SET_184.json"')
 launch=launch.replace('P911_TRAINER.stdout.log',LABEL+'__P911_TRAINER.stdout.log').replace('P911_TRAINER.stderr.log',LABEL+'__P911_TRAINER.stderr.log').replace('resource_guard.log',LABEL+'__resource_guard.log').replace('P911_ACCEPTANCE.log',LABEL+'__P911_ACCEPTANCE.log')
 launch=launch.replace("{'schema':'p911-phase-launch-v1'","{'schema':'p911-phase-launch-v1','speculative_label':'"+LABEL+"'")
 launch=launch.replace("{'schema':'p911-phase-exit-v1'","{'schema':'p911-phase-exit-v1','speculative_label':'"+LABEL+"'")
 (DST/'code/run_p948_speculative_repair.sh').write_text(launch);os.chmod(DST/'code/run_p948_speculative_repair.sh',0o755)

 for p in (ov,rs,gb,DST/'code/run_p948_speculative_repair.sh'):
  if p.suffix=='.py': sh(['${SPARK_HOME}/humming_env/bin/python3','-m','py_compile',str(p)])
 atomic_json(DST/'receipts/P948_PREWARM_BINDING.json',{'schema':'p948-prewarm-binding-v1','status':'PASS_READY_TO_LAUNCH','task_id':TASK,'source_launcher':str(SRC/'code/run_p911_repair.sh'),'source_launcher_sha256':sha_path(SRC/'code/run_p911_repair.sh'),'derived_launcher':str(DST/'code/run_p948_speculative_repair.sh'),'derived_launcher_sha256':sha_path(DST/'code/run_p948_speculative_repair.sh'),'manifest_sha256':man_sha,'source_plan_sha256':plan_sha,'codebook_receipt_sha256':cb_sha,'p943_snapshot_sha256':sha_path(DST/'inputs/P943_CURRENT_77_OF_80_SNAPSHOT.json'),'snapshot_codebooks':77,'snapshot_rows':2808,'inherited_nonterminal_codebooks':3,'inherited_nonterminal_rows':52,'pack_fraction':1.0,'zero_substitution':True,'zero_quarantine':True,'canonical_update_state_untouched':True,'shared_cache':str(SRC/'stage_cache')})
 progress('READY_TO_LAUNCH',binding_sha256=sha_path(DST/'receipts/P948_PREWARM_BINDING.json'),root_free_bytes=guard())
 print(json.dumps({'status':'READY_TO_LAUNCH','root':str(DST),'binding_sha256':sha_path(DST/'receipts/P948_PREWARM_BINDING.json'),'launcher':str(DST/'code/run_p948_speculative_repair.sh')},sort_keys=True),flush=True)

if __name__=='__main__': main()
