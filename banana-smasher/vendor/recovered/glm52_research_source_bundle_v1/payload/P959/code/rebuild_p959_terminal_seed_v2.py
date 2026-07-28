#!/usr/bin/env python3
import ast,copy,hashlib,json,os,re,subprocess,tempfile,time
from pathlib import Path
import torch
R=Path('${SPARK_HOME}/missions/P959_TRUE_C_REPAIR_t_8343707a_s3');C=Path('${SPARK_HOME}/HOST_CLAIM.json');TASK='task-redacted';P943='90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,o):
 p=Path(p);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
 with os.fdopen(fd,'w') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def atomic_torch(p,o):
 p=Path(p);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent);os.close(fd)
 try:
  torch.save(o,t);f=os.open(t,os.O_RDONLY);os.fsync(f);os.close(f);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
raw=C.read_bytes();claim_sha=hashlib.sha256(raw).hexdigest();claim=json.loads(raw);assert claim.get('owner')==TASK
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().splitlines();assert not apps,apps
ps=subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True).stdout.splitlines();scoped=[x for x in ps if str(os.getpid()) not in x and any(y in x for y in ['run_p959_controller.sh','run_p959_terminal_repair.sh','genesis_basic_repair.py'])];assert not scoped,scoped
marker='RuntimeError: terminal seed extension requires explicit gate: '
rows=[x[len(marker):] for x in (R/'logs/P911_TRAINER.stderr.log').read_text(errors='replace').splitlines() if x.startswith(marker)]
assert rows;observed=ast.literal_eval(rows[-1]);inherited={k for vals in observed.values() for k in vals};assert len(inherited)==12 and set(observed)=={'L0','L1','L2'},observed
basep=R/'seed/BASELINE_UPDATE_000.pt';termp=R/'seed/TERMINAL_UPDATE_000.pt';base=torch.load(basep,map_location='cpu',mmap=True,weights_only=False);old=torch.load(termp,map_location='cpu',mmap=True,weights_only=False)
basekeys={k for d in base['state']['codebooks'].values() for k in d};oldkeys={k for d in old['state']['codebooks'].values() for k in d};assert len(basekeys)==186 and len(oldkeys)==184
assert inherited <= basekeys and not (inherited & oldkeys)
rebuilt=copy.deepcopy(old)
for label,keys in observed.items():
 for key in keys:
  t=base['state']['codebooks'][label][key].detach().cpu().clone();assert hashlib.sha256(t.to(torch.float16).numpy().tobytes()).hexdigest()==key.rsplit('_',1)[-1];rebuilt['state']['codebooks'][label][key]=t
finalkeys={k for d in rebuilt['state']['codebooks'].values() for k in d};assert finalkeys==oldkeys|inherited and len(finalkeys)==196
rebuilt['identity']['terminal_seed_rebuild'].update({'complete_runtime_codebook_count':196,'p943_changed_surface_codebook_count':184,'unchanged_base_inherited_codebook_count':12,'unchanged_base_inherited_keys':observed,'source':'exact runtime live-key negative-gate inventory + canonical unchanged-codebook fp16 SHA authority','old_to_new_alias_mapping_used':False})
stamp=str(int(time.time()));backup=R/'rollback'/('TERMINAL_UPDATE_000.pt.pre_complete_inventory_'+stamp);os.replace(termp,backup);atomic_torch(termp,rebuilt)
check=torch.load(termp,map_location='cpu',mmap=True,weights_only=False);assert {k for d in check['state']['codebooks'].values() for k in d}==finalkeys
rp=R/'receipts/P959_TERMINAL_UPDATE_000_SEED_REBUILD.json';prior_sha=sha(rp);receipt={'schema':'p959-terminal-update-000-seed-rebuild-v2','status':'PASS_TERMINAL_UPDATE_000_REBUILT_FROM_WIRE','task_id':TASK,'claim_sha256':claim_sha,'p943_terminal_sha256':P943,'canonical_seed':str(basep),'canonical_seed_sha256':sha(basep),'terminal_seed':str(termp),'terminal_seed_sha256':sha(termp),'terminal_seed_bytes':termp.stat().st_size,'prior_incomplete_terminal_seed':str(backup),'prior_incomplete_terminal_seed_sha256':sha(backup),'prior_rebuild_receipt_sha256':prior_sha,'canonical_codebook_count':len(basekeys),'p943_changed_surface_codebook_count':len(oldkeys),'unchanged_base_inherited_codebook_count':len(inherited),'terminal_codebook_count':len(finalkeys),'shared_exact_count':len(basekeys&finalkeys),'terminal_added_from_authority_count':len(finalkeys-basekeys),'canonical_dropped_count':len(basekeys-finalkeys),'unchanged_base_inherited_keys':observed,'terminal_keyspace_exact':True,'terminal_values_exact':True,'non_codebook_state_inherited_exact':True,'optimizer_scheduler_will_reset_on_state_only_load':True,'old_to_new_alias_mapping_used':False,'training_updates_run':0,'speculative_seed_used':False,'runtime_negative_gate_stderr_sha256':sha(R/'logs/P911_TRAINER.stderr.log'),'gpu_apps':apps,'scoped_processes':scoped,'sealed_unix':time.time()};atomic_json(rp,receipt)
print(json.dumps({'status':receipt['status'],'terminal_seed_sha256':sha(termp),'receipt_sha256':sha(rp),'counts':{'canonical':len(basekeys),'p943_changed':len(oldkeys),'inherited_unchanged':len(inherited),'complete_terminal':len(finalkeys),'shared':len(basekeys&finalkeys),'terminal_only':len(finalkeys-basekeys),'dropped':len(basekeys-finalkeys)}},sort_keys=True))
