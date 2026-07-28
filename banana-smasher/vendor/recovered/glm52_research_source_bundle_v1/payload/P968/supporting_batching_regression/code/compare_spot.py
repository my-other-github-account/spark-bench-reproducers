#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
M=Path('${SPARK_HOME}/missions/IQ3_BATCH_INVARIANCE_t_fafb2fe1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,o):
 p=Path(p);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with tmp.open('w') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
pre=json.loads((M/'receipts/PRE_REGISTRATION.json').read_text());mans={a:json.loads((M/f'results/arm_{a}/GENERATION_MANIFEST.json').read_text()) for a in 'AB'}
for a,m in mans.items():
 if m.get('status')!='PASS' or m['ordered_task_ids']!=pre['ordered_task_ids'] or m['dataset_sha256']!=pre['dataset_sha256'] or m['model_receipt_sha256']!=pre['model_receipt_sha256']:raise RuntimeError(f'manifest identity drift arm {a}')
ra={r['task_id']:r for r in mans['A']['rows']};rb={r['task_id']:r for r in mans['B']['rows']};rows=[];diff={k:[] for k in ['visible_bytes','visible_sha256','finish_reason','completion_tokens','retokenized_visible_tokens','content_is_null','response_model','prompt_fingerprints','system_fingerprint']}
for tid in pre['ordered_task_ids']:
 pa=json.loads(Path(ra[tid]['raw_path']).read_text());pb=json.loads(Path(rb[tid]['raw_path']).read_text());ca=(pa['response']['choices'][0].get('message') or {}).get('content');cb=(pb['response']['choices'][0].get('message') or {}).get('content');ba=('' if ca is None else ca).encode();bb=('' if cb is None else cb).encode();eq={'visible_bytes':ba==bb,'visible_sha256':ra[tid]['visible_sha256']==rb[tid]['visible_sha256'],'finish_reason':ra[tid]['finish_reason']==rb[tid]['finish_reason'],'completion_tokens':ra[tid]['completion_tokens']==rb[tid]['completion_tokens'],'retokenized_visible_tokens':ra[tid]['retokenized_visible_tokens']==rb[tid]['retokenized_visible_tokens'],'content_is_null':ra[tid]['content_is_null']==rb[tid]['content_is_null'],'response_model':ra[tid]['response_model']==rb[tid]['response_model'],'prompt_fingerprints':ra[tid]['prompt_fingerprints']==rb[tid]['prompt_fingerprints'],'system_fingerprint':ra[tid]['system_fingerprint']==rb[tid]['system_fingerprint']}
 for k,v in eq.items():
  if not v:diff[k].append(tid)
 rows.append({'task_id':tid,'equal':eq,'arm_A':ra[tid],'arm_B':rb[tid]})
semantic_keys=['visible_bytes','finish_reason','completion_tokens','retokenized_visible_tokens','content_is_null','response_model','prompt_fingerprints'];different=sorted(set(sum((diff[k] for k in semantic_keys),[])),key=pre['ordered_task_ids'].index)
out={'schema':'iq3-batching-invariance-comparison-v1','status':'PASS','task':'task-redacted','ordered_task_ids':pre['ordered_task_ids'],'pre_registration_sha256':sha(M/'receipts/PRE_REGISTRATION.json'),'arm_A_manifest_sha256':sha(M/'results/arm_A/GENERATION_MANIFEST.json'),'arm_B_manifest_sha256':sha(M/'results/arm_B/GENERATION_MANIFEST.json'),'contract':pre['contract'],'batching_invariant_visible_and_tokens':not different,'different_row_ids':different,'differences_by_field':diff,'runtime_fingerprint_note':'system_fingerprint is compared and reported but excluded from semantic batching-invariance decision; server parallel/client concurrency intentionally differ','rows':rows,'created_epoch':time.time()};atomic(M/'receipts/GENERATION_COMPARISON.json',out);print(json.dumps({'status':'PASS','batching_invariant':out['batching_invariant_visible_and_tokens'],'different_row_ids':different,'differences_by_field':diff,'receipt_sha256':sha(M/'receipts/GENERATION_COMPARISON.json')},sort_keys=True))
