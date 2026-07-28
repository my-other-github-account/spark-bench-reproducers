#!/usr/bin/env python3
import hashlib,json,os,pathlib,time
MISSION=pathlib.Path('${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8')
EXPECTED_DATASET='42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f'
EXPECTED_CHECKPOINT='4086e9d8be9ece067ce3b713c22654e59bcad614af9444bdfacd2e66e0a02fd5'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p,text):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(text);os.replace(t,p)
dataset=MISSION/'assets/data/HumanEvalPlus-v0.1.10.jsonl';assert sha(dataset)==EXPECTED_DATASET
manifest_path=MISSION/'results/generation/GENERATION_MANIFEST.json';manifest=json.loads(manifest_path.read_text())
assert manifest['status']=='PASS' and manifest['sealed_rows']==164 and manifest['checkpoint_sha256']==EXPECTED_CHECKPOINT
rows=[]
for i,item in enumerate(manifest['rows']):
 tid=f'HumanEval/{i}';assert item['task_id']==tid
 p=pathlib.Path(item['raw_path']);assert p.is_file() and sha(p)==item['raw_sha256']
 raw=json.loads(p.read_text());assert raw['task_id']==tid
 message=raw['response']['choices'][0].get('message') or {};content=message.get('content')
 solution=content if isinstance(content,str) else "raise RuntimeError('null visible answer')\n"
 rows.append({'task_id':tid,'solution':solution})
out=MISSION/'results/evalplus';out.mkdir(parents=True,exist_ok=True)
raw_solutions=out/'raw_solutions.jsonl';atomic(raw_solutions,''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows))
receipt={'schema':'clean-he164-transfer8-evalplus-input-v1','status':'PASS','tasks':164,'checkpoint_sha256':EXPECTED_CHECKPOINT,'dataset_sha256':EXPECTED_DATASET,'generation_manifest':str(manifest_path),'generation_manifest_sha256':sha(manifest_path),'raw_solutions':str(raw_solutions),'raw_solutions_sha256':sha(raw_solutions),'created_epoch':time.time()}
p=out/'INPUT_RECEIPT.json';atomic(p,json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','tasks':164,'raw_solutions_sha256':sha(raw_solutions),'receipt_sha256':sha(p)},sort_keys=True))
