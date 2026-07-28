#!/usr/bin/env python3
import hashlib,json,os,pathlib
from evalplus.sanitize import sanitize
DATASET=pathlib.Path(os.environ.get('HUMANEVAL_OVERRIDE_PATH','/work/HumanEvalPlus-v0.1.10.jsonl'))
RAW=pathlib.Path('/work/raw_solutions.jsonl')
OUT=pathlib.Path('/work/samples.jsonl')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(DATASET)=='42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f'
dataset={x['task_id']:x for x in (json.loads(line) for line in DATASET.read_text().splitlines() if line.strip())}
raw=[json.loads(line) for line in RAW.read_text().splitlines() if line.strip()]
assert [x['task_id'] for x in raw]==[f'HumanEval/{i}' for i in range(164)]
rows=[]
for x in raw:
 rows.append({'task_id':x['task_id'],'solution':sanitize(x['solution'],entrypoint=dataset[x['task_id']]['entry_point'])})
OUT.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in rows))
print(json.dumps({'status':'PASS','sanitized':len(rows),'samples_sha256':sha(OUT)},sort_keys=True))
