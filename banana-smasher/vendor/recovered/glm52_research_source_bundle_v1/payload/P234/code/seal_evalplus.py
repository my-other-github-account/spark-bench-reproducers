#!/usr/bin/env python3
import hashlib,json,os,pathlib,time
MISSION=pathlib.Path('${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8')
OUT=MISSION/'results/evalplus'
CHECKPOINT='4086e9d8be9ece067ce3b713c22654e59bcad614af9444bdfacd2e66e0a02fd5'
DATASET='42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,obj):
 p=pathlib.Path(p);t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def atomic_text(p,text):
 p=pathlib.Path(p);t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(text);os.replace(t,p)
manifest_path=MISSION/'results/generation/GENERATION_MANIFEST.json';manifest=json.loads(manifest_path.read_text());assert manifest['sealed_rows']==164
samples=OUT/'samples.jsonl';sample_rows=[json.loads(x) for x in samples.read_text().splitlines() if x.strip()];assert [x['task_id'] for x in sample_rows]==[f'HumanEval/{i}' for i in range(164)]
results_path=OUT/'samples.eval_results.json';results=json.loads(results_path.read_text());ev=results['eval'];assert set(ev)==set(x['task_id'] for x in sample_rows)
per_task=[];base_pass=[];plus_pass=[];both_pass=[]
for i,item in enumerate(manifest['rows']):
 tid=f'HumanEval/{i}';assert item['task_id']==tid and len(ev[tid])==1
 e=ev[tid][0];base=e.get('base_status')=='pass';plus=e.get('plus_status')=='pass';both=base and plus
 if base:base_pass.append(tid)
 if plus:plus_pass.append(tid)
 if both:both_pass.append(tid)
 solution=sample_rows[i]['solution']
 per_task.append({'task_id':tid,'finish_reason':item['finish_reason'],'completion_tokens':item['completion_tokens'],'content_is_null':item['content_is_null'],'raw_sha256':item['raw_sha256'],'sanitized_solution_sha256':hashlib.sha256(solution.encode()).hexdigest(),'base_status':e.get('base_status'),'plus_status':e.get('plus_status'),'pass_base':base,'pass_plus':plus,'pass_both':both,'evalplus':e})
claim_path=pathlib.Path('${SPARK_HOME}/HOST_CLAIM.json');claim=json.loads(claim_path.read_text());assert claim.get('owner')=='task-redacted'
script_names=['claim_spark8.py','stage_assets.py','run_step8_server.sh','START_SERVER.sh','drop_cache_loop.sh','he164_generate.py','START_GENERATE.sh','first_receipt.py','prepare_evalplus.py','sanitize_evalplus.py','evalplus_codegen_pinned.py','SCORE_EVALPLUS.sh','seal_evalplus.py','STOP_RUNTIME.sh','release_spark8.py']
script_hashes={name:sha(MISSION/'code'/name) for name in script_names}
counts={'base':len(base_pass),'plus':len(plus_pass),'both':len(both_pass),'denominator':164}
science_claim={
 'metric':'humaneval_base_pass_at_1',
 'threshold':160,
 'observed':counts['base'],
 'passes':counts['base']>=160,
 'verdict':'PASS_GE_160_OF_164' if counts['base']>=160 else 'FAIL_LT_160_OF_164',
 'comparison_references':{
  'sealed_step0_base_passes':161,
  'ud_iq4_xs_base_passes':161,
  'delta_vs_sealed_step0':counts['base']-161,
  'delta_vs_ud_iq4_xs':counts['base']-161,
 },
}
verdict={'schema':'clean-he164-transfer8-evalplus-verdict-v1','status':'PASS','execution_status':'PASS','science_claim':science_claim,'task':'task-redacted','host':'spark-8','checkpoint_sha256':CHECKPOINT,'dataset':{'sha256':DATASET,'evalplus_name':'HumanEvalPlus-v0.1.10'},'evalplus':{'commit':'26d6d00','docker_image':'evalplus:26d6d00','docker_image_id':'sha256:ce82d4f2e99754feb576991dec8d558096cbcb43644b53faf941324d77981c95','network':'none','counts':counts,'base_pass_set':base_pass,'plus_pass_set':plus_pass,'both_pass_set':both_pass,'base_fail_set':[f'HumanEval/{i}' for i in range(164) if f'HumanEval/{i}' not in base_pass],'plus_fail_set':[f'HumanEval/{i}' for i in range(164) if f'HumanEval/{i}' not in plus_pass],'both_fail_set':[f'HumanEval/{i}' for i in range(164) if f'HumanEval/{i}' not in both_pass]},'generation':manifest['generation'],'served_model':manifest['model'],'system_fingerprint':manifest['system_fingerprint'],'sealed_raw_rows':manifest['sealed_rows'],'per_task':per_task,'provenance':{'claim_sha256_at_seal':sha(claim_path),'stage_receipt_sha256':sha(MISSION/'receipts/STAGE_QSFP_EXACT.json'),'serve_recipe_sha256':sha(MISSION/'receipts/SERVE_RECIPE.json'),'generation_manifest_sha256':sha(manifest_path),'samples_sha256':sha(samples),'eval_results_sha256':sha(results_path),'eval_log_sha256':sha(OUT/'evalplus.log'),'scripts_sha256':script_hashes},'created_epoch':time.time()}
verdict_path=OUT/'VERDICT.json';atomic_json(verdict_path,verdict)
md=f"# Clean HE164 Transfer-8 EvalPlus Verdict\n\n- Checkpoint SHA-256: `{CHECKPOINT}`\n- Dataset: HumanEvalPlus-v0.1.10 (`{DATASET}`)\n- EvalPlus commit/image: `26d6d00` / `sha256:ce82d4f2e99754feb576991dec8d558096cbcb43644b53faf941324d77981c95`\n- Raw generations sealed: **164/164**\n- HumanEval base pass@1: **{counts['base']}/164**\n- HumanEval+ pass@1: **{counts['plus']}/164**\n- Both pass@1: **{counts['both']}/164**\n- Binding clean claim (HumanEval base >=160/164): **{science_claim['verdict']}**\n- Delta vs sealed STEP0 161/164: **{science_claim['comparison_references']['delta_vs_sealed_step0']:+d}**\n- Delta vs UD-IQ4_XS 161/164 reference: **{science_claim['comparison_references']['delta_vs_ud_iq4_xs']:+d}**\n- Generation: temperature=0, top_p={manifest['generation']['top_p']}, max_tokens={manifest['generation']['max_tokens']}, seed=0, fingerprint=`{manifest['system_fingerprint']}`\n\nFailure sets and all per-task outputs are in `VERDICT.json`.\n"
atomic_text(OUT/'VERDICT.md',md)
print(json.dumps({'status':'PASS','science_claim':science_claim,'counts':counts,'verdict':str(verdict_path),'verdict_sha256':sha(verdict_path)},sort_keys=True))
