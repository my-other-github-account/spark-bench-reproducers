#!/usr/bin/env python3
"""Independent fail-closed verifier for P637 final re-spend + LP bound."""
import hashlib, importlib.util, json, math, os
from collections import Counter
from pathlib import Path
ROOT=Path('$HOME/run-bundles/P637_ACTUAL_PUBLIC_TASK_s3'); OUT=ROOT/'out'; ENVELOPE=101_346_700_411

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(p,o):
 t=p.with_name(p.name+f'.tmp.{os.getpid()}'); t.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); os.replace(t,p); return sha(p)
def close(a,b,t=1e-11): return abs(float(a)-float(b))<=t
spec=importlib.util.spec_from_file_location('r',ROOT/'code/respend_actual.py'); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
s,gs,opts,inc,incpred,ceilings,_=r.load_surface()
res=json.loads((OUT/'RESPEND_RESULT.json').read_text()); arec=json.loads((OUT/'ASSIGNMENT_RESPENT.json').read_text()); done=json.loads((OUT/'RESPEND_DONE.json').read_text())
amap=arec['assignment']; amap_wire=hashlib.sha256(json.dumps(amap,sort_keys=True,separators=(',',':')).encode()).hexdigest()
sel={}; valid=True
for l in range(gs.LAYERS):
 for e in range(gs.EXPERTS):
  for p in gs.PROJECTIONS:
   k=(l,e,p); t=amap[str(l)][str(e)][p]; sel[k]=t; valid=valid and t in opts[k]
pred,payload=r.summarize(gs,opts,inc,sel); obj=math.fsum(pred.values())/len(gs.CLASSES)
tiers=Counter(sel.values()); qkeys=[k for k,t in sel.items() if t==s.QTIP_TIER]; qlayers=Counter(k[0] for k in qkeys)
trans=Counter((inc[k],sel[k]) for k in sel if inc[k]!=sel[k])
qdelta=sum(int(opts[k][sel[k]]['bytes'])-int(opts[k][inc[k]]['bytes']) for k in qkeys)
nonq=(payload-ENVELOPE)-qdelta
lp=res['solver']['lp']; lower=float(lp['lower_bound']); gap=max(0.0,obj-lower)/max(abs(obj),1e-30)
# Exhaustive one-move closure over the identical ordinary menu. r.moves emits
# every strictly objective-improving alternative from the final assignment;
# feasibility is tested only against aggregate bytes and global class rows.
one_move_scanned=sum(len(local)-1 for local in opts.values())
improving_moves=r.moves(gs,opts,sel)
feasible_improving=[]
for dobj,db,k,tier,dc in improving_moves:
 if payload+db<=ENVELOPE and all(pred[c]+dc[c]>=-1e-12 and pred[c]+dc[c]<=ceilings[c]+1e-12 for c in gs.CLASSES):
  feasible_improving.append({'layer':k[0],'expert':k[1],'projection':k[2],'to':tier,'delta_bytes':db,'delta_objective':dobj})
checks={
 'result_sha_matches_done':sha(OUT/'RESPEND_RESULT.json')==done['result_sha256'],
 'assignment_sha_matches_done':sha(OUT/'ASSIGNMENT_RESPENT.json')==done['assignment_receipt_sha256'],
 'assignment_map_sha_closure':amap_wire==arec['assignment_map_sha256']==done['assignment_map_sha256']==res['assignment_map_sha256'],
 'all_cells_have_valid_ordinary_menu_option':valid and len(sel)==gs.LAYERS*gs.EXPERTS*len(gs.PROJECTIONS),
 'input_manifest_exact':res['input_manifest_sha256']==s.EXPECTED_INPUT_MANIFEST_SHA,
 'incumbent_assignment_exact':res['incumbent_assignment_sha256']==s.EXPECTED_ASSIGNMENT_SHA,
 'existing_menu_exact':res['existing_menu_sha256']==s.EXPECTED_EXISTING_MENU_SHA,
 'source_assignment_sha_exact':res['source_assignment_sha256']==sha(OUT/'ASSIGNMENT_WITH.json'),
 'source_result_sha_exact':res['source_result_sha256']==sha(OUT/'WITH_RESULT.json'),
 'exact_bytes_replay':payload==res['with']['bytes']==done['exact_bytes'],
 'byte_cap':payload<=ENVELOPE,
 'slack_closure':ENVELOPE-payload==res['with']['slack']==done['slack'],
 'class_prediction_replay':all(close(pred[c],res['with']['prediction'][c]) for c in gs.CLASSES),
 'class_nonnegative':all(pred[c]>=-1e-12 for c in gs.CLASSES),
 'global_class_ceilings':all(pred[c]<=ceilings[c]+1e-12 for c in gs.CLASSES),
 'objective_uniform_six_mean':close(obj,res['with']['objective']) and close(obj,done['objective']),
 'objective_nonregression':obj<=res['without']['objective']+1e-12,
 'objective_improves_respent_seed':obj<=res['preliminary_with']['objective']+1e-12,
 'qtip_present':len(qkeys)>0,
 'qtip_count_closure':len(qkeys)==res['qtip2']['selected_cells']==done['qtip2_selected_cells'],
 'qtip_by_layer_closure':all(qlayers[l]==int(res['qtip2']['selected_by_layer'][str(l)]) for l in s.ELIGIBLE),
 'tier_counts_closure':dict(sorted(tiers.items()))==res['tier_counts'],
 'transition_counts_closure':{f'{a}->{b}':n for (a,b),n in sorted(trans.items())}==res['transition_counts'],
 'qtip_byte_delta_closure':qdelta==res['bytes']['qtip2_net_delta'] and -qdelta==res['bytes']['qtip2_bytes_freed'],
 'non_qtip_reallocation_closure':nonq==res['bytes']['non_qtip_net_delta']==res['bytes']['reallocated_to_non_qtip_tiers'],
 'net_byte_closure':qdelta+nonq==payload-ENVELOPE==res['bytes']['net_with_minus_without'],
 'global_constraints_only':res['constraint_policy']['per_class_ceilings_are_global'] and res['constraint_policy']['cross_class_trades_permitted'] and not res['constraint_policy']['per_cell_or_per_move_code_nonworsening_veto'],
 'lp_relaxation_optimal':lp['status']=='OPTIMAL',
 'lp_bound_valid_relation':lower<=obj+1e-12,
 'gap_closure':close(gap,res['solver']['relative_gap']) and close(gap,done['relative_gap']),
 'usable_gap_le_30pct':gap<=0.30,
 'respend_code_sha_exact':res['respend_code_sha256']==sha(ROOT/'code/respend_actual.py'),
 'exhaustive_one_move_local_closure':len(feasible_improving)==0,
}
failed=[k for k,v in checks.items() if not v]
out={'schema':'p637-final-respend-independent-verifier-v2','status':'PASS' if not failed else 'FAIL','checks':checks,'failed_checks':failed,
 'one_move_scan':{'alternatives_scanned':one_move_scanned,'objective_improving_alternatives':len(improving_moves),
                  'aggregate_feasible_objective_improving':len(feasible_improving),'remaining_moves':feasible_improving[:20]},
 'result_sha256':sha(OUT/'RESPEND_RESULT.json'),'assignment_receipt_sha256':sha(OUT/'ASSIGNMENT_RESPENT.json'),
 'assignment_map_sha256':amap_wire,'done_sha256':sha(OUT/'RESPEND_DONE.json'),'objective':obj,'best_bound':lower,'relative_gap':gap,
 'exact_bytes':payload,'slack':ENVELOPE-payload,'qtip2_selected_cells':len(qkeys),'qtip2_selected_by_layer':{str(l):qlayers[l] for l in s.ELIGIBLE},
 'verifier_code_sha256':sha(Path(__file__))}
vsha=atomic(OUT/'RESPEND_VERIFIER.json',out); print(json.dumps({**out,'sha256':vsha},sort_keys=True)); raise SystemExit(0 if not failed else 2)
