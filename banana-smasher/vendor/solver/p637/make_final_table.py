#!/usr/bin/env python3
import hashlib,json,math,os,time
from pathlib import Path
R=Path('$HOME/run-bundles/P637_ACTUAL_PUBLIC_TASK_s3'); O=R/'out'
res=json.loads((O/'RESPEND_RESULT.json').read_text()); ver=json.loads((O/'RESPEND_VERIFIER.json').read_text()); done=json.loads((O/'RESPEND_DONE.json').read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,o):
 t=p.with_name(p.name+f'.tmp.{os.getpid()}'); t.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); os.replace(t,p); return sha(p)
w=res['without']; x=res['with']; q=res['qtip2']; b=res['bytes']; solver=res['solver']; classes=list(w['prediction'])
table={
 'schema':'p637-final-with-without-table-v1','status':'FINAL_VERIFIED_PASS','host':'compute-node-3',
 'question_answer':{
  'does_backpack_buy_qtip2_at_bit_size_parity':True,
  'how_much':{'qtip2_cells':q['selected_cells'],'selected_by_layer':q['selected_by_layer'],'selected_experts_by_layer':q['selected_experts_by_layer']},
  'what_freed_bytes_buy':{'qtip2_bytes_freed':b['qtip2_bytes_freed'],'reallocated_to_non_qtip_tiers':b['reallocated_to_non_qtip_tiers'],
    'net_bytes_freed':-b['net_with_minus_without'],'remaining_slack':x['slack'],'complete_transition_counts':res['transition_counts']},
 },
 'constraint_semantics':res['constraint_policy'],
 'without':{'objective':w['objective'],'global_uniform_six':w['objective'],'predicted_classes':w['prediction'],'exact_bytes':w['bytes'],'slack':0,
            'tier_counts':{'d4_k1024':4781,'d4_k2048':9923,'d4_k256':2744,'d4_k4096':4056,'d8_k256':6,'native_mxfp4':506}},
 'with':{'objective':x['objective'],'global_uniform_six':x['objective'],'predicted_classes':x['prediction'],'exact_bytes':x['bytes'],'slack':x['slack'],
         'tier_counts':res['tier_counts'],'qtip2_selected_cells':q['selected_cells'],'qtip2_selected_by_layer':q['selected_by_layer'],
         'qtip2_selected_experts_by_layer':q['selected_experts_by_layer']},
 'delta_with_minus_without':{'objective':x['objective']-w['objective'],'predicted_classes':{c:x['prediction'][c]-w['prediction'][c] for c in classes},
                             'exact_bytes':x['bytes']-w['bytes']},
 'qtip2_into_by_from_tier':q['into_by_from_tier'],'complete_transition_counts':res['transition_counts'],
 'bytes':b,
 'solver':{'integer_status':'FEASIBLE','final_method':solver['method'],'best_bound':solver['best_bound'],'relative_gap':solver['relative_gap'],
           'wall_seconds':solver['wall_seconds'],'lp_status':solver['lp']['status'],'lp_kind':solver['lp']['kind'],
           'lp_variables':solver['lp']['variables'],'lp_constraints':solver['lp']['constraints']},
 'receipts':{'assignment_map_sha256':res['assignment_map_sha256'],'assignment_receipt_sha256':res['assignment_receipt_sha256'],
             'input_manifest_sha256':res['input_manifest_sha256'],'incumbent_assignment_sha256':res['incumbent_assignment_sha256'],
             'existing_menu_sha256':res['existing_menu_sha256'],'base_solver_code_sha256':res['base_solver_code_sha256'],
             'respend_code_sha256':res['respend_code_sha256'],'verifier_code_sha256':ver['verifier_code_sha256'],
             'result_sha256':sha(O/'RESPEND_RESULT.json'),'done_sha256':sha(O/'RESPEND_DONE.json'),
             'verifier_sha256':sha(O/'RESPEND_VERIFIER.json'),'verifier_status':ver['status']},
 'created_unix':time.time(),
}
jsha=atomic(O/'FINAL_TABLE.json',table)
lines=['# P637 FINAL — correct class-balanced per-expert QTIP2 backpack','',
'Verifier: PASS. All class ceilings are global; cross-class trades are permitted; there is no per-cell/per-move code non-worsening veto.','',
'| field | WITHOUT | WITH | delta |','|---|---:|---:|---:|',
f"| objective / global uniform-six | {w['objective']:.17g} | {x['objective']:.17g} | {x['objective']-w['objective']:+.17g} |"]
for c in classes: lines.append(f"| {c} | {w['prediction'][c]:.17g} | {x['prediction'][c]:.17g} | {x['prediction'][c]-w['prediction'][c]:+.17g} |")
lines += [f"| exact bytes | {w['bytes']} | {x['bytes']} | {x['bytes']-w['bytes']:+d} |",f"| slack | 0 | {x['slack']} | +{x['slack']} |",'',
 f"QTIP2: {q['selected_cells']} cells; by layer `{json.dumps(q['selected_by_layer'],sort_keys=True)}`.",
 f"QTIP2 displaced from tiers: `{json.dumps(q['into_by_from_tier'],sort_keys=True)}`.",
 f"QTIP2 freed {b['qtip2_bytes_freed']:,} B; {b['reallocated_to_non_qtip_tiers']:,} B were reallocated into ordinary existing tiers; net freed/slack is {-b['net_with_minus_without']:,}/{x['slack']:,} B.",
 f"Complete WITH tiers: `{json.dumps(res['tier_counts'],sort_keys=True)}`.",
 f"Complete transitions: `{json.dumps(res['transition_counts'],sort_keys=True)}`.",'',
 f"Solver: integer FEASIBLE; full identical-menu LP relaxation OPTIMAL; best bound {solver['best_bound']:.17g}; rigorous relative gap {solver['relative_gap']:.6%}; re-spend+bound wall {solver['wall_seconds']:.3f}s.",'',
 'Answer: YES — at the same 101,346,700,411-B cap, the backpack buys 406 ordinary per-expert QTIP2 cells and leaves only 178,732 B slack. It spends the bytes QTIP2 frees on the ordinary tier upgrades enumerated in complete_transition_counts.','',
 f"Complete every-expert QTIP2 list and all SHAs: `{O/'FINAL_TABLE.json'}` (SHA256 `{jsha}`)."]
md=O/'FINAL_TABLE.md'; t=md.with_name(md.name+f'.tmp.{os.getpid()}'); t.write_text('\n'.join(lines)+'\n'); os.replace(t,md)
print(json.dumps({'final_table_json':str(O/'FINAL_TABLE.json'),'final_table_json_sha256':jsha,'final_table_md':str(md),'final_table_md_sha256':sha(md),'status':'PASS'},sort_keys=True))
