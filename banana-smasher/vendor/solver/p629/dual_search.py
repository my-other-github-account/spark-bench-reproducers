#!/usr/bin/env python3
import importlib.util,json,math,hashlib,time
from collections import Counter
from pathlib import Path
ROOT=Path('$HOME/run-bundles/P629_GLOBAL_QTIP2_PUBLIC_TASK_s1')
spec=importlib.util.spec_from_file_location('p629',ROOT/'code'/'solve_global_ab.py'); p629=importlib.util.module_from_spec(spec); spec.loader.exec_module(p629)
gs=p629.load_original(); anchors=gs.load_anchor_grid(ROOT/'inputs'/'rung1'/'ANCHOR_VERTICAL_GRID.csv'); rows=gs.load_profile(ROOT/'inputs'/'profile'/'PROFILE_ROWS.jsonl'); importance,_=gs.normalize_profile_rows(rows); step0=gs.step0_means(ROOT/'inputs'/'baseline'/'BQ3_STEP0_PER_CLASS.json'); old,_=gs.map_incumbent(ROOT/'inputs'/'baseline'/'DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json'); corrections,_=gs.fit_projection_corrections(old,importance,anchors,step0); cells=gs.make_cells(importance,anchors,corrections); _,inc=p629.read_assignment(gs); by={tuple(c['key']):c for c in cells}; inc_payload=sum(next(o['bytes'] for o in by[k]['options'] if o['tier']==t) for k,t in inc.items()); incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections); ceilings={c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES}
weights={c:1.0 for c in gs.CLASSES}; history=[]; best=None; best_v=None
for i in range(120):
 sol=gs.solve_weighted_greedy(cells,weights,inc_payload); pred=gs.predict_assignment(sol['assignment'],importance,anchors,gs.CLASSES,corrections=corrections); obj=math.fsum(pred.values())/len(gs.CLASSES); rel={c:max(0.0,(pred[c]-ceilings[c])/ceilings[c]) for c in gs.CLASSES}; violation=max(rel.values()); feasible=violation <= 1e-11
 row={'iteration':i,'objective':obj,'prediction':pred,'weights':dict(weights),'max_relative_violation':violation,'feasible':feasible}; history.append(row)
 if best_v is None or violation < best_v: best_v=violation
 if feasible and (best is None or obj < best[0]): best=(obj,sol,pred,dict(weights),i)
 # Multiplicatively penalize all violated coordinates, strongest coordinate most.
 for c in gs.CLASSES:
  if rel[c] > 0: weights[c] *= math.exp(min(0.35,0.08+3.0*rel[c]))
 # Every 12 iterations slightly relax inactive penalties to avoid permanent overshoot.
 if i and i%12==0:
  for c in gs.CLASSES:
   if rel[c]==0 and weights[c]>1: weights[c]=max(1.0,weights[c]*0.97)
 (ROOT/'out'/'DUAL_SEARCH_PROGRESS.json').write_text(json.dumps({'iteration':i,'current':row,'best_feasible_objective':best[0] if best else None,'best_relative_violation':best_v},indent=2,sort_keys=True)+'\n')
if best is None:
 (ROOT/'out'/'DUAL_SEARCH_FAIL.json').write_text(json.dumps({'history':history,'ceilings':ceilings,'best_relative_violation':best_v},indent=2,sort_keys=True)+'\n'); raise SystemExit('no feasible dual candidate')
obj,sol,pred,bw,bi=best; assignment={str(l):{str(e):{p:sol['assignment'][(l,e,p)] for p in gs.PROJECTIONS} for e in range(gs.EXPERTS)} for l in range(gs.LAYERS)}; text=json.dumps(assignment,sort_keys=True,separators=(',',':')).encode(); out={'schema':'p629-dual-warmstart-v1','iteration':bi,'weights':bw,'objective':obj,'prediction':pred,'ceilings':ceilings,'payload':sol['payload_bytes'],'physical_bytes':p629.ENVELOPE+(sol['payload_bytes']-inc_payload),'tier_counts':dict(Counter(sol['assignment'].values())),'assignment_map_sha256':hashlib.sha256(text).hexdigest(),'assignment':assignment,'history':history}; p=ROOT/'out'/'DUAL_WARMSTART.json'; p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:out[k] for k in ['iteration','weights','objective','prediction','physical_bytes','tier_counts','assignment_map_sha256']},sort_keys=True))
