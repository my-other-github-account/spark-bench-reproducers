#!/usr/bin/env python3
import importlib.util, json, math, hashlib
from collections import Counter
from pathlib import Path
ROOT=Path('$HOME/run-bundles/P629_GLOBAL_QTIP2_PUBLIC_TASK_s1')
spec=importlib.util.spec_from_file_location('p629',ROOT/'code'/'solve_global_ab.py')
p629=importlib.util.module_from_spec(spec); spec.loader.exec_module(p629)
gs=p629.load_original()
anchors=gs.load_anchor_grid(ROOT/'inputs'/'rung1'/'ANCHOR_VERTICAL_GRID.csv')
rows=gs.load_profile(ROOT/'inputs'/'profile'/'PROFILE_ROWS.jsonl')
importance,_=gs.normalize_profile_rows(rows)
step0=gs.step0_means(ROOT/'inputs'/'baseline'/'BQ3_STEP0_PER_CLASS.json')
old,_=gs.map_incumbent(ROOT/'inputs'/'baseline'/'DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json')
corrections,_=gs.fit_projection_corrections(old,importance,anchors,step0)
cells=gs.make_cells(importance,anchors,corrections)
_,inc=p629.read_assignment(gs)
by={tuple(c['key']):c for c in cells}
inc_payload=sum(next(o['bytes'] for o in by[k]['options'] if o['tier']==t) for k,t in inc.items())
incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections)
ceilings={c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES}
# Bounded coarse lambda sweep; refine only around the first feasible transition.
coarse=[i/10 for i in range(0,21)] + [2.5,3,4,5,8,12,20,50,100]
cache={}
def evaluate(lam):
 weights={c:(1.0+lam if c=='code' else 1.0) for c in gs.CLASSES}
 sol=gs.solve_weighted_greedy(cells,weights,inc_payload)
 pred=gs.predict_assignment(sol['assignment'],importance,anchors,gs.CLASSES,corrections=corrections)
 feasible=all(pred[c] <= ceilings[c]+1e-12 for c in gs.CLASSES)
 obj=math.fsum(pred.values())/len(gs.CLASSES)
 cache[lam]=(obj,sol,pred,feasible)
 return cache[lam]
for lam in coarse: evaluate(lam)
first_feasible=next((x for x in coarse if cache[x][3]),None)
if first_feasible is not None:
 idx=coarse.index(first_feasible)
 lo=coarse[max(0,idx-1)]; hi=first_feasible
 for _ in range(10):
  mid=(lo+hi)/2
  if evaluate(mid)[3]: hi=mid
  else: lo=mid
lambdas=sorted(cache)
rows_out=[]; best=None
for lam in lambdas:
 obj,sol,pred,feasible=cache[lam]
 row={'lambda':lam,'objective':obj,'prediction':pred,'feasible':feasible,'payload':sol['payload_bytes'],'tier_counts':dict(Counter(sol['assignment'].values()))}
 rows_out.append(row)
 if feasible and (best is None or obj < best[0]): best=(obj,lam,sol,pred)
if best is None: raise SystemExit('no feasible lagrange candidate')
obj,lam,sol,pred=best
assignment={str(l):{str(e):{p:sol['assignment'][(l,e,p)] for p in gs.PROJECTIONS} for e in range(gs.EXPERTS)} for l in range(gs.LAYERS)}
assignment_text=json.dumps(assignment,sort_keys=True,separators=(',',':')).encode()
out={'schema':'p629-lagrange-warmstart-v1','lambda':lam,'objective':obj,'prediction':pred,'ceilings':ceilings,'payload':sol['payload_bytes'],'physical_bytes':p629.ENVELOPE+(sol['payload_bytes']-inc_payload),'tier_counts':dict(Counter(sol['assignment'].values())),'assignment_map_sha256':hashlib.sha256(assignment_text).hexdigest(),'assignment':assignment,'sweep':rows_out}
p=(ROOT/'out'/'LAGRANGE_WARMSTART.json'); p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ['lambda','objective','prediction','ceilings','payload','physical_bytes','tier_counts','assignment_map_sha256']},sort_keys=True))
