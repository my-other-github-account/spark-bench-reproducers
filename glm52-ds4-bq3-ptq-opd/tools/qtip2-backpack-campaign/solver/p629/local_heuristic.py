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
incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections)
ceilings={c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES}
inc_payload=sum(next(o['bytes'] for o in by[k]['options'] if o['tier']==t) for k,t in inc.items())
selected=dict(inc); pred=dict(incpred); payload=inc_payload
candidates=[]
for k,cell in by.items():
 oldopt=next(o for o in cell['options'] if o['tier']==inc[k])
 for opt in cell['options']:
  if opt['tier']==inc[k]: continue
  dc={c:float(opt['costs'][c])-float(oldopt['costs'][c]) for c in gs.CLASSES}
  dobj=math.fsum(dc.values())/len(gs.CLASSES)
  db=int(opt['bytes'])-int(oldopt['bytes'])
  candidates.append((dobj,db,k,opt['tier'],dc))
def fits(db,dc):
 return payload+db <= inc_payload and all(pred[c]+dc[c] <= ceilings[c]+1e-12 and pred[c]+dc[c] >= -1e-12 for c in gs.CLASSES)
def apply(row):
 global payload
 dobj,db,k,tier,dc=row
 selected[k]=tier; payload+=db
 for c in gs.CLASSES: pred[c]+=dc[c]
# Pass 1: immediately useful sellers (strictly improve objective and do not consume bytes).
used=set()
for row in sorted((x for x in candidates if x[0] < -1e-18 and x[1] <= 0),key=lambda x:(x[0],x[1])):
 if row[2] in used or not fits(row[1],row[4]): continue
 apply(row); used.add(row[2])
# Pass 2: spend freed bytes on the largest remaining objective gains while respecting every ceiling.
for row in sorted((x for x in candidates if x[0] < -1e-18),key=lambda x:(x[0],x[1])):
 if row[2] in used or not fits(row[1],row[4]): continue
 apply(row); used.add(row[2])
obj=math.fsum(pred.values())/len(gs.CLASSES)
assignment={str(l):{str(e):{p:selected[(l,e,p)] for p in gs.PROJECTIONS} for e in range(gs.EXPERTS)} for l in range(gs.LAYERS)}
text=json.dumps(assignment,sort_keys=True,separators=(',',':')).encode()
# Exact qtip macro deltas from this feasible WITHOUT candidate.
delta_surface,measured,family_mean,family_global=p629.load_anchor_deltas(gs)
qtip_rows=[]
for layer in p629.ELIGIBLE:
 db=0; dc={c:0.0 for c in gs.CLASSES}; current_obj=0.0
 for e in range(gs.EXPERTS):
  for proj in gs.PROJECTIONS:
   k=(layer,e,proj); cell=by[k]
   oldopt=next(o for o in cell['options'] if o['tier']==inc[k])
   curopt=next(o for o in cell['options'] if o['tier']==selected[k])
   qcost={c:float(oldopt['costs'][c])+float(delta_surface[layer]['delta_by_class'][c])/512.0 for c in gs.CLASSES}
   db += p629.QTIP_BYTES[proj]-int(curopt['bytes'])
   for c in gs.CLASSES: dc[c]+=qcost[c]-float(curopt['costs'][c])
 dobj=math.fsum(dc.values())/len(gs.CLASSES)
 feasible=(payload+db <= inc_payload and all(pred[c]+dc[c] <= ceilings[c]+1e-12 for c in gs.CLASSES))
 qtip_rows.append({'layer':layer,'delta_bytes_vs_without':db,'delta_by_class_vs_without':dc,'delta_objective_vs_without':dobj,'measured_signed_global_delta_vs_shipped':delta_surface[layer]['global_delta'],'feasible_single_macro':feasible})
out={'schema':'p629-feasible-local-warmstart-v1','objective':obj,'incumbent_objective':math.fsum(incpred.values())/len(gs.CLASSES),'prediction':pred,'incumbent_prediction':incpred,'ceilings':ceilings,'payload':payload,'incumbent_payload':inc_payload,'physical_bytes':p629.ENVELOPE+(payload-inc_payload),'tier_counts':dict(Counter(selected.values())),'changed_cells':sum(selected[k]!=inc[k] for k in selected),'assignment_map_sha256':hashlib.sha256(text).hexdigest(),'assignment':assignment,'qtip_macro_rows':qtip_rows}
p=ROOT/'out'/'LOCAL_WARMSTART_WITHOUT.json'; p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ['objective','incumbent_objective','prediction','ceilings','physical_bytes','tier_counts','changed_cells','assignment_map_sha256','qtip_macro_rows']},sort_keys=True))
