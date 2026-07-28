#!/usr/bin/env python3
import importlib.util,json,math,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path('$HOME/run-bundles/P629_GLOBAL_QTIP2_PUBLIC_TASK_s1'); spec=importlib.util.spec_from_file_location('p629',ROOT/'code'/'solve_global_ab.py'); p629=importlib.util.module_from_spec(spec); spec.loader.exec_module(p629); gs=p629.load_original(); anchors=gs.load_anchor_grid(ROOT/'inputs'/'rung1'/'ANCHOR_VERTICAL_GRID.csv'); rows=gs.load_profile(ROOT/'inputs'/'profile'/'PROFILE_ROWS.jsonl'); importance,_=gs.normalize_profile_rows(rows); step0=gs.step0_means(ROOT/'inputs'/'baseline'/'BQ3_STEP0_PER_CLASS.json'); old,_=gs.map_incumbent(ROOT/'inputs'/'baseline'/'DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json'); corrections,_=gs.fit_projection_corrections(old,importance,anchors,step0); cells=gs.make_cells(importance,anchors,corrections); _,inc=p629.read_assignment(gs); by={tuple(c['key']):c for c in cells}; inc_payload=sum(next(o['bytes'] for o in by[k]['options'] if o['tier']==t) for k,t in inc.items()); incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections); ceilings={c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES}
# Reconstruct the best near-feasible dual point observed at iteration 93.
weights={c:(10898.754329638245 if c=='code' else 1.0) for c in gs.CLASSES}; sol=gs.solve_weighted_greedy(cells,weights,inc_payload); selected=dict(sol['assignment']); pred=gs.predict_assignment(selected,importance,anchors,gs.CLASSES,corrections=corrections); payload=sol['payload_bytes']; before_obj=math.fsum(pred.values())/len(gs.CLASSES)
repairs=[]
for k,cell in by.items():
 cur=next(o for o in cell['options'] if o['tier']==selected[k])
 for opt in cell['options']:
  if opt['tier']==selected[k]: continue
  db=int(opt['bytes'])-int(cur['bytes']); dc={c:float(opt['costs'][c])-float(cur['costs'][c]) for c in gs.CLASSES}; np={c:pred[c]+dc[c] for c in gs.CLASSES}
  if payload+db<=inc_payload and all(np[c] <= ceilings[c]+1e-12 for c in gs.CLASSES):
   repairs.append((math.fsum(dc.values())/len(gs.CLASSES),db,k,opt['tier'],dc,np))
if repairs:
 chosen=(min(repairs,key=lambda x:(x[0],x[1])),)
else:
 # The near-feasible dual point is exactly at the byte cap, so a code-improving
 # buyer may need a second byte-selling move. Search the best bounded two-cell repair.
 violation=pred['code']-ceilings['code']
 changes=[]
 for k,cell in by.items():
  cur=next(o for o in cell['options'] if o['tier']==selected[k])
  for opt in cell['options']:
   if opt['tier']==selected[k]: continue
   db=int(opt['bytes'])-int(cur['bytes']); dc={c:float(opt['costs'][c])-float(cur['costs'][c]) for c in gs.CLASSES}; dobj=math.fsum(dc.values())/len(gs.CLASSES)
   changes.append((dobj,db,k,opt['tier'],dc))
 fixes=[x for x in changes if x[4]['code'] <= -violation-1e-12 and all(pred[c]+x[4][c] <= ceilings[c]+1e-12 for c in gs.CLASSES if c!='code')]
 fixes=sorted(fixes,key=lambda x:(x[0],x[1]))[:2000]
 sellers=sorted((x for x in changes if x[1]<0),key=lambda x:(x[0],x[1]))
 best_pair=None
 for a in fixes:
  for b in sellers:
   if a[2]==b[2] or a[1]+b[1]>0: continue
   if all(pred[c]+a[4][c]+b[4][c] <= ceilings[c]+1e-12 for c in gs.CLASSES):
    score=a[0]+b[0]
    if best_pair is None or score < best_pair[0]: best_pair=(score,a,b)
    break
 if best_pair is None: raise SystemExit('no one- or two-cell repair')
 chosen=(best_pair[1],best_pair[2])
repair_rows=[]
for row in chosen:
 dobj,db,k,tier,dc=row
 old_tier=selected[k]; selected[k]=tier; payload+=db
 for c in gs.CLASSES: pred[c]+=dc[c]
 repair_rows.append({'layer':k[0],'expert':k[1],'projection':k[2],'from':old_tier,'to':tier,'delta_bytes':db,'delta_by_class':dc,'delta_objective':dobj})
obj=math.fsum(pred.values())/len(gs.CLASSES)
assignment={str(l):{str(e):{p:selected[(l,e,p)] for p in gs.PROJECTIONS} for e in range(gs.EXPERTS)} for l in range(gs.LAYERS)}; text=json.dumps(assignment,sort_keys=True,separators=(',',':')).encode(); out={'schema':'p629-repaired-dual-warmstart-v1','source_code_weight':weights['code'],'before_objective':before_obj,'repair':repair_rows,'objective':obj,'prediction':pred,'ceilings':ceilings,'payload':payload,'physical_bytes':p629.ENVELOPE+(payload-inc_payload),'tier_counts':dict(Counter(selected.values())),'assignment_map_sha256':hashlib.sha256(text).hexdigest(),'assignment':assignment}; p=ROOT/'out'/'REPAIRED_DUAL_WARMSTART.json'; p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:out[k] for k in ['before_objective','repair','objective','prediction','physical_bytes','tier_counts','assignment_map_sha256']},sort_keys=True))
