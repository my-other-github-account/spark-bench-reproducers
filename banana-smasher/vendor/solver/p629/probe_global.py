#!/usr/bin/env python3
import importlib.util, json, math
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
sol=gs.solve_weighted_greedy(cells,{c:1/len(gs.CLASSES) for c in gs.CLASSES},inc_payload)
pred=gs.predict_assignment(sol['assignment'],importance,anchors,gs.CLASSES,corrections=corrections)
incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections)
changes=[]
for k,t in sol['assignment'].items():
 if t!=inc[k]: changes.append({'layer':k[0],'expert':k[1],'projection':k[2],'from':inc[k],'to':t})
out={'inc_payload':inc_payload,'payload':sol['payload_bytes'],'objective':math.fsum(pred.values())/len(gs.CLASSES),'inc_objective':math.fsum(incpred.values())/len(gs.CLASSES),'prediction':pred,'inc_prediction':incpred,'ceilings':{c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES},'ceiling_pass':{c:pred[c] <= (incpred['code'] if c=='code' else step0[c])+1e-12 for c in gs.CLASSES},'tier_counts':dict(Counter(sol['assignment'].values())),'changed_cells':len(changes),'changes_preview':changes[:50]}
print(json.dumps(out,sort_keys=True))
