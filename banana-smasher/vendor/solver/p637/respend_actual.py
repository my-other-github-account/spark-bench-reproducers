#!/usr/bin/env python3
"""P637 final local re-spend plus global LP lower bound.

Starts from the verified P637 SCIP feasible assignment, repeatedly spends/frees bytes
with ordinary one-cell menu moves under only the global six-class caps, and solves
the continuous relaxation of the full ordinary per-expert menu as a rigorous lower
bound. No per-cell/per-move class veto is imposed; feasibility is checked only after
aggregating each proposed assignment state against the global class constraints.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, time
from collections import Counter, defaultdict
from pathlib import Path
from ortools.linear_solver import pywraplp

ROOT=Path('$HOME/run-bundles/P637_ACTUAL_PUBLIC_TASK_s3')
OUT=ROOT/'out'
ENVELOPE=101_346_700_411
EPS=1e-14


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def atomic_json(p:Path,obj)->str:
    payload=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode()
    t=p.with_name(p.name+f'.tmp.{os.getpid()}')
    with t.open('wb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(t,p); fd=os.open(p.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
    return sha256(p)


def load_surface():
    spec=importlib.util.spec_from_file_location('p637_base',ROOT/'code/solve_actual.py')
    s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
    gs=s.load_original()
    anchors=gs.load_anchor_grid(s.INPUTS/'rung1'/'ANCHOR_VERTICAL_GRID.csv')
    rows=gs.load_profile(s.INPUTS/'profile'/'PROFILE_ROWS.jsonl')
    importance,normalization=gs.normalize_profile_rows(rows)
    step0_path=s.INPUTS/'baseline'/'BQ3_STEP0_PER_CLASS.json'
    step0_doc=json.loads(step0_path.read_text()); step0=gs.step0_means(step0_path)
    old,_=gs.map_incumbent(s.INPUTS/'baseline'/'DUALVQ_K4096MENU_BQ3_BIN_MANIFEST.json')
    corrections,_=gs.fit_projection_corrections(old,importance,anchors,step0)
    cells=gs.make_cells(importance,anchors,corrections)
    _incdoc,inc=s.read_assignment(gs)
    incpred=gs.predict_assignment(inc,importance,anchors,gs.CLASSES,corrections=corrections)
    delta_surface,*_=s.load_anchor_deltas(gs)
    gw=s.weighted_global_weights(step0_doc,gs.CLASSES)
    qci,_qgi,_closure=s.build_qtip_surface(gs,importance,corrections,delta_surface,gw)
    by={tuple(c['key']):c for c in cells}
    opts={}
    for k,cell in by.items():
        local={str(o['tier']):dict(o) for o in cell['options']}
        if k in qci:
            oldopt=local[inc[k]]
            local[s.QTIP_TIER]={
                'tier':s.QTIP_TIER,
                'bytes':int(s.QTIP_PHYSICAL_BYTES_BY_LAYER[k[0]][k[2]]),
                'costs':{c:float(oldopt['costs'][c])+float(qci[k][c]) for c in gs.CLASSES},
            }
        opts[k]=local
    ceilings={c:(incpred['code'] if c=='code' else step0[c]) for c in gs.CLASSES}
    return s,gs,opts,inc,incpred,ceilings,normalization


def summarize(gs,opts,inc,sel):
    pred={c:0.0 for c in gs.CLASSES}; delta_bytes=0
    for k,t in sel.items():
        o=opts[k][t]; old=opts[k][inc[k]]
        delta_bytes+=int(o['bytes'])-int(old['bytes'])
        for c in gs.CLASSES: pred[c]+=float(o['costs'][c])
    return pred,ENVELOPE+delta_bytes


def moves(gs,opts,sel):
    out=[]
    for k,curtier in sel.items():
        cur=opts[k][curtier]
        for tier,o in opts[k].items():
            if tier==curtier: continue
            dc={c:float(o['costs'][c])-float(cur['costs'][c]) for c in gs.CLASSES}
            dobj=math.fsum(dc.values())/len(gs.CLASSES)
            if dobj >= -EPS: continue
            db=int(o['bytes'])-int(cur['bytes'])
            out.append((dobj,db,k,tier,dc))
    return out


def greedy_respend(gs,opts,sel,pred,payload,ceilings):
    ledger=[]
    def fits(db,dc):
        return payload+db<=ENVELOPE and all(
            pred[c]+dc[c]>=-1e-12 and pred[c]+dc[c]<=ceilings[c]+1e-12
            for c in gs.CLASSES)
    def apply(r,stage,round_no):
        nonlocal payload
        dobj,db,k,tier,dc=r; old=sel[k]
        sel[k]=tier; payload+=db
        for c in gs.CLASSES: pred[c]+=dc[c]
        ledger.append({'round':round_no,'stage':stage,'layer':k[0],'expert':k[1],
                       'projection':k[2],'from':old,'to':tier,'delta_bytes':db,
                       'delta_objective':dobj,'delta_by_class':dc})
    for rnd in range(1,31):
        changed=0; used=set()
        # Sellers first: objective-improving and byte-freeing. This includes QTIP2
        # trades once globally accumulated class headroom makes them legal.
        free=sorted((r for r in moves(gs,opts,sel) if r[1]<=0),key=lambda r:(r[0],r[1],r[2],r[3]))
        for r in free:
            if r[2] in used or sel[r[2]]==r[3] or not fits(r[1],r[4]): continue
            apply(r,'free_or_neutral',rnd); used.add(r[2]); changed+=1
        # Buyers: maximize objective improvement per byte, still checking only
        # aggregate six-class caps. Recompute from post-seller state.
        used=set()
        buy=sorted((r for r in moves(gs,opts,sel) if r[1]>0),
                   key=lambda r:(r[0]/r[1],r[0],r[1],r[2],r[3]))
        for r in buy:
            if r[2] in used or sel[r[2]]==r[3] or not fits(r[1],r[4]): continue
            apply(r,'reallocate_freed_bytes',rnd); used.add(r[2]); changed+=1
        if not changed: break
    return sel,pred,payload,ledger


def lp_lower_bound(gs,opts,inc,incpred,ceilings):
    lp=pywraplp.Solver.CreateSolver('GLOP')
    if lp is None: raise RuntimeError('GLOP unavailable')
    byte=lp.RowConstraint(-lp.infinity(),0.0,'physical_delta_le_zero')
    crows={c:lp.RowConstraint(0.0,ceilings[c]*1e6,f'global_{c}') for c in gs.CLASSES}
    obj=lp.Objective(); obj.SetMinimization()
    nvars=0
    for k,local in opts.items():
        one=lp.RowConstraint(1.0,1.0,f'one_{k[0]}_{k[1]}_{k[2]}')
        old=local[inc[k]]
        for tier,o in local.items():
            v=lp.NumVar(0.0,1.0,f'x_{k[0]}_{k[1]}_{k[2]}_{tier}'); nvars+=1
            one.SetCoefficient(v,1.0)
            byte.SetCoefficient(v,(int(o['bytes'])-int(old['bytes']))/1e6)
            for c in gs.CLASSES: crows[c].SetCoefficient(v,float(o['costs'][c])*1e6)
            obj.SetCoefficient(v,math.fsum(float(o['costs'][c]) for c in gs.CLASSES)/len(gs.CLASSES)*1e6)
    status=lp.Solve()
    names={lp.OPTIMAL:'OPTIMAL',lp.FEASIBLE:'FEASIBLE',lp.INFEASIBLE:'INFEASIBLE',lp.ABNORMAL:'ABNORMAL',lp.NOT_SOLVED:'NOT_SOLVED'}
    name=names.get(status,str(status))
    if status!=lp.OPTIMAL: raise RuntimeError({'lp_status':name})
    return {'kind':'GLOP full continuous relaxation of identical menu/constraints',
            'status':name,'lower_bound':obj.Value()/1e6,'wall_time_ms':lp.wall_time(),
            'iterations':lp.iterations(),'variables':nvars,'constraints':lp.NumConstraints(),
            'rigorous_relation':'LP feasible region is a superset of integer assignments; optimum is a valid lower bound'}


def main():
    started=time.time(); s,gs,opts,inc,incpred,ceilings,norm=load_surface()
    source=json.loads((OUT/'ASSIGNMENT_WITH.json').read_text())
    sel={(l,e,p):source['assignment'][str(l)][str(e)][p]
         for l in range(gs.LAYERS) for e in range(gs.EXPERTS) for p in gs.PROJECTIONS}
    pred,payload=summarize(gs,opts,inc,sel)
    initial={'objective':math.fsum(pred.values())/len(gs.CLASSES),'prediction':dict(pred),'bytes':payload}
    sel,pred,payload,ledger=greedy_respend(gs,opts,sel,pred,payload,ceilings)
    obj=math.fsum(pred.values())/len(gs.CLASSES)
    lp=lp_lower_bound(gs,opts,inc,incpred,ceilings)
    gap=max(0.0,obj-lp['lower_bound'])/max(abs(obj),1e-30)
    amap={str(l):{str(e):{p:sel[(l,e,p)] for p in gs.PROJECTIONS} for e in range(gs.EXPERTS)} for l in range(gs.LAYERS)}
    mapsha=hashlib.sha256(json.dumps(amap,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    arec={'schema':'p637-final-respent-assignment-v1','assignment':amap,'assignment_map_sha256':mapsha,
          'source_assignment_sha256':sha256(OUT/'ASSIGNMENT_WITH.json'),'input_manifest_sha256':s.EXPECTED_INPUT_MANIFEST_SHA,
          'incumbent_assignment_sha256':s.EXPECTED_ASSIGNMENT_SHA,'existing_menu_sha256':s.EXPECTED_EXISTING_MENU_SHA}
    arec_sha=atomic_json(OUT/'ASSIGNMENT_RESPENT.json',arec)
    tiers=Counter(sel.values()); qkeys=[k for k,t in sel.items() if t==s.QTIP_TIER]; qlayers=Counter(k[0] for k in qkeys)
    transitions=Counter((inc[k],sel[k]) for k in sel if inc[k]!=sel[k])
    qrows=[r for r in ledger if r['to']==s.QTIP_TIER or r['from']==s.QTIP_TIER]
    qdelta=sum(int(opts[k][sel[k]]['bytes'])-int(opts[k][inc[k]]['bytes']) for k in qkeys)
    nonq_delta=(payload-ENVELOPE)-qdelta
    result={
      'schema':'p637-final-respend-v1','status':'PASS_FEASIBLE_RESPENT_WITH_VALID_LP_BOUND',
      'host':os.uname().nodename,'objective_name':'uniform mean of six predicted class KLDs',
      'constraint_policy':{'per_class_ceilings_are_global':True,'per_cell_or_per_move_code_nonworsening_veto':False,
                           'cross_class_trades_permitted':True},
      'without':{'objective':math.fsum(incpred.values())/len(gs.CLASSES),'prediction':incpred,'bytes':ENVELOPE},
      'preliminary_with':initial,
      'with':{'objective':obj,'prediction':pred,'bytes':payload,'slack':ENVELOPE-payload,
              'delta_objective_vs_without':obj-math.fsum(incpred.values())/len(gs.CLASSES),
              'delta_objective_vs_preliminary':obj-initial['objective']},
      'ceilings':ceilings,'solver':{'method':'deterministic feasible global-cap greedy re-spend + full LP relaxation bound',
              'best_bound':lp['lower_bound'],'relative_gap':gap,'lp':lp,'wall_seconds':time.time()-started},
      'qtip2':{'selected_cells':len(qkeys),'selected_by_layer':{str(l):qlayers[l] for l in s.ELIGIBLE},
               'selected_cell_detail':[{'layer':k[0],'expert':k[1],'projection':k[2],'from':inc[k]} for k in sorted(qkeys)],
               'selected_experts_by_layer':{str(l):{str(e):sorted([p for ll,ee,p in qkeys if ll==l and ee==e])
                   for e in sorted({ee for ll,ee,p in qkeys if ll==l})} for l in s.ELIGIBLE},
               'into_by_from_tier':dict(sorted(Counter(inc[k] for k in qkeys).items()))},
      'bytes':{'envelope':ENVELOPE,'without_exact':ENVELOPE,'with_exact':payload,'slack':ENVELOPE-payload,
               'qtip2_net_delta':qdelta,'qtip2_bytes_freed':-qdelta,
               'reallocated_to_non_qtip_tiers':max(0,nonq_delta),'non_qtip_net_delta':nonq_delta,
               'net_with_minus_without':payload-ENVELOPE},
      'tier_counts':dict(sorted(tiers.items())),
      'transition_counts':{f'{a}->{b}':n for (a,b),n in sorted(transitions.items())},
      'greedy_move_count':len(ledger),'greedy_rounds':max((r['round'] for r in ledger),default=0),
      'greedy_ledger':ledger,'assignment_map_sha256':mapsha,'assignment_receipt_sha256':arec_sha,
      'source_assignment_sha256':sha256(OUT/'ASSIGNMENT_WITH.json'),'source_result_sha256':sha256(OUT/'WITH_RESULT.json'),
      'input_manifest_sha256':s.EXPECTED_INPUT_MANIFEST_SHA,'incumbent_assignment_sha256':s.EXPECTED_ASSIGNMENT_SHA,
      'existing_menu_sha256':s.EXPECTED_EXISTING_MENU_SHA,'base_solver_code_sha256':sha256(ROOT/'code/solve_actual.py'),
      'respend_code_sha256':sha256(Path(__file__)),'started_unix':started,'finished_unix':time.time(),
    }
    rsha=atomic_json(OUT/'RESPEND_RESULT.json',result)
    done={'schema':'p637-final-respend-done-v1','status':result['status'],'result_sha256':rsha,
          'assignment_receipt_sha256':arec_sha,'assignment_map_sha256':mapsha,'objective':obj,
          'best_bound':lp['lower_bound'],'relative_gap':gap,'exact_bytes':payload,'slack':ENVELOPE-payload,
          'qtip2_selected_cells':len(qkeys),'qtip2_selected_by_layer':result['qtip2']['selected_by_layer'],
          'respend_code_sha256':result['respend_code_sha256'],'elapsed_seconds':time.time()-started}
    dsha=atomic_json(OUT/'RESPEND_DONE.json',done)
    print(json.dumps({**done,'done_sha256':dsha},sort_keys=True))

if __name__=='__main__': main()
