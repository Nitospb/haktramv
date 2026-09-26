"""Test exogenous shapes while anchoring levels to the public-best baseline.

Reconciliation uses forecasts only, never future target values.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from feature_models import DEST as V4
from train_model import score

ROOT=V4.parent;DEST=ROOT/'v8';DEST.mkdir(exist_ok=True)
d=pd.read_csv(V4/'training_matrix.csv')
KEY=['route','date','hour']
MODES={'month':['route','month'],'month_daytype':['route','month','daytype'],
       'day':['route','date'],'month_workday':['route','month','workday']}

def reconcile(v,b,p,mode,blend,cancel):
    x=v.copy();x['b']=np.asarray(b);x['p']=np.asarray(p)
    keys=MODES[mode]
    levels=x.groupby(keys)[['b','p']].transform('sum')
    ratio=levels.b/(levels.p+1e-9)
    q=x.p.to_numpy()*ratio.to_numpy()
    empty=levels.p.to_numpy()<1e-6;q[empty]=x.b.to_numpy()[empty]
    y=(1-blend)*x.b.to_numpy()+blend*q
    if cancel:y*=1-.975*x.service_cancelled.to_numpy()
    return np.maximum(0,y)

def main():
    folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
    frames={};bs={};ps={};rows=[]
    for start,end in folds:
        v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
        bs[start]=np.load(V4/(start+'.npz'))['v2']
        ps[start]=pd.read_csv(ROOT/'v5_experts'/('validation_'+start+'.csv'),sep=';').prediction.to_numpy()
        for mode in MODES:
            for blend in [.25,.5,.75,1]:
                for cancel in [False,True]:
                    p=reconcile(v,bs[start],ps[start],mode,blend,cancel)
                    rows.append({'fold':start,'mode':mode,'blend':blend,'cancel':cancel,'score':score(v.boardings,p)})
    r=pd.DataFrame(rows);r.to_csv(DEST/'metrics.csv',index=False)
    # Average earlier windows, not hidden-score search. Autumn already used in research.
    rank=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby(['mode','blend','cancel']).score.mean().sort_values(ascending=False)
    mode,blend,cancel=rank.index[0]
    report={'selected':{'mode':mode,'blend':float(blend),'cancel':bool(cancel)},'scores':[],
        'selection':'Jul-Aug and Aug-Sep tuning. Idea motivated by failed public submission; all periods previously explored. No independent hidden-score guarantee.',
        'known_public_scores':{'baseline':.88882,'unanchored_experts':.87200},'new_public_score':None}
    for start,end in folds:
        v=frames[start];p=reconcile(v,bs[start],ps[start],mode,blend,cancel)
        report['scores'].append({'fold':start,'v2':score(v.boardings,bs[start]),'unanchored':score(v.boardings,ps[start]),'anchored':score(v.boardings,p)})
        out=v[KEY+['boardings']].copy();out['prediction']=p;out.to_csv(DEST/('validation_'+start+'.csv'),sep=';',index=False)
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    v=d[d.date>='2025-11-01'].reset_index(drop=True)
    b=pd.read_csv(ROOT/'v2/submission.csv',sep=';');p=pd.read_csv(ROOT/'v5_experts/submission.csv',sep=';')
    assert b[KEY].equals(v[KEY]) and p[KEY].equals(v[KEY])
    pred=reconcile(v,b.prediction.to_numpy(),p.prediction.to_numpy(),mode,blend,cancel)
    out=v[KEY].copy();out['prediction']=np.maximum(0,np.rint(pred)).astype(int)
    assert len(out)==14640 and not out.duplicated(KEY).any()
    out.to_csv(DEST/'submission.csv',sep=';',index=False)
    print('DONE V8',flush=True)

if __name__=='__main__':main()
