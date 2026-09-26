"""Regularized route-specific use of enriched models; fixed early-fold weight selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from feature_models import DEST as V4
from train_model import score

ROOT=V4.parent;DEST=ROOT/'v7';DEST.mkdir(exist_ok=True)
names=['service_experts','cancel_only','enriched','fare_cohorts']
v6name=json.loads((ROOT/'v6/report.json').read_text())['selected']
d=pd.read_csv(V4/'training_matrix.csv')
folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
frames={};xs={}
for start,end in folds:
    v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
    x1=pd.read_csv(ROOT/'v5_experts'/('validation_'+start+'.csv'),sep=';').prediction.to_numpy()
    x2=np.load(ROOT/'v5'/(start+'.npz'))['v4_cancel_prior']
    x3=np.load(ROOT/'v6'/(start+'_'+v6name+'.npy'))
    x4=np.load(ROOT/'v6_cohorts'/(start+'.npy'))
    xs[start]=np.stack([x1,x2,x3,x4],axis=1)
def optimize(x,y,prior,penalty):
    result=minimize(lambda w:np.abs(x@w-y).sum()/max(y.sum(),1)+penalty*np.square(w-prior).sum(),prior,
        method='SLSQP',bounds=[(.25,1),(0,.75),(0,.5),(0,.25)],constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':250,'ftol':1e-10})
    return result.x if result.success else prior
tuning=[f[0] for f in folds[:2]]
x=np.concatenate([xs[f] for f in tuning]);y=np.concatenate([frames[f].boardings.to_numpy() for f in tuning])
global_w=optimize(x,y,np.array([.75,.15,.1,0]),.003)
route_w={}
for route in sorted(d.route.unique()):
    x=np.concatenate([xs[f][frames[f].route==route] for f in tuning]);y=np.concatenate([frames[f].loc[frames[f].route==route,'boardings'].to_numpy() for f in tuning])
    w=optimize(x,y,global_w,.02) if y.sum()>0 else np.array([1,0,0,0])
    # Partial pooling limits changes justified by very few chronological windows.
    route_w[int(route)]=.5*w+.5*global_w
scores=[]
for start,end in folds:
    v=frames[start];p=np.zeros(len(v))
    for route,w in route_w.items():m=v.route==route;p[m]=xs[start][m]@w
    scores.append({'fold':start,'expert':score(v.boardings,xs[start][:,0]),'ensemble':score(v.boardings,p)})
    out=v[['route','date','hour','boardings']].copy();out['prediction']=p;out.to_csv(DEST/('validation_'+start+'.csv'),sep=';',index=False)
report={'global_weights':dict(zip(names,map(float,global_w))),'route_weights':{r:dict(zip(names,map(float,w))) for r,w in route_w.items()},'scores':scores,'selection':'Weights chosen on Jul-Aug and Aug-Sep; all periods were previously explored. Development diagnostics only. Retrospective supplied weather/events.','public_score':None}
(DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(scores,indent=2),flush=True)
paths=[ROOT/'v5_experts/submission.csv',ROOT/'v5/submission.csv',ROOT/'v6/submission.csv',ROOT/'v6_cohorts/submission.csv']
subs=[pd.read_csv(p,sep=';') for p in paths]
for s in subs[1:]:assert s[['route','date','hour']].equals(subs[0][['route','date','hour']])
x=np.stack([s.prediction.to_numpy() for s in subs],axis=1);v=subs[0][['route','date','hour']].copy();p=np.zeros(len(v))
for route,w in route_w.items():m=v.route==route;p[m]=x[m]@w
v['prediction']=np.maximum(0,np.rint(p)).astype(int);assert len(v)==14640 and not v.duplicated(['route','date','hour']).any()
v.to_csv(DEST/'submission.csv',sep=';',index=False);print('DONE V7',flush=True)
