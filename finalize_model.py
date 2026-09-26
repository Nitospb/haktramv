"""Production selection after the initial ensemble failed the autumn holdout.

All three folds are now tuning data; scores MUST NOT be called independent tests.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from advanced_train import fit_predict
from train_model import load, calendar, DATA, OUT, score

d=load()
metrics=pd.read_csv(OUT/'advanced_metrics.csv')
names=[n for n in metrics.model.unique() if not n.startswith('ridge') and 'Poisson' not in n]
folds=[('2025-05-01','2025-06-30'),('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31')]
xs=[];ys=[];frames=[]
for start,end in folds:
    v=d[(d.date>=start)&(d.date<=end)]
    frames.append(v)
    ys.append(v.boardings.values)
    xs.append(np.stack([np.load(OUT/('cache_'+start+'_'+n+'.npy')) for n in names],axis=1))
x=np.concatenate([a/b.sum() for a,b in zip(xs,ys)])
y=np.concatenate([b/b.sum() for b in ys])
w0=np.zeros(len(names));w0[np.argmin(abs(x-y[:,None]).sum(axis=0))]=1
res=minimize(lambda w:abs(x@w-y).sum(),w0,method='SLSQP',bounds=[(0,1)]*len(names),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':500,'ftol':1e-10})
w=res.x if res.success else w0;w[w<.005]=0;w/=w.sum()
report={'evaluation_type':'TUNING scores: all three folds used for model selection; no remaining independent holdout',
        'weights':{n:float(a) for n,a in zip(names,w) if a>0},
        'fold_scores':{f[0]:score(b,a@w) for f,a,b in zip(folds,xs,ys)},
        'target_score':.95,'hidden_competition_score':None,
        'initial_independent_holdout_score':json.loads((OUT/'advanced_report.json').read_text())['holdout_score']}
report['target_reached_on_all_tuning_folds']=all(s>=.95 for s in report['fold_scores'].values())
(OUT/'production_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
p=np.zeros(len(future))
for name,weight in zip(names,w):
    if weight>0:
        print('Refit',name,weight,flush=True)
        p+=weight*fit_predict(name,d,future,save=True)
sub=future[['route','date','hour']].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
assert len(sub)==14640 and sub.prediction.notna().all() and not sub.duplicated(['route','date','hour']).any()
sub.to_csv(OUT/'submission.csv',sep=';',index=False)
for f,v,a in zip(folds,frames,xs):
    v=v[['route','date','hour','boardings']].copy();v['prediction']=a@w
    v.to_csv(OUT/('production_validation_'+f[0]+'.csv'),sep=';',index=False)
print('DONE',flush=True)
