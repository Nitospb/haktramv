"""Conservative feature-model blend; all reported folds are tuning/diagnostics."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from feature_models import DEST,KEY,fit,base
from research_v2 import adaptive
from train_model import score

d=pd.read_csv(DEST/'training_matrix.csv')
names=['v2','cat_full_6_1_MAE','cat_full_8_1_MAE','cat_full_8_1_RMSE','cat_weather_8_1_MAE','base_1_112_0.5']
folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
xs={};ys={};vs={}
for start,end in folds:
    cache=np.load(DEST/(start+'.npz'));xs[start]=np.stack([cache[n] for n in names],axis=1)
    vs[start]=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);ys[start]=vs[start].boardings.to_numpy()
weights_fold=[.2,.3,.5]
x=np.concatenate([xs[f]*w/ys[f].sum() for (f,_),w in zip(folds[:3],weights_fold)])
y=np.concatenate([ys[f]*w/ys[f].sum() for (f,_),w in zip(folds[:3],weights_fold)])
initial=np.array([.5,.25,.25,0,0,0])
res=minimize(lambda w:abs(x@w-y).sum()+.002*np.square(w-initial).sum(),initial,
    method='SLSQP',bounds=[(.25,.75)]+[(0,.75)]*(len(names)-1),constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-10})
w=res.x if res.success else initial;w[w<.005]=0;w/=w.sum()
report={'weights':{n:float(v) for n,v in zip(names,w) if v>0},'evaluation':'Tuned on Jul-Aug, Aug-Sep, Sep-Oct with respective weights .2/.3/.5. October is one-month diagnostic; previously explored. Retrospective supplied weather and events.','scores':[],'public_score':None}
for f,_ in folds:
    p=xs[f]@w;report['scores'].append({'fold':f,'v2':score(ys[f],xs[f][:,0]),'ensemble':score(ys[f],p)})
    out=vs[f][KEY+['boardings']].copy();out['prediction']=p;out.to_csv(DEST/('ensemble_validation_'+f+'.csv'),sep=';',index=False)
(DEST/'ensemble_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
tr=d[d.date<'2025-11-01'];v=d[d.date>='2025-11-01'].copy();p=np.zeros(len(v))
for n,a in zip(names,w):
    if a==0:continue
    print('Final fit',n,a,flush=True)
    if n=='v2':pred=adaptive(tr,v,28,.5)
    elif n.startswith('base'):pred=base(tr,v,1,112,.5)
    else:
        _,group,depth,power,loss=n.split('_');pred=fit(tr,v,(group,int(depth),float(power),loss),DEST/(n+'.cbm'))
    p+=a*pred
out=v[KEY].copy();out['prediction']=np.rint(np.maximum(0,p)).astype(int)
assert len(out)==14640 and not out.duplicated(KEY).any() and out.prediction.notna().all()
out.to_csv(DEST/'submission_ensemble.csv',sep=';',index=False)
print('DONE ENSEMBLE',flush=True)
