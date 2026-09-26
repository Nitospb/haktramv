"""Stable group-specific shape blends with incumbent route-month volumes preserved."""
import json
import numpy as np
import pandas as pd
from reconcile_forecasts import reconcile,ROOT,KEY
from feature_models import DEST as V4
from train_model import score

DEST=ROOT/'v9';DEST.mkdir(exist_ok=True)
d=pd.read_csv(V4/'training_matrix.csv')
folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
frames={};bs={};ps={};incumbents={}
for start,end in folds:
    v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
    bs[start]=np.load(V4/(start+'.npz'))['v2']
    ps[start]=pd.read_csv(ROOT/'v5_experts'/('validation_'+start+'.csv'),sep=';').prediction.to_numpy()
    incumbents[start]=reconcile(v,bs[start],ps[start],'month',.75,True)

def predict(v,b,p,weights,reference):
    x=v.copy();x['base']=b;x['new']=p;x['reference']=reference
    totals=x.groupby(['route','month'])[['base','new']].transform('sum')
    ratio=totals.base/(totals.new+1e-9)
    shape=x.new.to_numpy()*ratio.to_numpy()
    shape[totals.new.to_numpy()<1e-6]=x.base.to_numpy()[totals.new.to_numpy()<1e-6]
    alpha=np.array([weights[(int(r),int(w))] for r,w in zip(x.route,x.workday)])
    x['prediction']=((1-alpha)*x.base.to_numpy()+alpha*shape)*(1-.975*x.service_cancelled.to_numpy())
    t=x.groupby(['route','month'])[['prediction','reference']].transform('sum')
    x['prediction']*=t.reference/(t.prediction+1e-9)
    out=x.prediction.to_numpy(copy=True);out[t.prediction.to_numpy()<1e-6]=x.reference.to_numpy()[t.prediction.to_numpy()<1e-6]
    return np.maximum(0,out)

def main():
    weights={(int(r),int(w)):.75 for r in d.route.unique() for w in [0,1]}
    changes=[];trainfolds=[f[0] for f in folds[:2]]
    # A change must improve both earlier periods by at least 0.1 percentage point
    # on the corresponding route before being halfway shrunk toward the incumbent.
    for group in sorted(weights):
        r,workday=group
        if r==5:continue
        before={f:predict(frames[f],bs[f],ps[f],weights,incumbents[f]) for f in trainfolds}
        candidates=[]
        for a in [.25,.5,1.]:
            trial=dict(weights);trial[group]=a;gains=[]
            for f in trainfolds:
                v=frames[f];m=v.route==r;y=v.loc[m,'boardings'].to_numpy()
                p=predict(v,bs[f],ps[f],trial,incumbents[f])
                gains.append(score(y,p[m])-score(y,before[f][m]))
            if min(gains)>.001:candidates.append((np.mean(gains),a,gains))
        if candidates:
            gain,a,gains=max(candidates);weights[group]=(.75+a)/2
            changes.append({'route':r,'workday':workday,'weight':weights[group],'unshrunk_weight':a,'earlier_route_gains':gains})
    rows=[]
    for f,_ in folds:
        v=frames[f];p=predict(v,bs[f],ps[f],weights,incumbents[f])
        rows.append({'fold':f,'incumbent':score(v.boardings,incumbents[f]),'refined':score(v.boardings,p)})
        out=v[KEY+['boardings']].copy();out['prediction']=p;out.to_csv(DEST/('validation_'+f+'.csv'),sep=';',index=False)
    report={'changes':changes,'scores':rows,'weights':{str(r)+'_'+str(w):a for (r,w),a in weights.items()},'selection':'Jul-Aug and Aug-Sep require improvement in both; weights shrunk by half. All periods previously explored. Route-month totals equal v8.','known_public_incumbent':.89214,'new_public_score':None}
    report['no_autumn_degradation']=all(row['refined']>=row['incumbent'] for row in rows[2:])
    (DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    v=d[d.date>='2025-11-01'].reset_index(drop=True)
    b=pd.read_csv(ROOT/'v2/submission.csv',sep=';');p=pd.read_csv(ROOT/'v5_experts/submission.csv',sep=';');ref=pd.read_csv(ROOT/'v8/submission.csv',sep=';')
    assert all(s[KEY].equals(v[KEY]) for s in [b,p,ref])
    pred=predict(v,b.prediction.to_numpy(),p.prediction.to_numpy(),weights,ref.prediction.to_numpy())
    # Largest-remainder rounding preserves the successful route-month totals exactly.
    out=v[KEY].copy();out['prediction']=np.floor(np.maximum(0,pred)).astype(int)
    for _,idx in v.groupby(['route','month']).groups.items():
        ix=np.asarray(list(idx));need=int(ref.iloc[ix].prediction.sum()-out.iloc[ix].prediction.sum())
        assert 0<=need<=len(ix)
        take=ix[np.argsort(-(pred[ix]-np.floor(pred[ix])),kind='stable')[:need]]
        out.loc[take,'prediction']+=1
    assert len(out)==14640 and not out.duplicated(KEY).any()
    out.to_csv(DEST/'submission.csv',sep=';',index=False);print('DONE V9',flush=True)

if __name__=='__main__':main()
