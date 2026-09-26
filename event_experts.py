"""Keep the general model on normal service; call regime experts only on changed service."""
import json
import numpy as np
import pandas as pd
from feature_models import DEST as V4
from service_models import DEST as V5,normal_profile,adjust,fit,KEY
from train_model import score

DEST=V4.parent/'v5_experts';DEST.mkdir(exist_ok=True)
d=pd.read_csv(V4/'training_matrix.csv')
folds=[('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
arrays={};frames={};rows=[]
experts=['profile_0.5_365','profile_1_365','cat_6_weather','cat_6_full','cat_8_full']
for start,end in folds:
    v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True);frames[start]=v
    a=dict(np.load(V5/(start+'.npz')));arrays[start]=a
    base=a['v4_cancel_prior']
    for name in experts:
        for weight in [.25,.5,.75,1.]:
            m=np.maximum(v.service_rerouted.to_numpy(),v.service_reduced_frequency.to_numpy())
            p=base*(1-m*weight)+a[name]*m*weight
            rows.append(dict(fold=start,expert=name,weight=weight,score=score(v.boardings,p)))
r=pd.DataFrame(rows);r.to_csv(DEST/'metrics.csv',index=False)
rank=r[r.fold.isin(['2025-07-01','2025-08-01'])].groupby(['expert','weight']).score.mean().sort_values(ascending=False)
name,weight=rank.index[0]
chosen=r[(r.expert==name)&(r.weight==weight)]
report={'expert':name,'weight':float(weight),'scores':chosen.to_dict('records'),'selection':'Jul-Aug and Aug-Sep; regime routing conceived after inspecting autumn residuals, so scores are development diagnostics. Retrospective supplied events.','actual_competition_score':None}
(DEST/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
for start,end in folds:
    v=frames[start];a=arrays[start];m=np.maximum(v.service_rerouted.to_numpy(),v.service_reduced_frequency.to_numpy())
    p=a['v4_cancel_prior']*(1-m*weight)+a[name]*m*weight
    out=v[KEY+['boardings']].copy();out['prediction']=p;out.to_csv(DEST/('validation_'+start+'.csv'),sep=';',index=False)
tr=d[d.date<'2025-11-01'].reset_index(drop=True);v=d[d.date>='2025-11-01'].reset_index(drop=True)
if name.startswith('profile'):
    _,power,window=name.split('_');expert=adjust(tr,v,normal_profile(tr,tr,float(power),int(window)),normal_profile(tr,v,float(power),int(window)))
else:
    _,depth,group=name.split('_');expert=fit(tr,v,int(depth),group,DEST/'expert.cbm')
base=pd.read_csv(V4/'submission_ensemble.csv',sep=';').prediction.to_numpy()*(1-.975*v.service_cancelled.to_numpy())
m=np.maximum(v.service_rerouted.to_numpy(),v.service_reduced_frequency.to_numpy());p=base*(1-m*weight)+expert*m*weight
out=v[KEY].copy();out['prediction']=np.maximum(0,np.rint(p)).astype(int)
assert len(out)==14640 and not out.duplicated(KEY).any()
out.to_csv(DEST/'submission.csv',sep=';',index=False)
print('DONE EXPERTS',flush=True)
