"""Use recent profiles only when the latest history matches forecast season."""
import json
import numpy as np
import pandas as pd
from research_v3 import DEST, weighted
from research_v2 import adaptive
from train_model import load,calendar,DATA,score

d=load()
folds=[('2025-05-01','2025-06-30'),('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]
preds={};frames={};masks={}
for start,end in folds:
    tr=d[d.date<start];v=d[(d.date>=start)&(d.date<=end)].reset_index(drop=True)
    frames[start]=v;preds[start]=dict(np.load(DEST/(start+'.npz')))
    latest_season=int(tr.loc[tr.time==tr.time.max(),'summer'].iloc[0])
    masks[start]=(v.summer.to_numpy()==latest_season)
rows=[]
for name in list(preds[folds[0][0]]):
    if name=='v2':continue
    for weight in [.25,.5,.75,1.]:
        for start,end in folds:
            v=frames[start];p=preds[start]
            blend=np.where(masks[start],weight,0)
            y=blend*p[name]+(1-blend)*p['v2']
            rows.append(dict(model=name,weight=weight,fold=start,score=score(v.boardings,y)))
metrics=pd.DataFrame(rows);metrics.to_csv(DEST/'gate_metrics.csv',index=False)
# Keep October out of configuration selection. Other folds were explored previously.
rank=metrics[metrics.fold!='2025-10-01'].groupby(['model','weight']).score.mean().sort_values(ascending=False)
name,weight=rank.index[0]
selected=metrics[(metrics.model==name)&(metrics.weight==weight)]
report={'method':'season gated recent profile','model':name,'weight':float(weight),'scores':selected.to_dict('records'),'evaluation':'May-Jun, Jul-Aug, Aug-Sep, Sep-Oct tuning; Oct one-month diagnostic. No new independent two-month test.','real_score':None,'previous_real_score_user_reported':.88882}
print(json.dumps(report,indent=2),flush=True)
(DEST/'gate_report.json').write_text(json.dumps(report,indent=2))
future=calendar(pd.read_csv(DATA/'test_submission.csv',sep=';').drop(columns='prediction'))
_,hl,pool,dec=name.split('_')
recent=weighted(d,future,int(hl),bool(int(pool)),bool(int(dec)))
base=adaptive(d,future,28,.5)
latest_season=int(d.loc[d.time==d.time.max(),'summer'].iloc[0])
w=np.where(future.summer.to_numpy()==latest_season,weight,0)
p=w*recent+(1-w)*base
sub=future[['route','date','hour']].copy();sub['prediction']=np.maximum(0,np.rint(p)).astype(int)
assert len(sub)==14640 and sub.prediction.notna().all() and not sub.duplicated(['route','date','hour']).any()
sub.to_csv(DEST/'submission_season_gate.csv',sep=';',index=False)
print('DONE GATE',flush=True)
