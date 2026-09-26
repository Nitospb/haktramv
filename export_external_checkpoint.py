"""Export the autumn-best direct checkpoint as a separate exploratory candidate."""
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from fleet_forecast_study import ROOT, KEY


def main():
    out=ROOT/'outputs/longer_moscow_external_20260927'
    curves=pd.read_csv(out/'learning_curves.csv')
    autumn=curves[(curves.origin=='2025-09-01')&(curves.model=='direct')]
    row=autumn.loc[autumn.validation_score.idxmax()];n=int(row.trees)
    model=lgb.Booster(model_file=str(out/'304_all_direct.txt'))
    assert model.num_trees()>=n
    d=pd.read_parquet(ROOT/'data/complete_external_20260927/all_features_matrix.parquet')
    d=d[d.date>='2025-11-01'].copy();cols=model.feature_name()
    x=d[cols].astype(float).replace([np.inf,-np.inf],np.nan).fillna(-1)
    p=np.maximum(0,model.predict(x,num_iteration=n,num_threads=6))
    p*=np.clip(1-.975*d.service_cancelled.to_numpy(),0,1)*(d.route.to_numpy()!=5)
    sub=d[KEY].copy();sub['prediction']=np.rint(p).astype(int)
    path=out/('submission_direct_'+str(n)+'_autumn.csv');sub.to_csv(path,sep=';',index=False)
    assert len(sub)==14640 and not sub.duplicated(KEY).any() and sub.prediction.ge(0).all()
    check=pd.read_csv(path,sep=';');pd.testing.assert_frame_equal(check,sub.reset_index(drop=True))
    report=dict(file=path.name,trees=n,autumn_validation_score=float(row.validation_score),
        selection='Highest Sep-Oct score among 800/1600/3200. Separate exploratory selection; the mean-of-three-windows winner remains 3200.',
        public_score=None,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(sub),
        forecast_volume=int(sub.prediction.sum()),checkpoint_replay=True)
    (out/'autumn_checkpoint_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(report)


if __name__=='__main__':
    main()
