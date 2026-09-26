"""Reproduce final external-prior submission and audit its dependencies."""
import hashlib
import json
import numpy as np
import pandas as pd
from foreign_season_transfer import ROOT,OUT,KEY,align,level,reconcile,read_matrix,artifact_directory,score


def main():
    report=json.loads((OUT/'report.json').read_text())
    priors=json.loads((ROOT/'data/external_kazakhstan/priors.json').read_text())
    priors.update(json.loads((ROOT/'data/external_moscow_modes/priors.json').read_text()))
    assert all(set(v['years']).isdisjoint({2020,2021,2022}) and max(v['years'])<2025 for v in priors.values())
    d=read_matrix();h=d[d.date<'2025-11-01'].copy();f=d[d.date>='2025-11-01'].copy().reset_index(drop=True)
    name=report['best_foreign'];c=report['configs'][name];index=np.array(priors[c['prior']]['index'])
    expert=align(f,artifact_directory('v5_experts')/'submission.csv')
    frozen=align(f,ROOT/'data/external_kazakhstan/frozen_v8_up3.csv')
    old=reconcile(f,level(h,f,np.ones(12),0),expert)
    new=reconcile(f,level(h,f,index,c['power']),expert)
    ratio=np.divide(new,old,out=np.ones(len(old)),where=old>1e-9)
    changed=f.copy();changed['boardings']=999983
    np.testing.assert_array_equal(new,reconcile(changed,level(h,changed,index,c['power']),expert))
    out=f[KEY].copy();out['prediction']=np.rint(frozen*ratio).astype(np.int64);out.loc[out.route==5,'prediction']=0
    template=pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
    out=template[KEY].merge(out,on=KEY,how='left',validate='one_to_one')
    path=ROOT/report['artifacts'][0]['file'];assert out.to_csv(sep=';',index=False,lineterminator='\n').encode()==path.read_bytes()
    assert len(out)==14640 and not out.duplicated(KEY).any() and out.prediction.notna().all() and (out.prediction>=0).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['artifacts'][0]['sha256']
    checks=[];metrics=pd.read_csv(OUT/'metrics.csv')
    for file in sorted(OUT.glob('validation_*.csv.gz')):
        date=file.name[len('validation_'):len('validation_')+10]
        v=pd.read_csv(file);x=v.merge(d[KEY+['boardings','time']],on=KEY,how='left',validate='one_to_one')
        mask=x.time-x.time.min()<61
        for model in v.columns.drop(KEY):
            actual=score(x.loc[mask,'boardings'].to_numpy(),x.loc[mask,model].to_numpy())
            saved=float(metrics[(metrics.origin==date)&(metrics.model==model)].score.iloc[0])
            assert abs(saved-actual)<1e-12
            checks.append(dict(origin=date,model=model,score=actual))
    result=dict(final_submission_byte_exact=True,final_future_label_perturbation_exact=True,
                all_priors_pre_2025=True,excluded_years=[2020,2021,2022],validation_scores_recomputed=checks,
                sha256=report['artifacts'][0]['sha256'])
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
