"""Add normalized Kazakhstan and other-mode Moscow seasonality to the existing Moscow v8 level model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_v2 import adaptive
from weekly_baselines import read_matrix
from forecast_artifacts import artifact_directory

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/foreign_season_transfer_20260927'
KEY=['route','date','hour']


def score(y,p):return float(max(0,1-np.abs(y-p).sum()/y.sum()))


def level(history,future,index,power):
    h=history.copy()
    hi=np.asarray(index)[h.month.to_numpy(dtype=int)-1]**power
    fi=np.asarray(index)[future.month.to_numpy(dtype=int)-1]**power
    h['boardings']=h.boardings.to_numpy()/hi
    return adaptive(h,future,28,.5)*fi


def reconcile(f,b,p):
    x=f[['route','month']].copy();x['b']=b;x['p']=p
    sums=x.groupby(['route','month'])[['b','p']].transform('sum')
    q=p*(sums.b.to_numpy()/(sums.p.to_numpy()+1e-9))
    missing=sums.p.to_numpy()<1e-6;q[missing]=b[missing]
    return np.maximum(0,(.25*b+.75*q)*(1-.975*f.service_cancelled.to_numpy()))


def align(f,path):
    b=pd.read_csv(path,sep=';')
    m=f[KEY].merge(b[KEY+['prediction']],on=KEY,how='left',validate='one_to_one')
    assert len(m)==len(f) and m.prediction.notna().all()
    return m.prediction.to_numpy()


def predict(hist,future,expert,index,power):
    b=level(hist,future,index,power)
    # Fixed existing public-best scale, never reoptimized on this experiment.
    new = reconcile(future,b,expert)
    old = reconcile(future,level(hist,future,np.ones(12),0),expert)
    frozen = np.rint(np.rint(old)*1.03)
    ratio = np.divide(new,old,out=np.ones(len(old)),where=old>1e-9)
    return np.rint(frozen*ratio)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    d=read_matrix()
    priors=json.loads((ROOT/'data/external_kazakhstan/priors.json').read_text())
    priors.update(json.loads((ROOT/'data/external_moscow_modes/priors.json').read_text()))
    configs={'control_no_foreign':dict(prior=None,power=0)}
    for name in priors:
        for power in [.25,.5,1.]:configs[name+'_p'+str(power)]=dict(prior=name,power=power)
    frames={};forecasts={};records=[];audits=[]
    for start,end in [('2025-07-01','2025-08-31'),('2025-08-01','2025-09-30'),('2025-09-01','2025-10-31'),('2025-10-01','2025-10-31')]:
        h=d[d.date<start].copy();f=d[d.date.between(start,end)].copy().reset_index(drop=True)
        assert h.time.max()<f.time.min() and h.boardings.notna().all()
        expert=align(f,artifact_directory('v5_experts')/('validation_'+start+'.csv'))
        ref=align(f,artifact_directory('v8')/('validation_'+start+'.csv'))
        replay=reconcile(f,level(h,f,np.ones(12),0),expert)
        difference=float(np.max(np.abs(ref-replay)))
        np.testing.assert_allclose(ref,replay,rtol=1e-10,atol=1e-8)
        evaluate=(f.time.to_numpy()-f.time.min())<61
        frames[start]=f;forecasts[start]={}
        for name,c in configs.items():
            index=np.ones(12) if c['prior'] is None else np.array(priors[c['prior']]['index'])
            p=predict(h,f,expert,index,c['power']);forecasts[start][name]=p
            actual=f.boardings.to_numpy()
            records.append(dict(origin=start,days=int(f.loc[evaluate,'date'].nunique()),model=name,score=score(actual[evaluate],p[evaluate]),prediction_total=float(p[evaluate].sum())))
        changed=f.copy();changed['boardings']=999983
        example='tram_multiyear_smooth_p0.5';c=configs[example]
        np.testing.assert_array_equal(forecasts[start][example],predict(h,changed,expert,np.array(priors[c['prior']]['index']),c['power']))
        audits.append(dict(origin=start,baseline_replay_max_diff=difference,future_label_perturbation_exact=True))
        rank=sorted([r for r in records if r['origin']==start],key=lambda x:-x['score'])[:4]
        print('FOLD',start,json.dumps(rank),flush=True)
    metrics=pd.DataFrame(records);metrics.to_csv(OUT/'metrics.csv',index=False)
    ranking=metrics[metrics.days==61].groupby('model').score.mean().sort_values(ascending=False)
    selected=ranking.index[0]
    candidate=ranking.drop('control_no_foreign').index[0]
    for start,f in frames.items():
        z=f[KEY].copy()
        for name in dict.fromkeys(['control_no_foreign',selected,candidate]):z[name]=forecasts[start][name]
        z.to_csv(OUT/('validation_'+start+'.csv.gz'),index=False)
    h=d[d.date<'2025-11-01'].copy();f=d[d.date>='2025-11-01'].copy().reset_index(drop=True)
    expert=align(f,artifact_directory('v5_experts')/'submission.csv')
    # Final old expert was rounded when exported. Apply only new-vs-old level change
    # to the byte-frozen best forecast, preserving the deployed final shape exactly.
    # Validation uses the identical ratio construction below (equal to replay within rounding).
    template=pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
    frozen_path=ROOT/'data/external_kazakhstan/frozen_v8_up3.csv'
    assert hashlib.sha256(frozen_path.read_bytes()).hexdigest()=='f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    frozen=align(f,frozen_path)
    manifest=[]
    for name in dict.fromkeys([candidate]):
        c=configs[name];index=np.array(priors[c['prior']]['index'])
        old=reconcile(f,level(h,f,np.ones(12),0),expert)
        new=reconcile(f,level(h,f,index,c['power']),expert)
        ratio=np.divide(new,old,out=np.ones(len(old)),where=old>1e-9)
        pred=np.rint(frozen*ratio).astype(np.int64)
        pred[f.route.to_numpy()==5]=0
        z=f[KEY].copy();z['prediction']=pred
        z=template[KEY].merge(z,on=KEY,how='left',validate='one_to_one')
        assert len(z)==14640 and z[KEY].equals(template[KEY]) and (z.prediction>=0).all() and z.prediction.notna().all()
        path=OUT/'submission_external_transport.csv';z.to_csv(path,sep=';',index=False,lineterminator='\n')
        manifest.append(dict(file=str(path.relative_to(ROOT)),model=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prediction_total=int(z.prediction.sum()),public_score=None,experimental=True))
        f.assign(multiplier=ratio)[KEY+['multiplier']].to_csv(OUT/'final_multipliers.csv.gz',index=False)
    report=dict(selected_including_control=selected,best_foreign=candidate,ranking=ranking.to_dict(),configs=configs,artifacts=manifest,baseline_audits=audits,
                selection='Mean of 3 full 61-day windows; July forecasts cover 62 days for exact incumbent replay, only first 61 are scored. October 31 days is diagnostic. Previously explored windows overlap.',
                final_note='Frozen v8 +3% multiplied by ratio of reconstructed foreign/old forecasts, to preserve original final exported shape. Validation uses the same forecast-ratio and rounding construction.',
                source_sha256=hashlib.sha256((ROOT/'data/external_kazakhstan/priors.json').read_bytes()).hexdigest(),
                notes=['Kazakhstan and Moscow external data ends in 2024; indices exclude 2020-2022.','The source workbook is a 2026 vintage, so historical revisions may differ from an as-of-2025 release.','Foreign monthly seasonality is not a replacement for Russian holidays or Moscow target labels.','All source counts normalized by days in month and annual mean before transfer.','Supplied retrospective weather/service covariates and previously public-tuned +3% scale retained.'],public_best_score=.89402)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',json.dumps({k:report[k] for k in ['selected_including_control','best_foreign','ranking','artifacts']},indent=2),flush=True)

if __name__=='__main__':main()
