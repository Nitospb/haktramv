"""Replay exported forecast, audit chronology and recompute saved backtest scores."""
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from gradient_recency_study import Experiment, OUT, ROOT, KEY, metric


def main():
    report=json.loads((OUT/'report.json').read_text())
    e=Experiment()
    for origin,horizon in [(56,61),(181,61),(243,61),(304,61)]:
        e.verify_causal_inputs(origin,horizon)
    audits=[]
    for p in sorted(OUT.glob('*.json')):
        if p.name in ['report.json','verification.json']:
            continue
        a=json.loads(p.read_text())
        if 'max_training_time' not in a:
            continue
        assert a['max_training_time']<a['origin']
        assert 'boardings' not in a['features']
        assert hashlib.sha256(p.with_suffix('.txt').read_bytes()).hexdigest()==a['model_sha256']
        audits.append(p.name)
    checks=[]
    for row in report['metrics']:
        if row['model'].startswith('v8_'):
            continue
        f=pd.read_csv(OUT/(row['origin']+'_'+row['model']+'.csv.gz'))
        merged=f.merge(e.d[KEY+['boardings']],on=KEY,how='left',validate='one_to_one')
        assert len(merged)==row['horizon']*240 and merged.boardings.notna().all()
        result=metric(merged.boardings.to_numpy(),merged.prediction.to_numpy())
        assert abs(result-row['score'])<1e-12
        checks.append(dict(origin=row['origin'],model=row['model'],score=result))
    name=report['selected']; config=report['configs'][name]
    audit=json.loads((OUT/('final_'+name+'.json')).read_text())
    future=e.d.iloc[304*240:].copy()
    if config['mode']=='raw':
        x=e.raw_input(future);base=np.zeros(len(future))
    else:
        x,base,_=e.input(304,61)
    x=x[audit['features']]
    model=lgb.Booster(model_file=str(OUT/('final_'+name+'.txt')))
    p=e.s.post(model.predict(x,num_threads=6)+base,304,61)
    expected=future[KEY].copy();expected['prediction']=np.rint(p.reshape(-1)).astype(np.int64)
    template=pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
    expected=template[KEY].merge(expected,on=KEY,how='left',validate='one_to_one')
    path=ROOT/report['submission']['file'];actual=pd.read_csv(path,sep=';')
    pd.testing.assert_frame_equal(expected,actual)
    assert expected.to_csv(sep=';',index=False,lineterminator='\n').encode()==path.read_bytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==report['submission']['sha256']
    result=dict(checkpoint_audits=len(audits),validation_scores_recomputed=len(checks),
                future_label_perturbation_origins=[56,181,243,304],byte_exact_submission_replay=True,
                submission_sha256=report['submission']['sha256'],validation_scores=checks)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='validation_scores'},indent=2))

if __name__=='__main__':
    main()
