"""Select and assess separately trained daily and whole-week recursive networks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import score
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
DEST = artifact_directory('v16_weekly')
KEY = ['route','date','hour']
FOLDS = ['2025-07-01','2025-08-01','2025-09-01','2025-10-01']
SEED = 20260926
EPOCHS = 80


def forecasts(family,origin,model):
    tag = f'{origin}_{model}_s{SEED}_e{EPOCHS}'
    frame = pd.read_csv(artifact_directory(family)/(tag+'.csv.gz'))
    # Use the exact stored float32 predictions, promoted to float64 BEFORE
    # blending. CSV display precision must not decide rounding at x.5.
    with np.load(artifact_directory(family)/(tag+'.npz')) as stored:
        for path in stored.files:
            frame['step'+path] = stored[path].astype(np.float64)
    return frame


def align(frame,path):
    old = pd.read_csv(path,sep=';')
    p = frame[KEY].merge(old[KEY+['prediction']],on=KEY,how='left',validate='one_to_one').prediction
    assert p.notna().all()
    return p.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select',action='store_true')
    parser.add_argument('--export',action='store_true')
    parser.add_argument('--stable',action='store_true',help='Select the volume-bounded architecture after the drift diagnostic')
    args = parser.parse_args()
    selection_file = DEST/'dual_selection.json'
    if args.select:
        names = {}; ranks = {}
        for family,path in [('v15_cascade','step1'),('v16_weekly','step7')]:
            m = pd.read_csv(artifact_directory(family)/f'metrics_s{SEED}_e{EPOCHS}.csv')
            m = m[(m.part=='all') & m.origin.isin(FOLDS[:2]) & (m.path==path) & (m.model!='long_only')]
            if args.stable:
                m = m[m.model.str.contains('stable')]
            assert m.groupby('model').origin.nunique().eq(2).all()
            rank = m.groupby('model').score.mean().sort_values(ascending=False)
            names[family] = rank.index[0]; ranks[family] = rank.to_dict()
        pairs = {}
        for origin in FOLDS[:2]:
            daily = forecasts('v15_cascade',origin,names['v15_cascade'])
            weekly = forecasts('v16_weekly',origin,names['v16_weekly'])
            pd.testing.assert_frame_equal(daily[KEY+['boardings']],weekly[KEY+['boardings']])
            pairs[origin] = (daily,weekly)
        blend_rank = []
        for weight in np.linspace(0,1,11):
            s = [score(d.boardings,weight*d.step1+(1-weight)*w.step7) for d,w in pairs.values()]
            blend_rank.append({'daily_weight':float(weight),'mean_score':float(np.mean(s))})
        blend_rank.sort(key=lambda x:-x['mean_score'])
        choice = {'daily_model':names['v15_cascade'],'weekly_model':names['v16_weekly'],
                  'daily_path':'step1','weekly_path':'step7','daily_weight':blend_rank[0]['daily_weight'],
                  'selection_origins':FOLDS[:2],'seed':SEED,'epochs':EPOCHS,'component_rankings':ranks,
                  'weight_ranking':blend_rank,'v8_is_component':False,'v13_is_component':False}
        choice['volume_bounded_architectures'] = args.stable
        choice['architecture_development_note'] = 'Volume bounds introduced after observing September drift; no late period is an independent holdout.'
        selection_file.write_text(json.dumps(choice,indent=2))
        print('DUAL SELECTION',json.dumps(choice,indent=2),flush=True)
        return
    choice = json.loads(selection_file.read_text())
    a = choice['daily_weight']; records = []; errors = []; improvements = []
    for origin in FOLDS:
        daily = forecasts('v15_cascade',origin,choice['daily_model'])
        weekly = forecasts('v16_weekly',origin,choice['weekly_model'])
        pd.testing.assert_frame_equal(daily[KEY+['boardings']],weekly[KEY+['boardings']])
        y = daily.boardings.to_numpy(); dp = daily.step1.to_numpy(); wp = weekly.step7.to_numpy()
        combined = a*dp+(1-a)*wp
        ps = {'daily_network':dp,'weekly_network':wp,'daily_weekly_ensemble':combined,
              'v8':align(daily,artifact_directory('v8')/('validation_'+origin+'.csv')),
              'v13':align(daily,artifact_directory('v13_anchored')/('selected_validation_'+origin+'.csv'))}
        days = (pd.to_datetime(daily.date)-pd.Timestamp(origin)).dt.days.to_numpy()
        for name,p in ps.items():
            for part,mask in [('all',days>=0),('first_7_days',days<7),('days_8_28',(days>=7)&(days<28)),
                              ('days_29_61',days>=28),('last_7_days',days>=days.max()-6)]:
                records.append(dict(origin=origin,model=name,part=part,score=score(y[mask],p[mask])))
        errors.append({'origin':origin,'daily_weekly_error_correlation':float(np.corrcoef(dp-y,wp-y)[0,1]),
                       'ensemble_improvement_over_better_component':score(y,combined)-max(score(y,dp),score(y,wp))})
        improvements.append({'origin':origin,'vs_v13':score(y,combined)-score(y,ps['v13']),
                             'vs_v8':score(y,combined)-score(y,ps['v8'])})
        out = daily[KEY+['boardings']].copy();out['prediction']=combined
        out.to_csv(DEST/('dual_validation_'+origin+'.csv'),sep=';',index=False)
    r = pd.DataFrame(records);r.to_csv(DEST/'dual_metrics.csv',index=False)
    print(r[r.part=='all'].pivot(index='model',columns='origin',values='score').round(6).to_string(),flush=True)
    report = {'selection':choice,'scores':records,'error_diversity':errors,'improvements':improvements,
              'public_score':None,'public_best':.89214,'v13_user_reported_score':.89209,
              'caveat':'Previously explored overlapping development periods. October is 31 days. '
                       'Two components trained independently; each rolls forward on its own predictions. '
                       'All forecast weights frozen on July and August origins.'}
    if args.export:
        daily = forecasts('v15_cascade','2025-11-01',choice['daily_model'])
        weekly = forecasts('v16_weekly','2025-11-01',choice['weekly_model'])
        pd.testing.assert_frame_equal(daily[KEY],weekly[KEY])
        template = pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
        variants = {'submission_dual_recursive_v16.csv':a*daily.step1.to_numpy()+(1-a)*weekly.step7.to_numpy(),
                    'submission_daily_recursive_v15.csv':daily.step1.to_numpy(),
                    'submission_weekly_recursive_v16.csv':weekly.step7.to_numpy()}
        report['submissions'] = {}
        for name,p in variants.items():
            assert np.isfinite(p).all() and (p>=0).all()
            x = daily[KEY].copy();x['prediction']=np.rint(p).astype(np.int64)
            out = template[KEY].merge(x,on=KEY,how='left',validate='one_to_one')
            assert len(out)==14640 and out[KEY].equals(template[KEY]) and out.prediction.notna().all()
            file = ROOT/'outputs'/name
            out.to_csv(file,sep=';',index=False,lineterminator='\n')
            (DEST/name).write_bytes(file.read_bytes())
            report['submissions'][name] = {'path':'outputs/'+name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                         'rows':len(out),'prediction_total':int(out.prediction.sum()),'public_score':None}
        report['experimental_submission'] = 'submission_dual_recursive_v16.csv'
        report['promotion_to_public_best'] = False
        report['validation_decision'] = 'Experimental only: retain the confirmed v8 file; the September-October regression remains unresolved.'
        print('SUBMISSIONS',json.dumps(report['submissions'],indent=2),flush=True)
    (DEST/'dual_report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
