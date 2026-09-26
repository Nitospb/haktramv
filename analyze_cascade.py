"""Freeze the short-transition cascade on two earlier origins; export one path."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import score
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
DEST = artifact_directory('v15_cascade')
KEY = ['route', 'date', 'hour']
FOLDS = ['2025-07-01','2025-08-01','2025-09-01','2025-10-01']


def aligned(frame, file):
    source = pd.read_csv(file, sep=';')
    p = frame[KEY].merge(source[KEY+['prediction']], on=KEY, how='left', validate='one_to_one').prediction
    assert p.notna().all()
    return p.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--stable', action='store_true')
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    suffix = f'_s{args.seed}_e{args.epochs}'
    selection_path = DEST/'selection.json'
    m = pd.read_csv(DEST/f'metrics_s{args.seed}_e{args.epochs}.csv')
    if args.select:
        early = m[(m.part == 'all') & m.origin.isin(FOLDS[:2])]
        assert early.groupby(['model','path']).origin.nunique().eq(2).all()
        ranking = early.groupby(['model','path']).score.mean().sort_values(ascending=False)
        # The long-only model is a control, not a short-transition candidate.
        eligible = ranking[[name != 'long_only' for name, path in ranking.index]]
        if args.stable:
            eligible = eligible[['stable' in name for name, path in eligible.index]]
        config, path = eligible.index[0]
        choice = {'model': config, 'path': path, 'seed': args.seed, 'epochs': args.epochs,
                  'selection_origins': FOLDS[:2], 'selection_mean': float(eligible.iloc[0]),
                  'ranking': [dict(model=n,path=p,mean=float(v)) for (n,p),v in ranking.items()],
                  'single_checkpoint': True, 'single_recursive_path': True,
                  'external_forecast_blend': False, 'direct_61_day_prediction': False}
        selection_path.write_text(json.dumps(choice, indent=2))
        print('SELECTED', json.dumps(choice, indent=2), flush=True)
        return
    choice = json.loads(selection_path.read_text())
    assert (args.seed,args.epochs) == (choice['seed'],choice['epochs'])
    records = []; stability = []; disagreements = []
    for origin in FOLDS:
        frame = pd.read_csv(DEST/(origin+'_'+choice['model']+suffix+'.csv.gz'))
        control = pd.read_csv(DEST/(origin+'_long_only'+suffix+'.csv.gz'))
        pd.testing.assert_frame_equal(frame[KEY+['boardings']], control[KEY+['boardings']])
        old = aligned(frame, artifact_directory('v13_anchored')/('selected_validation_'+origin+'.csv'))
        best = aligned(frame, artifact_directory('v8')/('validation_'+origin+'.csv'))
        predictions = {'v8': best, 'v13': old, 'cascade_step1': frame.step1.to_numpy(),
                       'cascade_step7': frame.step7.to_numpy(), 'cascade_selected': frame[choice['path']].to_numpy(),
                       'long_only_step1': control.step1.to_numpy(), 'long_only_step7': control.step7.to_numpy()}
        days = (pd.to_datetime(frame.date)-pd.Timestamp(origin)).dt.days.to_numpy()
        for name,p in predictions.items():
            for part,mask in [('all',days >= 0),('days_1_7',days < 7),('days_8_28',(days >= 7)&(days < 28)),
                              ('days_29_61',days >= 28),('last_7_days',days >= days.max()-6)]:
                records.append(dict(origin=origin,model=name,part=part,score=score(frame.boardings.to_numpy()[mask],p[mask])))
        differences = (frame.step1-frame.step7).abs().sum()/frame.boardings.sum()
        disagreements.append({'origin':origin,'daily_weekly_disagreement_per_actual_volume':float(differences)})
        selected = frame[KEY+['boardings']].copy(); selected['prediction'] = frame[choice['path']]
        selected.to_csv(DEST/('selected_validation_'+origin+'.csv'),sep=';',index=False)
        # Extra seeds are diagnostics only; they never enter the submission.
        for seed in [20260927,20260928]:
            f = DEST/f'{origin}_{choice["model"]}_s{seed}_e{args.epochs}.csv.gz'
            if f.exists():
                extra = pd.read_csv(f)
                pd.testing.assert_frame_equal(frame[KEY],extra[KEY])
                stability.append({'origin':origin,'seed':seed,'path':choice['path'],
                                  'score':score(extra.boardings,extra[choice['path']])})
    result = pd.DataFrame(records); result.to_csv(DEST/'selected_metrics.csv',index=False)
    print(result[result.part == 'all'].pivot(index='model',columns='origin',values='score').round(6).to_string(),flush=True)
    report = {'selection':choice,'scores':records,'extra_seed_diagnostics':stability,'path_disagreement':disagreements,
              'public_score':None,'v13_user_reported_public_score':.89209,'v8_public_best':.89214,
              'caveat':'Previously explored overlapping development windows. October has 31 days. '
                       'All labels after an origin stay hidden throughout the rollout. '
                       'Extra seeds and the alternate path are not averaged into the submission.'}
    if args.export:
        future = pd.read_csv(DEST/('2025-11-01_'+choice['model']+suffix+'.csv.gz'))
        template = pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
        out = template[KEY].merge(future[KEY+[choice['path']]],on=KEY,how='left',validate='one_to_one')
        p = out.pop(choice['path']).to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        out['prediction'] = np.rint(p).astype(np.int64)
        assert len(out) == 14640 and out[KEY].equals(template[KEY]) and not out[KEY].duplicated().any()
        file = ROOT/'outputs/submission_cascade_v15.csv'
        out.to_csv(file,sep=';',index=False,lineterminator='\n')
        (DEST/'submission.csv').write_bytes(file.read_bytes())
        report['submission'] = {'path':'outputs/submission_cascade_v15.csv','sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                                'rows':len(out),'prediction_total':int(out.prediction.sum()),'model':choice['model'],
                                'path_used':choice['path'],'seed':args.seed,'public_score':None,
                                'single_checkpoint':True,'no_blending':True,'v8_best_replaced':False}
        print('SUBMISSION',json.dumps(report['submission'],indent=2),flush=True)
    (DEST/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
