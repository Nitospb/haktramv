"""Select v13 on the two earlier 61-day windows, then report all diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import score

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v13_anchored'
ART = ROOT / 'outputs'
if not (ART / 'v8/submission.csv').exists():
    ART = ART / 'studio'
if not DEST.exists():
    DEST = ART / 'v13_anchored'
FOLDS = ['2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01']
KEY = ['route', 'date', 'hour']


def read_forecasts(origin, config, seed):
    return pd.read_csv(DEST / f'{origin}_{config}_s{seed}_e60.csv.gz')


def aligned(frame, path, column):
    other = pd.read_csv(path, sep=';' if path.parent.name == 'v8' else ',')
    result = frame[KEY].merge(other[KEY+[column]], on=KEY, how='left', validate='one_to_one')[column]
    assert result.notna().all()
    return result.to_numpy()


def calibrate(frame, prediction, mode):
    if mode == 'raw':
        return prediction
    x = frame[['route', 'date', 'incumbent_v8']].copy()
    x['month'] = x.date.str[:7]; x['prediction'] = prediction
    totals = x.groupby(['route', 'month'])[['prediction', 'incumbent_v8']].transform('sum')
    p = prediction * (totals.incumbent_v8/(totals.prediction+1e-9)).to_numpy()
    empty = totals.prediction.to_numpy() < 1e-6
    p[empty] = x.incumbent_v8.to_numpy()[empty]
    if mode == 'month_blend50':
        p = .5*p + .5*x.incumbent_v8.to_numpy()
    return np.maximum(0, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='20260926')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(',')))
    selection_file = DEST / 'selection.json'
    r = pd.read_csv(DEST / 'metrics_s20260926.csv')
    eligible = r[(r.part == 'all') & r.origin.isin(FOLDS[:2]) & (r.model != 'direct') &
                 r.path.str.endswith(('all_paths_mean', 'weekly_direct_mean', 'step7'))]
    candidates = []
    for config, group in eligible.groupby('model'):
        frames = {f: read_forecasts(f, config, seeds[0]) for f in FOLDS[:2]}
        for path in group.path.unique():
            for calibration in ['raw', 'month', 'month_blend50']:
                values = [score(frame.boardings, calibrate(frame, frame[path].to_numpy(), calibration))
                          for frame in frames.values()]
                candidates.append(dict(model=config, path=path, calibration=calibration, mean=float(np.mean(values))))
    rank = pd.DataFrame(candidates).sort_values('mean', ascending=False)
    assert len(rank)
    config, path, calibration = rank.iloc[0][['model', 'path', 'calibration']]
    selection = {'model': config, 'path': path, 'calibration': calibration, 'selection_origins': FOLDS[:2],
                 'selection_seed': 20260926, 'ranking': rank.head(10).to_dict('records'),
                 'public_score': None}
    selection_file.write_text(json.dumps(selection, indent=2))
    print('SELECTED', json.dumps(selection, indent=2), flush=True)
    records = []; variability = []
    for origin in FOLDS:
        frames = [read_forecasts(origin, config, seed) for seed in seeds]
        frame = frames[0]
        for other in frames[1:]:
            pd.testing.assert_frame_equal(frame[KEY+['boardings']], other[KEY+['boardings']])
        values = np.stack([other[path].to_numpy() for other in frames])
        prediction = calibrate(frame, values.mean(0), calibration)
        old_path = ART / 'v11_weekly' / f'validation_{origin}_consistent7_c5_tail2_e60.csv'
        old = aligned(frame, old_path, '7')
        ps = {'incumbent_v8': frame.incumbent_v8.to_numpy(), 'previous_consistency_c5': old,
              'new_consistency_raw': values.mean(0),
              'new_consistency_single_seed': calibrate(frame, values[0], calibration), 'new_consistency_seed_mean': prediction}
        days = (pd.to_datetime(frame.date)-pd.Timestamp(origin)).dt.days.to_numpy()
        for name, p in ps.items():
            for part, mask in [('all', days >= 0), ('last_7_days', days >= days.max()-6)]:
                records.append(dict(origin=origin, model=name, part=part,
                                    score=score(frame.boardings.to_numpy()[mask], p[mask])))
        variability.append(dict(origin=origin, seeds=seeds,
                                individual_scores=[score(frame.boardings, calibrate(frame, p, calibration)) for p in values]))
        out = frame[KEY+['boardings']].copy(); out['prediction'] = prediction
        out.to_csv(DEST / ('selected_validation_'+origin+'.csv'), sep=';', index=False)
    result = pd.DataFrame(records)
    result.to_csv(DEST / 'selected_metrics.csv', index=False)
    print(result[result.part == 'all'].pivot(index='model', columns='origin', values='score').round(6).to_string(), flush=True)
    report = {'selection': selection, 'ensemble_seeds': seeds, 'scores': records,
              'seed_variability': variability, 'public_score': None,
              'caveat': 'Previously explored, overlapping development windows; October has only 31 days. '
                        'v8 guide inherits its historical tuning. Future labels are not supplied in a rollout.'}
    if args.export:
        frames = [read_forecasts('2025-11-01', config, seed) for seed in seeds]
        frame = frames[0]
        for other in frames[1:]:
            pd.testing.assert_frame_equal(frame[KEY], other[KEY])
        prediction = calibrate(frame, np.mean([other[path].to_numpy() for other in frames], axis=0), calibration)
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        out = frame[KEY].copy(); out['prediction'] = np.maximum(0, np.rint(prediction)).astype(np.int64)
        template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
        out = template[KEY].merge(out, on=KEY, how='left', validate='one_to_one')
        assert len(out) == 14640 and out[KEY].equals(template[KEY]) and out.prediction.notna().all()
        destination = ROOT / 'outputs/submission_consistency_v13.csv'
        out.to_csv(destination, sep=';', index=False, lineterminator='\n')
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        (DEST / 'submission.csv').write_bytes(destination.read_bytes())
        report['submission'] = {'path': 'outputs/submission_consistency_v13.csv', 'sha256': digest,
                                'rows': len(out), 'prediction_total': int(out.prediction.sum()),
                                'model': config, 'forecast_path': path, 'calibration': calibration, 'seeds': seeds,
                                'public_score': None, 'active_model_replaced': False}
        print('SUBMISSION', json.dumps(report['submission'], indent=2), flush=True)
    (DEST / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
