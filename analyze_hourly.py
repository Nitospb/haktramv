"""Compare more frequent state updates and select a conservative combination.

Selection uses only the July and August origins. All windows have already been
explored and overlap; none is a new independent holdout. No public labels are used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_model import score
from forecast_artifacts import artifact_directory

ROOT = Path(__file__).resolve().parent
DEST = artifact_directory('v14_hourly')
FOLDS = ['2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01']
KEY = ['route', 'date', 'hour']


def read_frame(family, origin, config, seed):
    return pd.read_csv(artifact_directory(family) / f'{origin}_{config}_s{seed}_e60.csv.gz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='20260926')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    seeds = list(map(int, args.seeds.split(',')))
    selections = {}; frequency = []; ranking = []
    for family in ['v14_hourly', 'v14_shape']:
        metrics = pd.read_csv(artifact_directory(family) / 'metrics_s20260926.csv')
        eligible = metrics[(metrics.part == 'all') & metrics.origin.isin(FOLDS[:2]) &
                           metrics.path.str.endswith('_primary')]
        rank = eligible.groupby(['model', 'path']).score.mean().sort_values(ascending=False)
        config, path = rank.index[0]
        selections[family] = {'config': config, 'path': path, 'selection_mean': float(rank.iloc[0])}
        ranking.extend(dict(family=family, config=c, path=p, mean=float(v)) for (c, p), v in rank.items())
        for origin in FOLDS:
            for guide in ['profile_primary', 'v8_primary', 'v13_primary']:
                sub = metrics[(metrics.part == 'all') & (metrics.origin == origin) & (metrics.path == guide)]
                scores = sub.set_index('model').score
                frequency.append(dict(family=family, origin=origin, path=guide,
                                      h1=float(scores.h1), h24=float(scores.h24),
                                      difference=float(scores.h1-scores.h24)))
    print('SELECTED NETWORKS', json.dumps(selections, indent=2), flush=True)
    frames = {}; components = {}; variability = []
    for origin in FOLDS:
        comp = {}
        for family, chosen in selections.items():
            fs = [read_frame(family, origin, chosen['config'], seed) for seed in seeds]
            f = fs[0]
            for other in fs[1:]:
                pd.testing.assert_frame_equal(f[KEY], other[KEY])
            if origin in frames:
                pd.testing.assert_frame_equal(frames[origin][KEY], f[KEY])
            frames[origin] = f
            predictions = np.stack([other[chosen['path']].to_numpy() for other in fs])
            comp[family] = predictions.mean(0)
            if origin != '2025-11-01':
                variability.append(dict(origin=origin, family=family, seeds=seeds,
                                        scores=[score(f.boardings, p) for p in predictions]))
        comp['v8'] = f.incumbent_v8.to_numpy()
        comp['v13'] = f.previous_v13.to_numpy()
        comp['v8_v13_equal'] = .5*comp['v8']+.5*comp['v13']
        components[origin] = comp
    # A small predefined family of convex combinations, including a zero hourly
    # contribution. A new component is not forced in if it fails validation.
    blends = []
    for anchor in ['v8', 'v13', 'v8_v13_equal']:
        for family in ['v14_hourly', 'v14_shape']:
            for weight in [0, .1, .25, .5, .75, 1]:
                scores = []
                for origin in FOLDS[:2]:
                    c = components[origin]
                    p = (1-weight)*c[anchor]+weight*c[family]
                    scores.append(score(frames[origin].boardings, p))
                blends.append(dict(anchor=anchor, family=family, hourly_weight=weight,
                                   mean=float(np.mean(scores))))
    blends.sort(key=lambda a: (-a['mean'], a['hourly_weight']))
    frozen_file = DEST / 'selection_seed20260926.json'
    if seeds == [20260926]:
        chosen = blends[0]
        frozen_file.write_text(json.dumps({'networks': selections, 'combination': chosen,
                                           'selection_origins': FOLDS[:2], 'seed': 20260926}, indent=2))
    else:
        frozen = json.loads(frozen_file.read_text())
        assert frozen['networks'] == selections
        chosen = frozen['combination']
    print('SELECTED COMBINATION', json.dumps(chosen), flush=True)
    records = []; deltas = []
    for origin, c in components.items():
        f = frames[origin]
        prediction = (1-chosen['hourly_weight'])*c[chosen['anchor']]+chosen['hourly_weight']*c[chosen['family']]
        c['selected'] = prediction
        out = f[KEY+['boardings']].copy(); out['prediction'] = prediction
        out.to_csv(DEST / ('selected_validation_'+origin+'.csv'), sep=';', index=False)
        if origin == '2025-11-01':
            continue
        days = (pd.to_datetime(f.date)-pd.Timestamp(origin)).dt.days.to_numpy()
        for name, p in c.items():
            for part, mask in [('all', days >= 0), ('last_7_days', days >= days.max()-6)]:
                records.append(dict(origin=origin, model=name, part=part,
                                    score=score(f.boardings.to_numpy()[mask], p[mask])))
        deltas.append(dict(origin=origin, selected_minus_v8=score(f.boardings, prediction)-score(f.boardings, c['v8']),
                           selected_minus_v13=score(f.boardings, prediction)-score(f.boardings, c['v13'])))
    pd.DataFrame(records).to_csv(DEST / 'selected_metrics.csv', index=False)
    pd.DataFrame(frequency).to_csv(DEST / 'frequency_comparison.csv', index=False)
    report = {'network_selections': selections, 'selection_origins': FOLDS[:2],
              'selection_seed': 20260926, 'selection_frozen_before_extra_seeds': True, 'network_ranking': ranking,
              'combination': chosen, 'combination_ranking': blends,
              'seeds': seeds, 'seed_variability': variability, 'scores': records,
              'deltas': deltas, 'frequency_comparison': frequency,
              'public_score': None, 'best_public_score': .89214, 'previous_v13_public_score': .89209,
              'caveat': 'Previously explored overlapping development windows, not independent holdouts. '
                        'October has only 31 days. Guidance uses saved forecasts, not hidden actuals.'}
    print(pd.DataFrame(records).query("part == 'all'").pivot(index='model', columns='origin', values='score').round(6).to_string(), flush=True)
    if args.export:
        template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
        c = {}
        for name, folder in [('v8', 'v8'), ('v13', 'v13_anchored')]:
            f = pd.read_csv(artifact_directory(folder) / 'submission.csv', sep=';')
            aligned = template[KEY].merge(f[KEY+['prediction']], on=KEY, how='left', validate='one_to_one')
            assert aligned.prediction.notna().all()
            c[name] = aligned.prediction.to_numpy()
        c['v8_v13_equal'] = .5*c['v8']+.5*c['v13']
        prediction = c[chosen['anchor']]
        if chosen['hourly_weight'] > 0:
            network = selections[chosen['family']]
            ps = []
            for seed in seeds:
                f = read_frame(chosen['family'], '2025-11-01', network['config'], seed)
                aligned = template[KEY].merge(f[KEY+[network['path']]], on=KEY, how='left', validate='one_to_one')
                assert aligned[network['path']].notna().all()
                ps.append(aligned[network['path']].to_numpy())
            prediction = (1-chosen['hourly_weight'])*prediction + chosen['hourly_weight']*np.mean(ps, axis=0)
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        out = template[KEY].copy(); out['prediction'] = np.rint(prediction).astype(np.int64)
        assert len(out) == 14640 and out[KEY].equals(template[KEY]) and out.prediction.notna().all()
        destination = ROOT / 'outputs/submission_consistency_v14.csv'
        out.to_csv(destination, sep=';', index=False, lineterminator='\n')
        (DEST / 'submission.csv').write_bytes(destination.read_bytes())
        report['submission'] = {'path': 'outputs/submission_consistency_v14.csv',
                                'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                                'rows': len(out), 'total': int(out.prediction.sum()),
                                'combination': chosen, 'public_score': None, 'active_model_replaced': False}
        print('SUBMISSION', json.dumps(report['submission'], indent=2), flush=True)
    (DEST / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
