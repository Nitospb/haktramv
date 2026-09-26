"""Compare v12 paths and fit a small ensemble using earlier origins only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from train_model import score

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v12_multistream'
if not DEST.exists():
    DEST = ROOT / 'outputs/studio/v12_multistream'
FOLDS = ['2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01']
SOURCES = ['incumbent_v8', 's1_f1_w28_remaining', 's3_f3_w28_remaining',
           's7_f7_w28_remaining', 's14_f14_w28_remaining', 's28_f28_w28_remaining',
           'streams_balanced', 'streams_median', 'synchronized_mean_anchor0',
           'synchronized_median_anchor0', 'streams_windows_median', 'streams_blocks_mean']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['full', 'cached_shape'], default='full')
    args = parser.parse_args()
    global DEST
    if not (DEST / (args.engine + '_' + FOLDS[0] + '.csv.gz')).exists():
        DEST = ROOT / 'outputs/studio/v12_multistream'
    frames = {fold: pd.read_csv(DEST / (args.engine + '_' + fold + '.csv.gz')) for fold in FOLDS}
    # Normalize each complete 61-day window separately, so each has equal weight.
    x = np.concatenate([frames[f][SOURCES].to_numpy() / frames[f].boardings.sum() for f in FOLDS[:2]])
    y = np.concatenate([frames[f].boardings.to_numpy() / frames[f].boardings.sum() for f in FOLDS[:2]])
    initial = np.zeros(len(SOURCES)); initial[0] = 1
    def objective(w):
        return np.abs(x @ w - y).sum() / 2 + .001 * np.square(w - initial).sum()
    result = minimize(objective, initial, method='SLSQP', bounds=[(0, 1)] * len(initial),
                      constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                      options={'maxiter': 500, 'ftol': 1e-11})
    weights = result.x if result.success else initial
    weights = np.maximum(0, weights); weights /= weights.sum()
    rows = []; agreement = []; by_route = []; by_week = []
    model_names = [c for c in frames[FOLDS[0]] if c not in ['route', 'date', 'hour', 'boardings']]
    early = {name: np.mean([score(frames[f].boardings, frames[f][name]) for f in FOLDS[:2]])
             for name in model_names}
    selected = max(early, key=early.get)
    for fold, d in frames.items():
        d['trained_stream_ensemble'] = d[SOURCES].to_numpy() @ weights
        model_cols = [c for c in d if c not in ['route', 'date', 'hour', 'boardings']]
        days = (pd.to_datetime(d.date) - pd.Timestamp(fold)).dt.days.to_numpy()
        for name in model_cols:
            for period, mask in [('all', days >= 0), ('last_7_days', days >= days.max() - 6)]:
                rows.append(dict(origin=fold, model=name, part=period,
                                 score=score(d.boardings.to_numpy()[mask], d[name].to_numpy()[mask])))
        paths = ['s%d_f%d_w28_remaining' % (step, step) for step in [1, 3, 7, 14, 28]]
        stack = d[paths].to_numpy()
        errors = stack - d.boardings.to_numpy()[:, None]
        corr = np.corrcoef(errors.T)
        off_diagonal = corr[np.triu_indices(len(paths), 1)]
        spread = np.percentile(stack, 90, axis=1) - np.percentile(stack, 10, axis=1)
        agreement.append(dict(origin=fold, mean_error_correlation=float(off_diagonal.mean()),
                              p10_p90_spread_over_actual_total=float(spread.sum() / d.boardings.sum()),
                              mean_pointwise_path_std=float(np.std(stack, axis=1).mean())))
        for route, group in d.groupby('route'):
            if group.boardings.sum() > 0:
                for name in ['incumbent_v8', 'streams_balanced', 'synchronized_mean_anchor0',
                             'trained_stream_ensemble', selected]:
                    by_route.append(dict(origin=fold, route=int(route), model=name,
                                         score=score(group.boardings, group[name])))
        d['forecast_week'] = days // 7 + 1
        for week, group in d.groupby('forecast_week'):
            for name in ['incumbent_v8', 'streams_balanced', 'synchronized_mean_anchor0',
                         'trained_stream_ensemble', selected]:
                by_week.append(dict(origin=fold, week=int(week), model=name,
                                    actual_total=float(group.boardings.sum()),
                                    predicted_total=float(group[name].sum()),
                                    score=score(group.boardings, group[name])))
        d[['route', 'date', 'hour', 'boardings', 'trained_stream_ensemble']].to_csv(
            DEST / (args.engine + '_trained_ensemble_' + fold + '.csv.gz'), index=False)
    r = pd.DataFrame(rows)
    r.to_csv(DEST / (args.engine + '_comparison.csv'), index=False)
    pd.DataFrame(by_route).to_csv(DEST / (args.engine + '_by_route.csv'), index=False)
    pd.DataFrame(by_week).to_csv(DEST / (args.engine + '_by_week.csv'), index=False)
    table = r[r.part == 'all'].pivot(index='model', columns='origin', values='score')
    chosen = list(dict.fromkeys(['incumbent_v8', 'direct_engine', *SOURCES[1:6],
                                'streams_balanced', 'streams_median',
                                'synchronized_mean_anchor0', 'synchronized_median_anchor0',
                                'synchronized_mean_anchor50', 'trained_stream_ensemble', selected]))
    print(table.loc[chosen].round(6).to_string(), flush=True)
    report = {'engine': args.engine, 'fit_origins': FOLDS[:2], 'diagnostic_origins': FOLDS[2:],
              'weights': dict(zip(SOURCES, weights.tolist())), 'fit_converged': bool(result.success),
              'selected_by_earlier_origins': selected,
              'scores': table.loc[chosen].reset_index().to_dict('records'),
              'stream_error_agreement': agreement,
              'new_public_score': None,
              'caveat': 'All periods are previously explored development diagnostics. July-August '
                        'is cropped to 61 days. October is 31 days. Earlier windows overlap, and '
                        'the inherited incumbent blend was previously tuned with September-October. '
                        'No future observations are supplied during a rollout.'}
    (DEST / (args.engine + '_report.json')).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
