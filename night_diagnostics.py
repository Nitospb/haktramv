"""Recompute fixed-origin errors from saved forecasts; no model training."""
import re
import numpy as np
import pandas as pd
from night_forecast_common import Data, OUT, ROOT, KEY, SELECTION, CONTROL, save_json


def main():
    data = Data()
    totals, slices = [], []

    def evaluate(frame, column, name, origin):
        z = frame[KEY + ['boardings']].copy()
        assert len(z) == 14640 and not z.duplicated(KEY).any()
        truth = data.frame(origin)[KEY + ['boardings']]
        check = z.merge(truth, on=KEY, validate='one_to_one', suffixes=('', '_reference'))
        assert len(check) == 14640
        np.testing.assert_array_equal(check.boardings, check.boardings_reference)
        z['prediction'] = np.maximum(0, np.rint(frame[column].to_numpy()))
        assert np.isfinite(z.prediction).all() and z.boardings.notna().all()
        z['error'] = (z.prediction-z.boardings).abs()
        dates = pd.to_datetime(z.date)
        z['weekday'] = dates.dt.dayofweek
        z['week'] = ((dates-dates.min()).dt.days//7+1)

        def metrics(q):
            y = float(q.boardings.sum())
            e = float(q.error.sum())
            return dict(score=max(0, 1-e/max(1, y)), target=y,
                        prediction=float(q.prediction.sum()), absolute_error=e,
                        bias=(float(q.prediction.sum())-y)/max(1, y))

        rec = dict(model=name, origin=origin, **metrics(z))
        daily = z.groupby(['route', 'date'])[['boardings', 'prediction']].sum()
        rec['daily_total_score'] = 1-float((daily.prediction-daily.boardings).abs().sum())/max(1, float(z.boardings.sum()))
        totals.append(rec)
        for field in ['route', 'weekday', 'hour', 'week']:
            for value, q in z.groupby(field):
                slices.append(dict(model=name, origin=origin, dimension=field,
                                   value=int(value), **metrics(q)))

    for origin in SELECTION + [CONTROL]:
        for half in [14, 28, 56, 112]:
            base = data.profile(origin, half)[data.daytype[origin:origin+61]].reshape(-1)
            frame, p = data.prediction(origin, base)
            frame['prediction'] = p
            evaluate(frame, 'prediction', 'weekday_profile_'+str(half), origin)
    for family in ['catboost', 'sequence', 'sequence_large', 'calendar_analog']:
        for path in sorted((OUT/family).glob('*.csv')):
            match = re.search(r'_(151|181|212|243)\.csv$', path.name)
            if not match:
                continue
            frame = pd.read_csv(path)
            if 'prediction' in frame:
                evaluate(frame, 'prediction', family+'/'+path.stem.rsplit('_', 1)[0], int(match[1]))
    for origin in [181, 212, CONTROL]:
        for directory, columns in [
            ('all_moscow_external_20260927', ['confirmed_v8_up3']),
            ('longer_moscow_external_20260927', ['direct_800', 'direct_1600', 'direct_3200']),
        ]:
            path = ROOT/'outputs'/directory/('forecast_'+str(origin)+'.csv')
            if path.exists():
                frame = pd.read_csv(path)
                for column in columns:
                    evaluate(frame, column, directory+'/'+column, origin)
    folder = OUT/'diagnostics'
    folder.mkdir(exist_ok=True)
    pd.DataFrame(totals).to_csv(folder/'scores.csv', index=False)
    pd.DataFrame(slices).to_csv(folder/'error_slices.csv', index=False)
    save_json(folder/'verification.json', dict(forecasts_checked=len(totals),
        metric='1 - sum(abs(target-round(prediction)))/sum(target), clipped at 0',
        rows_per_forecast=14640, targets_verified=True,
        warning='Snapshots may omit checkpoints from jobs that are still running. Development windows overlap.'))
    print(pd.DataFrame(totals).pivot(index='model', columns='origin', values='score').to_string())


if __name__ == '__main__':
    main()
