"""Five user-authorized leaderboard probes from saved forecasts, without fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/reset_probes_20260926'
KEY = ['route', 'date', 'hour']
DESCRIPTIONS = {
    '01_level_down3': 'V8 total level multiplied by 0.97.',
    '02_level_up3': 'V8 total level multiplied by 1.03.',
    '03_month_tilt': 'First month +3%, offset by second month reduction; exact two-month total per route preserved.',
    '04_weekends_up6': 'Non-working days +6%, offset by working days; exact monthly total per route preserved.',
    '05_weekly_shape': 'Pure v16 weekly-network shape, rescaled to exact V8 monthly totals per route.',
}


def integer_reconcile(frame, prediction, keys):
    """Keep V8 group totals exactly, with deterministic largest-remainder rounding."""
    output = np.zeros(len(frame), dtype=np.int64)
    for _, ix in frame.groupby(keys, sort=False).indices.items():
        target = int(frame.prediction.to_numpy()[ix].sum())
        values = np.maximum(prediction[ix], 0.)
        if target == 0:
            continue
        if values.sum() <= 0:
            values = frame.prediction.to_numpy()[ix].astype(float)
        values = values * (target / values.sum())
        rounded = np.floor(values).astype(np.int64)
        remainder = target - rounded.sum()
        assert 0 <= remainder <= len(ix)
        order = np.argsort(-(values - rounded), kind='stable')
        rounded[order[:remainder]] += 1
        output[ix] = rounded
        assert int(output[ix].sum()) == target
    return output


def variants(frame, weekly):
    base = frame.prediction.to_numpy(dtype=float)
    ps = {'01_level_down3': np.rint(base * .97).astype(np.int64),
          '02_level_up3': np.rint(base * 1.03).astype(np.int64)}
    p = base.copy()
    first_month = frame.month.min()
    for _, ix in frame.groupby('route', sort=False).indices.items():
        first = ix[frame.month.to_numpy()[ix] == first_month]
        second = ix[frame.month.to_numpy()[ix] != first_month]
        if len(second) and base[second].sum() > 0:
            moved = .03 * base[first].sum()
            p[first] *= 1.03
            p[second] *= max(0, 1 - moved / base[second].sum())
    ps['03_month_tilt'] = integer_reconcile(frame, p, ['route'])
    p = base.copy()
    for _, ix in frame.groupby(['route', 'month'], sort=False).indices.items():
        weekend = ix[frame.workday.to_numpy()[ix] == 0]
        working = ix[frame.workday.to_numpy()[ix] == 1]
        if len(working) and base[working].sum() > 0:
            moved = .06 * base[weekend].sum()
            p[weekend] *= 1.06
            p[working] *= max(0, 1 - moved / base[working].sum())
    ps['04_weekends_up6'] = integer_reconcile(frame, p, ['route', 'month'])
    ps['05_weekly_shape'] = integer_reconcile(frame, np.asarray(weekly, dtype=float), ['route', 'month'])
    return ps


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(ROOT / 'outputs/studio/v4/training_matrix.csv', usecols=KEY + ['workday'])
    # Freeze the original experiment anchor even after a probe is promoted.
    base_path = ROOT / 'outputs/studio/v8/submission.csv'
    base_bytes = base_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    assert base_hash == 'b8f8238db449a7401f9d63543c60288b018ed9a1e991ba04745825f82d056524'
    template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
    base = pd.read_csv(base_path, sep=';')
    frame = base.merge(cal, on=KEY, how='left', validate='one_to_one')
    frame['month'] = frame.date.str[:7]
    assert frame[KEY].equals(template[KEY]) and frame.workday.notna().all()
    weekly = pd.read_csv(ROOT / 'outputs/submission_weekly_recursive_v16.csv', sep=';')
    weekly = frame[KEY].merge(weekly, on=KEY, how='left', validate='one_to_one').prediction
    assert weekly.notna().all()
    files = []
    for name, prediction in variants(frame, weekly.to_numpy()).items():
        result = frame[KEY].copy()
        result['prediction'] = prediction
        assert len(result) == 14640 and result[KEY].equals(template[KEY])
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        assert (result.loc[result.route == 5, 'prediction'] == 0).all()
        path = DEST / ('submission_' + name + '.csv')
        result.to_csv(path, sep=';', index=False, lineterminator='\n')
        files.append({'name': name, 'hypothesis': DESCRIPTIONS[name],
                      'submission': str(path.relative_to(ROOT)),
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'prediction_total': int(prediction.sum()),
                      'changed_rows': int((prediction != frame.prediction.to_numpy()).sum()),
                      'public_score': None})
    assert len({f['sha256'] for f in files}) == 5
    metrics = []
    for origin in ['2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01']:
        old = pd.read_csv(ROOT / 'outputs/studio/v8' / ('validation_' + origin + '.csv'), sep=';')
        old = old[pd.to_datetime(old.date) < pd.Timestamp(origin) + pd.Timedelta(days=61)].reset_index(drop=True)
        old['prediction'] = np.rint(old.prediction).astype(np.int64)
        old = old.merge(cal, on=KEY, how='left', validate='one_to_one')
        old['month'] = old.date.str[:7]
        other = pd.read_csv(ROOT / 'outputs/studio/v16_weekly' / (origin + '_weekly_stable_s20260926_e80.csv.gz'))
        aligned = old[KEY].merge(other[KEY + ['step7']], on=KEY, how='left', validate='one_to_one').step7
        assert aligned.notna().all()
        predictions = {'v8_rounded': old.prediction.to_numpy(), **variants(old, aligned.to_numpy())}
        actual = old.boardings.to_numpy()
        for name, p in predictions.items():
            metrics.append({'origin': origin, 'model': name,
                            'score': float(max(0, 1 - np.abs(actual-p).sum()/actual.sum())),
                            'days': old.date.nunique(),
                            'applicable': name != '03_month_tilt' or old.month.nunique() == 2})
    pd.DataFrame(metrics).to_csv(DEST / 'development_metrics.csv', index=False)
    report = {'base_sha256': base_hash, 'base_public_score': .89214, 'rows_per_file': 14640,
              'hypotheses': files, 'development_metrics': metrics,
              'purpose': 'User requested five free submission probes before reset; no training, no hidden labels.',
              'caveat': 'Diagnostic probes, not validated improvements. Existing development windows overlap and were previously explored. Month-tilt is inapplicable to October-only validation.',
              'best_submission_unchanged': base_path.read_bytes() == base_bytes}
    (DEST / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['best_submission_unchanged']
    print(json.dumps(files, indent=2))
    print(pd.DataFrame(metrics).pivot(index='model', columns='origin', values='score').round(6).to_string())


if __name__ == '__main__':
    main()
