"""Reproduce the public-best v8 submission from saved forecasts, without training."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEY = ['route', 'date', 'hour']


def reproduce():
    outputs = ROOT / 'outputs'
    artifacts = outputs / 'studio' if (outputs / 'studio/v2/submission.csv').exists() else outputs
    active = json.loads((outputs / 'active_model.json').read_text())
    assert active['active_model'] in ['v8/monthly_reconciliation', 'v8/level_scale_1.03']
    base = pd.read_csv(artifacts / 'v2/submission.csv', sep=';')
    expert = pd.read_csv(artifacts / 'v5_experts/submission.csv', sep=';')
    template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
    assert base[KEY].equals(expert[KEY]) and base[KEY].equals(template[KEY])
    service = pd.read_csv(ROOT / 'data/features/02_repairs_and_service/route_hour_features_full_calendar.csv',
                          usecols=KEY + ['service_cancelled'])
    x = base[KEY].merge(service, on=KEY, how='left', validate='one_to_one')
    assert x.service_cancelled.notna().all()
    x['month'] = pd.to_datetime(x.date).dt.month
    x['base'] = base.prediction.to_numpy(); x['expert'] = expert.prediction.to_numpy()
    total = x.groupby(['route', 'month'])[['base', 'expert']].transform('sum')
    adjusted = x.expert.to_numpy() * (total.base / (total.expert + 1e-9)).to_numpy()
    empty = total.expert.to_numpy() < 1e-6
    adjusted[empty] = x.base.to_numpy()[empty]
    prediction = (.25 * x.base.to_numpy() + .75 * adjusted) * (1 - .975 * x.service_cancelled.to_numpy())
    result = base[KEY].copy()
    result['prediction'] = np.maximum(0, np.rint(prediction)).astype(int)
    if active['active_model'] == 'v8/level_scale_1.03':
        result['prediction'] = np.rint(result.prediction.to_numpy() * 1.03).astype(int)
    assert len(result) == 14640 and not result.duplicated(KEY).any()
    assert np.isfinite(result.prediction).all() and (result.prediction >= 0).all()
    buffer = io.StringIO(); result.to_csv(buffer, sep=';', index=False, lineterminator='\n')
    payload = buffer.getvalue().encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == active['sha256'], 'Reproduced output differs from the recorded public-best artifact'
    assert payload == (outputs / active['submission']).read_bytes(), 'Active submission has changed'
    return payload, digest, active['user_reported_score']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional path for the reproduced CSV')
    args = parser.parse_args()
    payload, digest, public_score = reproduce()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(json.dumps({'rows': 14640, 'sha256': digest, 'matches_public_best': True,
                      'user_reported_public_score': public_score,
                      'output': str(args.output) if args.output else None}, indent=2))


if __name__ == '__main__':
    main()
