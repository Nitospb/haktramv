"""Fit the v11 strong-consistency model and export its weekly recursive path.

The requested configuration is fixed: lambda=5, final-week weight=2, 60 epochs.
The public-best submission is never overwritten. Run training on Studio/MPS.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unrolled_weekly import EXOG, KEY, audit_gradient, dataset, predict, train

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/v11_consistency_submission'
MODE = 'consistent7_c5_tail2'
EPOCHS = 60


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/submission_consistency_c5.csv')
    args = parser.parse_args()
    assert args.output.resolve() != (ROOT / 'outputs/submission_best.csv').resolve()
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    if device.type == 'mps':
        assert torch.backends.mps.is_available(), 'Studio MPS device is unavailable'

    d, y, ex, groups = dataset()
    origin = int((pd.Timestamp('2025-11-01') - pd.Timestamp('2025-01-01')).days)
    horizon = 61
    assert origin == 304 and np.isfinite(y[:origin]).all()
    assert np.isnan(y[origin:origin+horizon]).all(), 'Hidden labels unexpectedly present'
    frame = d[d.time.between(origin, origin+horizon-1)].reset_index(drop=True)
    assert len(frame) == 14640
    assert frame.date.min() == '2025-11-01' and frame.date.max() == '2025-12-31'

    model, normalization, audit = train(y, ex, origin, MODE, EPOCHS, device)
    paths = {name: predict(model, y, ex, origin, horizon, normalization, device, path)
             for name, path in [('recursive7', '7'), ('direct61', 'direct')]}
    paths['mean_paths'] = .5 * paths['recursive7'] + .5 * paths['direct61']
    for p in paths.values():
        assert p.shape == (14640,) and np.isfinite(p).all() and (p >= 0).all()

    changed = y.copy(); changed[origin:] = 999999
    corrupted_prediction = predict(model, changed, ex, origin, horizon, normalization, device, '7')
    perturbation = float(np.max(np.abs(corrupted_prediction - paths['recursive7'])))
    assert perturbation == 0
    gradient = audit_gradient(model, y, ex, origin, normalization, device, step=7)
    assert audit['max_training_label_day'] < origin

    # Keep exactly the requested model's weekly recursive forecast. Other paths
    # are retained for inspection, without selecting on unknown November labels.
    result = frame[KEY].copy()
    result['prediction'] = np.maximum(0, np.rint(paths['recursive7'])).astype(np.int64)
    template = pd.read_csv(ROOT / 'data/test_submission.csv', sep=';')
    result = template[KEY].merge(result, on=KEY, how='left', validate='one_to_one')
    assert result[KEY].equals(template[KEY]) and not result.duplicated(KEY).any()
    assert result.prediction.notna().all() and np.isfinite(result.prediction).all()
    assert (result.prediction >= 0).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep=';', index=False, lineterminator='\n')
    check = pd.read_csv(args.output, sep=';')
    pd.testing.assert_frame_equal(check, result)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    # Self-contained experiment copy in addition to the user-facing file.
    (DEST / 'submission.csv').write_bytes(args.output.read_bytes())
    np.savez_compressed(DEST / 'forecast_paths.npz', **paths)
    all_paths = frame[KEY].copy()
    for name, p in paths.items():
        all_paths[name] = p
    all_paths.to_csv(DEST / 'forecast_paths.csv.gz', index=False)

    discrepancy = float(np.abs(paths['recursive7'] - paths['direct61']).sum() /
                        max(1, paths['mean_paths'].sum()))
    audit.update({'mode': MODE, 'epochs': EPOCHS, 'device': str(device),
                  'torch_version': torch.__version__, 'forecast_origin': '2025-11-01',
                  'observed_history_through': '2025-10-31',
                  'last_supervised_training_date': str(pd.Timestamp('2025-01-01') +
                                                      pd.Timedelta(days=audit['max_training_label_day']))[:10],
                  'forecast_days': horizon, 'rows': len(result), 'path_exported': 'recursive7',
                  'consistency_weight': 5, 'last_week_loss_weight': 2,
                  'target_perturbation_max_abs_change': perturbation,
                  'last_to_first_block_gradient_norm': gradient,
                  'recursive_direct_disagreement_over_mean_prediction': discrepancy,
                  'disagreement_denominator': 'Total mean prediction, since hidden actual volume is unavailable',
                  'prediction_total': int(result.prediction.sum()),
                  'sha256': digest, 'new_public_score': None,
                  'known_best_public_score': .89214, 'active_model_replaced': False,
                  'requested_historical_configuration': {
                      'origin': '2025-09-01', 'recursive_score': .839425,
                      'recursive_direct_disagreement_over_actual_total': .007389507117145731},
                  'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'caveat': 'Experimental candidate requested by user despite weaker development scores. '
                            'Supplied retrospective weather/service features; external seasonality uses 2023-2024.'})
    torch.save({'state_dict': model.cpu().state_dict(), 'normalization': normalization,
                'features': EXOG, 'group_order': groups.to_dict('records'), 'training_audit': audit},
               DEST / 'model.pt')
    (DEST / 'report.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    print('SUBMISSION', args.output, flush=True)


if __name__ == '__main__':
    main()
