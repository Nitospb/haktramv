"""Reproduce the final three CSVs without fitting or loading optimizer states."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np
import lightgbm as lgb
from night_forecast_common import ROOT, OUT, KEY
from night_calendar_analog_study import CalendarData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/night_20260927/reproduction')
    args = parser.parse_args()
    dest = (ROOT/args.output).resolve(); dest.relative_to(ROOT)
    dest.mkdir(parents=True, exist_ok=True)
    source = OUT/'deliverables'
    manifest = json.loads((source/'verification.json').read_text())
    model_path = source/'direct_1600.txt'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == manifest['compact_model_sha256']
    data = CalendarData()
    model = lgb.Booster(model_file=str(model_path))
    future = data.frame(304)
    predictions = [
        ('submission_01_direct.csv', model.predict(future[model.feature_name()], num_threads=6)),
        ('submission_02_calendar.csv', data.analog(304, 61)['term']),
    ]
    for name, p in predictions:
        frame, p = data.prediction(304, p)
        z = frame[KEY].copy(); z['route'] = z.route.astype(int); z['hour'] = z.hour.astype(int)
        z['prediction'] = np.rint(p).astype(int)
        z.to_csv(dest/name, sep=';', index=False)
    subprocess.run([sys.executable, str(ROOT/'attach_route5_to_external.py'), '--base',
                    str((dest/'submission_01_direct.csv').relative_to(ROOT))], check=True)
    shutil.copyfile(dest/'submission_moscow_with_route5.csv', dest/'submission_03_direct_route5.csv')
    for item in manifest['submissions']:
        actual = hashlib.sha256((dest/item['file']).read_bytes()).hexdigest()
        assert actual == item['sha256'], (item['file'], actual, item['sha256'])
    print('All three CSV files reproduced byte-for-byte; no training performed.')


if __name__ == '__main__':
    main()
