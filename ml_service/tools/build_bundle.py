"""Copy only approved, non-transaction artifacts into a self-contained image context."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'ml_service/artifacts'
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]


def main():
    DEST.mkdir(exist_ok=True)
    copies = {'base.csv': 'outputs/studio/v2/submission.csv',
              'expert.csv': 'outputs/studio/v5_experts/submission.csv',
              'expected.csv': 'outputs/submission_best.csv'}
    for name, source in copies.items():
        shutil.copyfile(ROOT/source, DEST/name)
    with (ROOT/'data/features/02_repairs_and_service/route_hour_features_full_calendar.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if '2025-11-01' <= r['date'] <= '2025-12-31']
    with (DEST/'cancellations.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['route','date','hour','service_cancelled'], extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    geo = ROOT/'data/features/05_stops_and_nearby_objects'
    stops = [s for s in json.loads((geo/'stops.json').read_text()) if any(str(r) in s.get('routes',[]) for r in ROUTES)]
    (DEST/'stops.json').write_text(json.dumps(stops,ensure_ascii=False))
    routes = json.loads((geo/'routes.geojson').read_text())
    routes['features'] = [f for f in routes['features'] if any(str(r) in f['properties'].get('routes',[]) for r in ROUTES)]
    (DEST/'routes.geojson').write_text(json.dumps(routes,ensure_ascii=False))
    names = json.loads((geo/'routes_info.json').read_text())
    (DEST/'route_names.json').write_text(json.dumps({str(r): names.get(str(r),str(r)) for r in ROUTES},ensure_ascii=False))
    assert hashlib.sha256((DEST/'expected.csv').read_bytes()).hexdigest() == 'f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    manifest = dict(model_id='v8/level_scale_1.03',public_score=.89402,
        public_score_source='User-reported competition result', start='2025-11-01', end='2025-12-31',
        timezone='Europe/Moscow', routes=ROUTES, rows=14640, target='hourly_boardings',
        mode='frozen_competition_forecast', trained_through='2025-10-31',
        inference='Reconstruct v8 from saved base/expert predictions, monthly reconciliation, cancellations, then rounded +3%. No training or external network at request time.',
        limitations=['No validated forecasts outside Nov-Dec 2025.', 'No stop-level or segment-level target model.',
                     'Route 5 has no positive training target; preserved best predicts zero.',
                     'Retrospective external data; this is not an as-of live forecasting service.',
                     'Target is boardings, not onboard occupancy or vehicle capacity.'],
        upstream_code=['research_v2.py','feature_models.py','service_models.py','event_experts.py','reconcile_forecasts.py'],
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir()) if p.name!='manifest.json' and p.is_file()})
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Bundled',len(manifest['files']),'files; no raw transactions.')


if __name__ == '__main__':
    main()
