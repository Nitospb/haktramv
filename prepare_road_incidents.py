"""Nearby reported injury crashes, not congestion measurements or closure duration."""
import hashlib
import json
import zipfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/complete_external_20260927'
URL = 'https://dtp-stat.ru/media/opendata/moskva.geojson.zip'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Raw participant-level records are not retained in the repository.
    cache = Path('/tmp/haktramv_moscow_dtp.zip')
    if not cache.exists():
        r = requests.get(URL, timeout=180); r.raise_for_status(); cache.write_bytes(r.content)
    with zipfile.ZipFile(cache) as z:
        data = json.loads(z.read('moskva.geojson'))
    records = []
    for f in data['features']:
        p = f['properties']
        if str(p['datetime']).startswith('2025-'):
            lon, lat = f['geometry']['coordinates']
            records.append((p['id'], p['datetime'], lon, lat))
    events = pd.DataFrame(records, columns=['id','timestamp','longitude','latitude']).drop_duplicates('id')
    events['timestamp'] = pd.to_datetime(events.timestamp).dt.floor('h')
    assert len(events) > 0 and events.timestamp.dt.month.nunique() == 12
    stops = pd.read_csv(ROOT / 'data/features/03_weather_and_calendar/districts/stop_route_weather_cells.csv')
    hours = pd.date_range('2025-01-01', '2025-12-31 23:00:00', freq='h')
    parts = []; matches = {}
    for route, s in stops.groupby('route'):
        # Great-circle distance to the nearest listed stop, not route polyline/district.
        lat1 = np.deg2rad(events.latitude.to_numpy())[:, None]
        lat2 = np.deg2rad(s.latitude.to_numpy())[None, :]
        dlon = np.deg2rad(events.longitude.to_numpy()[:, None] - s.longitude.to_numpy()[None, :])
        a = np.sin((lat1-lat2)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        distance = (6371000 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min(axis=1)
        d = pd.DataFrame(index=hours)
        matches[str(route)] = {}
        for radius in [300, 700, 1500]:
            nearby = events[distance <= radius]
            count = nearby.groupby('timestamp').size().reindex(hours, fill_value=0)
            d['road_crash_' + str(radius) + 'm_hour'] = count
            d['road_crash_' + str(radius) + 'm_past3h'] = count.rolling(3, min_periods=1).sum()
            matches[str(route)][str(radius)] = len(nearby)
        d['route'] = route; d['date'] = d.index.strftime('%Y-%m-%d'); d['hour'] = d.index.hour
        parts.append(d.reset_index(drop=True))
    frame = pd.concat(parts, ignore_index=True)
    assert len(frame) == 87600 and not frame.duplicated(['route','date','hour']).any()
    frame.to_parquet(OUT / 'route_hour_road_incidents_2025.parquet', index=False)
    report = dict(source=URL, documentation='https://dtp-stat.ru/opendata/', retrieved_date='2026-09-27',
                  archive_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(), events_2025=len(events),
                  events_by_month=events.groupby(events.timestamp.dt.strftime('%Y-%m')).size().to_dict(),
                  matched_events_by_route_radius=matches, rows=len(frame),
                  interpretation='Reported crashes near stops, NOT traffic speed/intensity. Zero means no matching record, not proof of clear roads.',
                  caveats=['Offset-free Moscow source timestamps interpreted as Moscow local time.',
                           'Primarily police-reported crashes with injuries; minor crashes may be absent.',
                           'Coordinates corrected by dtp-stat; no measured duration or demonstrated tram delay.',
                           'Past 3h window is a feature hypothesis, not an observed road closure.',
                           'Stop geometries are supplied snapshots, not historically versioned.'],
                  search_results=[
                      dict(url='https://gucodd.mos.ru/', status='Current city congestion; no usable district-hour 2025 export found.'),
                      dict(url='https://www.tomtom.com/downloads/traffic-index/', status='Moscow absent from listed free city CSV downloads.'),
                      dict(url='https://www.yandex.ru/support/maps/ru/concept/stoppers', status='Current traffic display, no open 2025 street-hour export located.'),
                      dict(url='https://sourceforge.net/projects/ymarchive/files/', status='Archive latest 2015; not 2025 observations.')])
    (OUT / 'road_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
