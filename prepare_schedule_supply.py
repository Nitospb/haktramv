"""Export archived minute schedules and derive average scheduled in-service trams.

Does not turn a current route-5 timetable into a verified 2025 timetable.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'data/features/06_schedules/export'
OUT = ROOT / 'data/schedule_research_20260927'
ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def minutes(column):
    parts = column.str.split(':', expand=True).astype(int)
    return parts[0] * 60 + parts[1] + parts[2] / 60


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    s = pd.read_csv(SRC / 'moscow_tram_full_schedule_by_stop.csv', dtype={'route_number': str, 'service_id': str})
    s['route'] = pd.to_numeric(s.route_number, errors='coerce')
    s = s[s.route.isin(ROUTES)].copy()
    s.route = s.route.astype(int)
    s = s.sort_values(['trip_id', 'stop_sequence'])
    s.to_csv(OUT / 'minute_schedule_archived_9_routes.csv.gz', index=False)
    s['departure_minute'] = minutes(s.departure_time)
    s['arrival_minute'] = minutes(s.arrival_time)
    t = s.groupby(['route', 'service_id', 'direction_id', 'trip_id'], as_index=False).agg(
        first_departure_minute=('departure_minute', 'first'),
        last_arrival_minute=('arrival_minute', 'last'), origin=('stop_name', 'first'),
        destination=('stop_name', 'last'))
    assert (t.last_arrival_minute >= t.first_departure_minute).all()
    t.to_csv(OUT / 'terminal_departures_archived.csv', index=False)
    keys = ['route', 'service_id', 'direction_id', 'service_hour']
    records = []
    for row in t.itertuples():
        a, b = row.first_departure_minute, row.last_arrival_minute
        for h in range(int(a // 60), int(b // 60) + 1):
            records.append((row.route, row.service_id, row.direction_id, h,
                            max(0., min(b, (h + 1) * 60) - max(a, h * 60)) / 60,
                            int(a // 60 == h)))
    q = pd.DataFrame(records, columns=keys + ['scheduled_mean_moving_trams', 'scheduled_terminal_departures'])
    q = q.groupby(keys, as_index=False).sum()
    with zipfile.ZipFile(SRC / 'moscow_tram_gtfs_filtered.zip') as z:
        cal = pd.read_csv(z.open('calendar.txt'), dtype={'service_id': str})
        has_exceptions = 'calendar_dates.txt' in z.namelist()
        assert not has_exceptions, 'Add explicit calendar exceptions support before using another feed'
    cal = cal[cal.service_id.isin(t.service_id)].drop(columns=['Unnamed: 11'], errors='ignore')
    cal.to_csv(OUT / 'archived_service_calendar.csv', index=False)
    chunks = []
    max_lag = int(q.service_hour.max() // 24)
    for day in pd.date_range(pd.Timestamp('2025-01-01') - pd.Timedelta(days=max_lag), '2025-12-31'):
        stamp = int(day.strftime('%Y%m%d'))
        services = cal.loc[(cal.start_date <= stamp) & (cal.end_date >= stamp) & (cal[WEEK[day.dayofweek]] == 1), 'service_id']
        x = q[q.service_id.isin(services)].copy()
        x['date'] = (day + pd.to_timedelta(x.service_hour // 24, unit='D')).dt.strftime('%Y-%m-%d')
        x['hour'] = x.service_hour % 24
        chunks.append(x)
    x = pd.concat(chunks, ignore_index=True)
    numeric = ['scheduled_mean_moving_trams', 'scheduled_terminal_departures']
    x = x.groupby(['route', 'date', 'hour'], as_index=False)[numeric].sum()
    grid = pd.MultiIndex.from_product([ROUTES, pd.date_range('2025-01-01', '2025-12-31').strftime('%Y-%m-%d'), range(24)], names=['route', 'date', 'hour']).to_frame(index=False)
    grid = grid.merge(x, how='left', on=['route', 'date', 'hour'], validate='one_to_one')
    # Coverage is day-level: zero night service differs from missing schedule.
    covered = x[['route', 'date']].drop_duplicates().assign(schedule_day_covered=1)
    grid = grid.merge(covered, how='left', on=['route', 'date'], validate='many_to_one')
    grid['schedule_day_covered'] = grid.schedule_day_covered.fillna(0).astype(int)
    for c in numeric:
        grid.loc[grid.schedule_day_covered.eq(1), c] = grid.loc[grid.schedule_day_covered.eq(1), c].fillna(0)
    grid = grid.sort_values(['date', 'route', 'hour'])
    grid.to_csv(OUT / 'archived_supply_hourly_2025.csv', index=False)
    # Conservation check: hourly vehicle-minutes must equal trip duration.
    np.testing.assert_allclose(q.scheduled_mean_moving_trams.sum() * 60,
                               (t.last_arrival_minute - t.first_departure_minute).sum(), atol=1e-6)
    route5 = []
    for item in json.loads((OUT / 'web_sources.json').read_text()):
        path = OUT / item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        direction, day = re.search(r'route5_([AB])_day(\d)', path.name).groups()
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        blocks = soup.select('.bus-stop')
        assert len(blocks) == 16, (path, len(blocks))
        for b in blocks:
            label = b.get_text(' ', strip=True)
            seq, name = re.match(r'(\d+)\)\s*(.*)', label).groups()
            ts = [v.get_text(strip=True) for v in b.parent.select('.stop-times span:not(.show-all)')]
            assert len(ts) > 80 and all(re.fullmatch(r'\d{2}:\d{2}', v) for v in ts)
            assert len(ts) == len(set(ts)), (path, seq)
            for tm in ts:
                h, m = map(int, tm.split(':'))
                assert 0 <= h < 24 and 0 <= m < 60
                route5.append(dict(route=5, direction=direction, weekday=(int(day) - 1) % 7,
                    stop_sequence=int(seq), stop_name=name, clock_time=tm,
                    minute_of_day=h * 60 + m, source_url=item['url'],
                    historical_2025_verified=0))
    five = pd.DataFrame(route5).sort_values(['weekday', 'direction', 'stop_sequence', 'minute_of_day'])
    five.to_csv(OUT / 'route5_current_minute_schedule.csv', index=False)
    first = five[five.stop_sequence == 1].copy()
    first['hour'] = first.minute_of_day // 60
    first.groupby(['route', 'weekday', 'direction', 'hour']).size().rename('terminal_departures').to_csv(OUT / 'route5_current_hourly_departures.csv')
    coverage = s.groupby('route').agg(stop_time_rows=('trip_id', 'size'), trip_templates=('trip_id', 'nunique'), stops=('stop_id', 'nunique')).reset_index()
    coverage.to_csv(OUT / 'archived_coverage.csv', index=False)
    report = dict(archived_routes=sorted(s.route.unique().tolist()), route5_rows=len(five),
        archived_stop_time_rows=len(s), archived_trip_templates=len(t),
        calendar_exceptions_present=has_exceptions, hourly_rows=len(grid),
        supply_definition='Sum of trip time spent in hour / 60: average scheduled concurrent in-service trips; excludes terminal layover, not observed fleet.',
        validity='Archived community GTFS retrieved June 2026, internal source date May 2025. Not verified as actual service for all dates. calendar_dates absent; holidays and later changes not fully represented.',
        route5_validity='Current aggregator schedule for all seven weekdays and both directions. Effective date absent; do not apply as verified December 2025 schedule.',
        route5_launch_date='2025-12-16', route5_launch_source='https://transport.mos.ru/mostrans/all_news/127652',
        forecast_asof='Route-5 launch source is after Oct 31; event is retrospective, not confirmed known at forecast origin.',
        checks=['All 9 archived target routes covered', 'All 14 route-5 weekday/direction pages parsed, 16 stops each',
                'Vehicle-minute conservation', 'Unique route-date-hour keys', 'Missing day coverage kept separate from zero service'])
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
