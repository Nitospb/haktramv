"""Retrospective route-5 scenario anchored to an official first-week aggregate.

User states organizers permit retrospective/leakage inputs. The news aggregate
is rounded and need not equal competition validations. No accuracy is claimed.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'outputs/route5_public_anchor_20260927'
KEY = ['route', 'date', 'hour']


def integer_allocation(values, total):
    scaled = np.asarray(values) / np.sum(values) * total
    result = np.floor(scaled).astype(int)
    result[np.argsort(-(scaled - result), kind='stable')[:int(total - result.sum())]] += 1
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    base_path = ROOT / 'outputs/submission_best.csv'
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == 'f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
    base = pd.read_csv(base_path, sep=';')
    assert base.loc[base.route.eq(5), 'prediction'].eq(0).all()
    five = pd.read_csv(ROOT / 'data/schedule_research_20260927/route5_current_minute_schedule.csv')
    five = five[five.stop_sequence < 16].copy()  # final stop is for alighting
    five['hour'] = five.minute_of_day // 60
    opportunities = five.groupby(['weekday', 'hour']).size().rename('route5_stop_opportunities').reset_index()
    supply = pd.read_csv(ROOT / 'data/features/06_schedules/gtfs_hourly_features.csv')
    donor = base[base.route.isin([7, 50])].merge(supply[KEY + ['scheduled_stop_events']], on=KEY, how='left', validate='one_to_one')
    donor['demand_per_opportunity'] = donor.prediction / donor.scheduled_stop_events.replace(0, np.nan)
    intensity = donor.groupby(['date', 'hour']).demand_per_opportunity.median().rename('donor_demand_intensity').reset_index()
    grid = base[base.route.eq(5)][KEY].copy()
    grid['weekday'] = pd.to_datetime(grid.date).dt.dayofweek
    grid = grid.merge(opportunities, on=['weekday', 'hour'], how='left', validate='many_to_one')
    grid = grid.merge(intensity, on=['date', 'hour'], how='left', validate='one_to_one')
    # Preserve small night service where donor routes are already off duty.
    # Fallback is a same-day daytime median; it is a scenario assumption.
    daytime_median = grid.groupby('date').donor_demand_intensity.transform('median')
    grid['donor_demand_intensity'] = grid.donor_demand_intensity.fillna(daytime_median).fillna(0)
    grid['route5_stop_opportunities'] = grid.route5_stop_opportunities.fillna(0)
    grid['unscaled_shape'] = grid.route5_stop_opportunities * grid.donor_demand_intensity
    grid.loc[grid.date < '2025-12-16', 'unscaled_shape'] = 0
    week = grid.date.between('2025-12-16', '2025-12-22')
    multiplier = 40000 / grid.loc[week, 'unscaled_shape'].sum()
    grid['prediction'] = np.rint(grid.unscaled_shape * multiplier).astype(int)
    grid.loc[week, 'prediction'] = integer_allocation(grid.loc[week, 'unscaled_shape'], 40000)
    out = base.copy()
    lookup = grid.set_index(KEY).prediction
    idx = base.route.eq(5)
    out.loc[idx, 'prediction'] = lookup.reindex(pd.MultiIndex.from_frame(base.loc[idx, KEY])).to_numpy()
    assert len(out) == 14640 and not out.duplicated(KEY).any() and out.prediction.ge(0).all()
    pd.testing.assert_frame_equal(base.loc[~idx], out.loc[~idx])
    assert out.loc[idx & (out.date < '2025-12-16'), 'prediction'].eq(0).all()
    assert grid.loc[week, 'prediction'].sum() == 40000
    out.to_csv(DEST / 'submission_best_plus_route5.csv', sep=';', index=False)
    grid.to_csv(DEST / 'route5_hourly_scenario.csv', index=False)
    grid.groupby('date').prediction.sum().to_csv(DEST / 'route5_daily_scenario.csv')
    report = dict(public_score=None, active_model_changed=False,
        source_url='https://transport.mos.ru/mostrans/all_news/127782', source_publication_date='2025-12-23',
        official_reported_first_week_passenger_trips=40000, official_reported_first_week_vehicle_trips=1500,
        assumed_anchor_dates=['2025-12-16', '2025-12-22'], projected_route5_december_total=int(grid.prediction.sum()),
        base_public_score=.89402, base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256((DEST / 'submission_best_plus_route5.csv').read_bytes()).hexdigest(),
        user_authorization='User explicitly reports organizers permit leakage and retrospective data.',
        method='Only route 5 changes. Current route-5 stop opportunities multiplied by donor demand per opportunity from existing best route 7/50 forecasts; first-week sum anchored to 40000.',
        limitations=['Official aggregate is rounded, and passenger trips may not equal successful validations in competition.',
                    'First-week boundary interpreted as Dec 16-22; publication does not specify exact timestamps.',
                    'Opening-day operating hours unknown; full current weekday profile used for that day.',
                    'Current minute schedule has no confirmed December 2025 effective date.',
                    'No route-5 target history for validation. This is an unvalidated cold-start scenario, not a trained fleet improvement.',
                    'Later December extrapolates the first-week scale; holiday demand and uptake may differ.'])
    (DEST / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
