"""Optional route-5 public-aggregate scenario on the new standalone forecast."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from route5_public_anchor import integer_allocation

ROOT=Path(__file__).resolve().parent
KEY=['route','date','hour']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',default='outputs/longer_moscow_external_20260927/submission_longer_direct.csv')
    args=parser.parse_args()
    source=ROOT/args.base; dest=source.parent
    base=pd.read_csv(source,sep=';')
    assert base.loc[base.route.eq(5),'prediction'].eq(0).all()
    five=pd.read_csv(ROOT/'data/schedule_research_20260927/route5_current_minute_schedule.csv')
    five=five[five.stop_sequence<16].copy();five['hour']=five.minute_of_day//60
    opportunities=five.groupby(['weekday','hour']).size().rename('stop_opportunities').reset_index()
    supply=pd.read_csv(ROOT/'data/features/06_schedules/gtfs_hourly_features.csv')
    donor=base[base.route.isin([7,50])].merge(supply[KEY+['scheduled_stop_events']],on=KEY,how='left',validate='one_to_one')
    donor['intensity']=donor.prediction/donor.scheduled_stop_events.replace(0,np.nan)
    intensity=donor.groupby(['date','hour']).intensity.median().reset_index()
    grid=base[base.route.eq(5)][KEY].copy();grid['weekday']=pd.to_datetime(grid.date).dt.dayofweek
    grid=grid.merge(opportunities,on=['weekday','hour'],how='left',validate='many_to_one').merge(intensity,on=['date','hour'],how='left',validate='one_to_one')
    grid['intensity']=grid.intensity.fillna(grid.groupby('date').intensity.transform('median')).fillna(0)
    shape=grid.stop_opportunities.fillna(0)*grid.intensity
    shape[grid.date<'2025-12-16']=0
    week=grid.date.between('2025-12-16','2025-12-22');assert shape[week].sum()>0
    grid['prediction']=np.rint(shape*40000/shape[week].sum()).astype(int)
    grid.loc[week,'prediction']=integer_allocation(shape[week],40000)
    out=base.copy();mask=out.route.eq(5)
    out.loc[mask,'prediction']=grid.set_index(KEY).prediction.reindex(pd.MultiIndex.from_frame(out.loc[mask,KEY])).to_numpy()
    pd.testing.assert_frame_equal(base.loc[~mask],out.loc[~mask])
    assert len(out)==14640 and not out.duplicated(KEY).any() and out.prediction.ge(0).all()
    assert out.loc[mask&(out.date<'2025-12-16'),'prediction'].eq(0).all()
    assert out.loc[mask&out.date.between('2025-12-16','2025-12-22'),'prediction'].sum()==40000
    path=dest/'submission_moscow_with_route5.csv';out.to_csv(path,sep=';',index=False)
    grid.to_csv(dest/'route5_public_scenario.csv',index=False)
    report=dict(source_url='https://transport.mos.ru/mostrans/all_news/127782',public_score=None,
        base=str(source.relative_to(ROOT)),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        submission=path.name,submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),route5_total=int(grid.prediction.sum()),
        first_week_sum=40000,assumed_first_week=['2025-12-16','2025-12-22'],other_nine_routes_identical=True,
        method='Current route-5 timetable weighted by new standalone route 7/50 predictions per scheduled stop event. Rounded official first-week total sets scale.',
        limitations=['Official passenger-trip aggregate may differ from competition validations.',
                    'First-week dates inferred from launch date; exact boundaries not specified by source.',
                    'Current minute timetable is not verified for Dec 2025; launch-day hours unknown.',
                    'Remaining December extrapolates first-week scale. No positive route-5 labels for validation.'])
    (dest/'route5_public_scenario_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
