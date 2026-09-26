"""Build other-mode Moscow seasonal priors from the user-supplied monthly archive."""
import calendar
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/external_moscow_modes'
MONTHS=['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь']


def build(raw):
    x=raw[pd.to_numeric(raw.Year,errors='coerce').notna()].copy()
    x['year']=x.Year.astype(int);x['month']=x.Month.map(dict(zip(MONTHS,range(1,13))))
    x=x[x.year<=2024].copy();x['passengers']=pd.to_numeric(x.PassengerTraffic)
    assert not x.duplicated(['year','month','TransportType']).any()
    modes={'metro':['Московский метрополитен'],'mcc':['Московское центральное кольцо'],
           'road_combined':['Автобус','Электробус','Троллейбус'],'tram_control':['Трамвай']}
    output=[];priors={}
    angle=2*np.pi*np.arange(12)/12
    design=np.column_stack([np.ones(12),np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])
    for mode,names in modes.items():
        z=x[x.TransportType.isin(names)].groupby(['year','month']).passengers.sum().reset_index()
        z['daily']=z.passengers/np.array([calendar.monthrange(int(y),int(m))[1] for y,m in zip(z.year,z.month)])
        z['mode']=mode;output.append(z)
        for period,years in [('recent',[2023,2024]),('multiyear',[2019,2023,2024])]:
            q=z[z.year.isin(years)].copy()
            assert len(q)==len(years)*12 and q.daily.notna().all()
            q['index']=q.daily/q.groupby('year').daily.transform('mean')
            values=q.groupby('month')['index'].median().reindex(range(1,13)).to_numpy()
            raw_values=values/values.mean()
            priors['moscow_'+mode+'_'+period+'_monthly']=dict(mode=mode,years=years,index=raw_values.tolist())
            values=np.exp(design@np.linalg.lstsq(design,np.log(values),rcond=None)[0]);values/=values.mean()
            priors['moscow_'+mode+'_'+period+'_smooth']=dict(mode=mode,years=years,index=values.tolist())
    return pd.concat(output,ignore_index=True),priors


def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'data/features/08_external_seasonality/moscow_monthly_passengers.csv'
    raw=pd.read_csv(source,sep=';');d,priors=build(raw)
    changed=raw.copy();changed.loc[pd.to_numeric(changed.Year,errors='coerce')>=2025,'PassengerTraffic']='999983'
    dd,pp=build(changed);pd.testing.assert_frame_equal(d,dd);assert priors==pp
    d.to_csv(OUT/'monthly_history_through_2024.csv',index=False)
    (OUT/'priors.json').write_text(json.dumps(priors,indent=2)+'\n')
    manifest=dict(source_file=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_repository='https://github.com/ArtNikVal/Analyseofmoscowtaransportproject',
                  provenance='User-provided archived CSV; its repository attributes data to the Moscow open-data portal. This extraction is not a fresh independently verified portal download.',
                  modes= ['metro','mcc','road_combined','tram_control'],excluded_years=[2020,2021,2022],future_external_perturbation_passed=True,
                  notes=['Bus, electric bus and trolleybus are combined to reduce reclassification effects.','MCC is not MCD. No MCD series is present in this archive.','Monthly prior smooths broad seasonality; Russian holiday/daytype handling remains in the Moscow forecasting model.','2019 versus 2023/2024 may differ due to route openings and network changes. Normalize annual scale; this cannot remove all structural breaks.'])
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:dict(Nov_Oct=round(v['index'][10]/v['index'][9],4),Dec_Oct=round(v['index'][11]/v['index'][9],4)) for k,v in priors.items()},indent=2))

if __name__=='__main__':main()
