"""Extract official Kazakhstan transport history; exclude 2025+ from model inputs."""
import calendar
import hashlib
import json
import re
from pathlib import Path
import openpyxl
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'data/external_kazakhstan'
URL='https://stat.gov.kz/api/iblock/element/445264/file/ru/'

def build(path):
    w=openpyxl.load_workbook(path,data_only=True)
    data=[];year=None
    modes={'трамваями':'tram','троллейбусами':'trolleybus','автобусами':'bus'}
    for row in w['Автомобильный'].values:
        label=str(row[0]).strip()
        if re.fullmatch(r'20\d\d(?:1\))?',label):
            year=int(label[:4]);continue
        if label not in modes or year is None or year>2024:
            continue
        for month,value in enumerate(row[1:13],1):
            valid=isinstance(value,(int,float)) and np.isfinite(value) and value>0
            data.append(dict(year=year,month=month,mode=modes[label],passengers_thousands=float(value) if valid else np.nan))
    d=pd.DataFrame(data)
    assert d.year.max()==2024 and not d.duplicated(['year','month','mode']).any()
    d['days']=[calendar.monthrange(int(y),int(m))[1] for y,m in zip(d.year,d.month)]
    d['daily_thousands']=d.passengers_thousands/d.days
    return d

def main():
    d=build(DEST/'passengers_monthly.xlsx')
    d.to_csv(DEST/'monthly_history_through_2024.csv',index=False)
    priors={};records=[]
    groups={'tram_recent':('tram',[2023,2024]),'tram_multiyear':('tram',[2016,2017,2018,2019,2023,2024]),'trolley_recent':('trolleybus',[2023,2024])}
    for name,(mode,years) in groups.items():
        z=d[(d['mode']==mode)&d.year.isin(years)].copy()
        assert len(z)==12*len(years) and z.daily_thousands.notna().all()
        z['index']=z.daily_thousands/z.groupby('year').daily_thousands.transform('mean')
        series=z.groupby('month')['index'].median().reindex(range(1,13)).to_numpy()
        series/=series.mean()
        angle=2*np.pi*np.arange(12)/12
        design=np.column_stack([np.ones(12),np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])
        smooth=np.exp(design@np.linalg.lstsq(design,np.log(series),rcond=None)[0]);smooth/=smooth.mean()
        for kind,values in [('monthly',series),('smooth',smooth)]:
            priors[name+'_'+kind]=dict(mode=mode,years=years,index=values.tolist())
        for _,r in z.iterrows():records.append(dict(series=name,year=int(r.year),month=int(r.month),index=float(r['index'])))
    # Changing future external observations must have no effect on extracted inputs.
    w=openpyxl.load_workbook(DEST/'passengers_monthly.xlsx',data_only=True)
    s=w['Автомобильный'];future=False
    for row in s:
        label=str(row[0].value).strip()
        if re.fullmatch(r'20\d\d(?:1\))?',label):future=int(label[:4])>=2025
        if future:
            for cell in row[1:13]:
                if isinstance(cell.value,(int,float)):cell.value=999983
    temp=DEST/'_perturbed.xlsx';w.save(temp)
    try:pd.testing.assert_frame_equal(d,build(temp))
    finally:temp.unlink()
    (DEST/'priors.json').write_text(json.dumps(priors,indent=2)+'\n')
    pd.DataFrame(records).to_csv(DEST/'annual_normalized_indices.csv',index=False)
    manifest=dict(source_url=URL,landing_url='https://stat.gov.kz/ru/industries/business-statistics/stat-transport/dynamic-tables/?period=month',retrieved_on='2026-09-27',source_sha256=hashlib.sha256((DEST/'passengers_monthly.xlsx').read_bytes()).hexdigest(),input_period=[int(d.year.min()),2024],model_years=groups,excluded_years=[2020,2021,2022],future_external_perturbation_passed=True,
                  notes=['Workbook vintage 2026; historical revisions may not match an as-of-2025 snapshot.','Only pre-2025 rows enter models; raw source workbook retained for provenance.','National counts, not Moscow-like route labels; transfer normalized seasonality only.','Monthly series cannot identify holiday-day response. Russian day types remain from Moscow data.','Source notes a 2023 methodology change for individual road transport operators, and revises 2022. Bus not used as donor.','Normalized year levels remove absolute network scale, not all route closures or changes.'])
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{'years':v['years'],'Nov_Oct':round(v['index'][10]/v['index'][9],4),'Dec_Oct':round(v['index'][11]/v['index'][9],4)} for k,v in priors.items()},indent=2))

if __name__=='__main__':main()
