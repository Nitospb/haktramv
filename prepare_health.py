"""Historical-only health summaries. No target-period values are joined at inference."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent;F=ROOT/'data/features/07_transaction_derived';OUT=ROOT/'data/enriched';OUT.mkdir(exist_ok=True)
p=pd.read_parquet(F/'validator_peers/route_peer_daily.parquet')
p['date']=pd.to_datetime(p.date).dt.strftime('%Y-%m-%d')
cols=['expected','observed','silent_share','failure_rate','concentration','returning','known_coverage']
p=p[['route','date']+cols].rename(columns={c:'peer_'+c for c in cols})
o=pd.read_parquet(F/'offline_accumulation/offline_hourly.parquet')
o['date']=pd.to_datetime(o.event_hour).dt.strftime('%Y-%m-%d')
cols=['delay_share_30m','delay_share_2h','delay_share_12h','delay_clock_valid_share']
o=o.groupby(['route','date'])[cols].mean().reset_index().rename(columns={c:'offline_'+c for c in cols})
a=pd.read_parquet(F/'operational_asof_hourly.parquet')
a['date']=pd.to_datetime(a.event_hour).dt.strftime('%Y-%m-%d')
cols=['vehicles_seen','validators_seen','failure_rate','intervalidation_gap_median_min']
a=a.groupby(['route','date'])[cols].mean().reset_index().rename(columns={c:'operation_'+c for c in cols})
p=p.merge(o,on=['route','date'],how='outer',validate='one_to_one').merge(a,on=['route','date'],how='outer',validate='one_to_one')
p.to_csv(OUT/'health_daily.csv.gz',index=False)
print('Historical route health:',p.shape,flush=True)
