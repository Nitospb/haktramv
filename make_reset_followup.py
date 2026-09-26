"""Two follow-up probes using immutable v8 and the confirmed +3% winner."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from make_reset_probes import integer_reconcile
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'outputs/reset_probes_20260926'
KEY=['route','date','hour']
base_path=ROOT/'outputs/studio/v8/submission.csv'
assert hashlib.sha256(base_path.read_bytes()).hexdigest()=='b8f8238db449a7401f9d63543c60288b018ed9a1e991ba04745825f82d056524'
base=pd.read_csv(base_path,sep=';')
winner_path=DEST/'submission_02_level_up3.csv'
assert hashlib.sha256(winner_path.read_bytes()).hexdigest()=='f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425'
winner=pd.read_csv(winner_path,sep=';')
template=pd.read_csv(ROOT/'data/test_submission.csv',sep=';')
assert base[KEY].equals(winner[KEY]) and base[KEY].equals(template[KEY])
frame=winner.copy();frame['month']=frame.date.str[5:7]
p=frame.prediction.to_numpy(dtype=float).copy()
for _,ix in frame.groupby('route',sort=False).indices.items():
 nov=ix[frame.month.to_numpy()[ix]=='11'];dec=ix[frame.month.to_numpy()[ix]=='12']
 moved=.03*p[nov].sum()
 if p[dec].sum()>0:
  p[nov]*=.97
  p[dec]*=1+moved/p[dec].sum()
variants={'06_level_up6':np.rint(base.prediction.to_numpy()*1.06).astype(np.int64),
          '08_level_up8':np.rint(base.prediction.to_numpy()*1.08).astype(np.int64),
          '10_level_up10':np.rint(base.prediction.to_numpy()*1.10).astype(np.int64),
          '07_winner_december_up':integer_reconcile(frame,p,['route'])}
reports=[]
for name,p in variants.items():
 out=base[KEY].copy();out['prediction']=p
 assert len(out)==14640 and out[KEY].equals(template[KEY])
 assert np.isfinite(p).all() and (p>=0).all()
 assert (out.loc[out.route==5,'prediction']==0).all()
 file=DEST/('submission_'+name+'.csv');out.to_csv(file,sep=';',index=False,lineterminator='\n')
 check=pd.read_csv(file,sep=';');pd.testing.assert_frame_equal(out,check)
 if name.startswith('07'):
  pd.testing.assert_series_equal(out.groupby('route').prediction.sum(),winner.groupby('route').prediction.sum())
 reports.append({'submission':str(file.relative_to(ROOT)),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'rows':len(out),'total':int(p.sum()),'public_score':None})
results={'01_level_down3':(.88400,'23:55'),'02_level_up3':(.89402,'23:54'),'03_month_tilt':(.88975,'23:55'),'04_weekends_up6':(.88905,'23:54'),'05_weekly_shape':(.88643,'23:56')}
public=[]
for name,(score,tm) in results.items():
 path=DEST/('submission_'+name+'.csv')
 public.append({'submission':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score':score,'submitted_at':'2026-09-26 '+tm+' +03:00','source':'User explicitly identified each submitted probe and reported competition score'})
# Retain later user-reported scores when recreating these immutable probes.
ledger_path=ROOT/'outputs/leaderboard_results.json'
if ledger_path.exists():
 for record in json.loads(ledger_path.read_text()):
  if record.get('score') is not None and record.get('submission','').startswith('outputs/reset_probes_20260926/'):
   public=[x for x in public if x['submission']!=record['submission']]+[record]
for item in reports:
 match=next((x for x in public if x['submission']==item['submission'] and x['sha256']==item['sha256']),None)
 if match:item['public_score']=match['score']
(DEST/'public_results.json').write_text(json.dumps(public,indent=2)+'\n')
(DEST/'followup_report.json').write_text(json.dumps({'previous_results':public,'confirmed_best_score':.89402,'confirmed_best_submission':str(winner_path.relative_to(ROOT)),'new_probes':reports,'hypotheses':{'06':'V8 +6% total: test beyond confirmed +3% winner.','07':'Confirmed +3% winner; November -3% relative to winner and compensated December increase; exact winner total per route.'},'base_source':'Frozen v8 artifact; live best file may be managed by concurrent experiments.'},indent=2)+'\n')
print(json.dumps(reports,indent=2))
