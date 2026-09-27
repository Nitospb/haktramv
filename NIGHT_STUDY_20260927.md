# Overnight Studio study — 27 September 2026

User authorized an overnight exploration, including 1–2 hour larger training
runs on their Mac Studio. Aim is improvement toward public score 0.95; no
promise that this is attainable. Target delivery: morning, approximately 09:00
Moscow time on 27 September. Do not start new training after 07:30; finalize
available results by 09:00. A thread heartbeat runs every 30 minutes until 09:30.

## Locations and current best

- Local: `/Users/nikitatolochko/Documents/ChatGPT/хак`.
- Studio: `ssh -i ~/.ssh/mac-studio -o IdentitiesOnly=yes -o BatchMode=yes khan@5.16.21.103`.
- Remote root: `/Users/khan/tram-forecast-20260926`; interpreter `.venv/bin/python`.
- Remote output: `outputs/night_20260927/`.
- Confirmed public best: `outputs/submission_best.csv`, **0.89402**, SHA-256
  `f2bb6115595f45a28790e6002defbcf4cc0ec6d9422c0c485c6017a0d63f9425`.
- Latest local direct candidate: 233 external features, CatBoost-independent
  LightGBM at 1600 trees, Sep–Oct **0.8999364**, public score unknown:
  `outputs/longer_moscow_external_20260927/submission_direct_1600_autumn.csv`.
- Existing optional route-5 official-aggregate scenario is separate and untested.

## Bounded jobs

```sh
cd /Users/khan/tram-forecast-20260926
caffeinate -i .venv/bin/python -u night_catboost_study.py --hours 2
caffeinate -i .venv/bin/python -u night_sequence_study.py --hours 2 --steps 6000
# Queued after the initial sequence process exits:
.venv/bin/python -u night_large_launcher.py
```

Jobs use their own output-directory `run.lock` with an exclusive OS lock, plus
`status.json` and checkpoint files. Launch through `nohup` with separate logs
`outputs/night_20260927/catboost.log` and `sequence.log` so they survive SSH
disconnects. Do not start duplicate jobs. Check actual PIDs and timestamps first.
`caffeinate` is process-bound; no permanent power-management changes are made.

Both programs passed smoke runs on Studio: CatBoost absolute and residual modes
at 12 trees; the sequence model at 8 gradient updates with MPS. Smoke runs verify
execution and saved-model replay, not forecasting quality.

The initial 3,686,616-parameter network is faster than expected after MPS warmup.
A larger variant is therefore queued: `night_sequence_large.py --hours 2
--steps 12000`, output `outputs/night_20260927/sequence_large/`. Its width is 384,
with six encoder and four decoder layers. Checkpoints: 500/1500/3000/6000/12000.
`night_large_launcher.py` waits on the first sequence job's lock, preventing
the two networks from competing for MPS. It will not launch after 07:30 Moscow.
The large variant also excludes cancelled-service slots from training loss,
matching the boosted-model retained rows and the inference service multiplier.
The initial medium variant retained these target slots. Thus this is a new
candidate, not a clean one-factor test of capacity alone. Do not attribute a
score difference solely to network size.

### CatBoost

1. Depth 9, up to 6000 trees, MAE, all 233 local external features.
2. Depth 8, up to 6000 trees, MAE residual relative to a causal weekday profile.
   Pseudo-origins every 14 days, horizons up to 61 days, profiles with decay
   half-lives 14/28/56/112 days. Repeated training labels have multiplicity
   correction; scaled residual sample weights recover absolute-error emphasis.

Checkpoints 1000/3000/6000. Select each family's tree count using mean WAPE-score
on origins 151/181/212 (June 1, July 1, August 1). Fit and evaluate that selected
configuration on origin 243 (September 1) without using its score for the initial
selection. Then refit through October 31 (origin 304) for Nov–Dec. Six CPU threads.

### Sequence network

Shared-route Transformer, hidden width 192, six attention heads, four encoder
and three decoder layers. It sees 56 past days of 24-hour route profiles plus
the global cross-route profile, and a sequence of 61 future calendar, weather,
schedule and incident feature tokens. Predicts a bounded multiplicative residual
to the causal weekday baseline, jointly for all 61 × 24 future hours.

Loss combines absolute hourly error, daily-total error (weight 0.15), and
weekly-total error (weight 0.05). Train-tail targets after the fitting cutoff
are masked. Batch 24, AdamW, dropout 0.15, up to 6000 updates; checkpoints
500/1500/3000/6000. Same selection/control/final origins as above. MPS acceleration;
four CPU threads. State and optimizer are saved for resumption. On resume the
data sampler restarts its seed; model/optimizer resume, not an exact uninterrupted
training trajectory.

## Validation boundaries

- Full 61-day forecasts, no actual Moscow target feedback inside the window.
- Future external observed weather and reports are allowed by user instruction.
- Missing future vehicle counts and passenger transactions are not invented.
- Profile features passed future-target perturbation tests.
- These are previously explored, overlapping development windows. September is
  a control for this particular series, not a globally untouched holdout.
- Do not promote a candidate to public best based only on local scores.
- Route 5 is zero in standalone models because no positive historical labels
  exist. Its separate public-aggregate scenario can be attached and documented.

## Follow-up workflow

1. Inspect status/logs and processes. Fix real failures; resume only incomplete
   work. The per-invocation wall limit is two hours. Another invocation may
   resume completed checkpoints if time remains before 07:30.
2. Compare complete 61-day results to causal profile baselines and saved v8 ×1.03
   or latest 233-feature model on matching windows. Break errors down by route,
   weekday, hour, volume bias and late forecast weeks.
3. If substantial time remains, use the observed failure mode to motivate one
   additional experiment. Do not pad runtime or endlessly sweep configurations.
   Keep standalone candidates; the user previously requested no mixing.
4. Export 1–3 best defensible candidates, integer/nonnegative 14640-row semicolon
   CSV, unique route/date/hour, Nov 1–Dec 31. Recompute WAPE and replay saved
   models. Report local versus public evidence separately.
5. Download via SSH tar if SCP download hangs. Validate archive member paths
   before extraction. Keep large optimizer states on Studio; publish reproducible
   code, reports and practical inference artifacts, checking GitHub file limits.
6. GitHub publishing is authorized: `git push origin HEAD:main` to
   `Nitospb/haktramv`. Never publish individual transactions or SSH keys. Preserve
   concurrent README.md/reproduce_best.py/directional/v18–v24 work and unrelated
   ledger edits; stage only owned files. The current branch is
   `codex/publish-tram-project`.
7. By 09:00 Moscow time, write the Russian morning report with artifact links
   even if no model approaches 0.95. State actual outcomes. Disable the heartbeat
   after completion (automation id `studio`). No new training after deadline.

## First heartbeat, approximately 03:05 Moscow

Both original drivers were alive and advancing; no duplicate training launched.
The medium sequence study is complete and chose step 500; Sep–Oct score is
0.838431. CatBoost produced Sep–Oct 0.893446 (absolute, 6000 trees) and 0.848603
(profile residual, 3000 trees). The larger network was still training origin 181
at the initial inspection; inspect its live status before further work.

`night_diagnostics.py` is now available locally and on Studio. Run it again after
remaining jobs finish. It recomputes every saved forecast's WAPE, checks all
14640 keys and target equality against the matrix, and writes `diagnostics/`
scores, route/weekday/hour/forecast-week error slices, and verification JSON.
The first snapshot's scores are downloaded locally; detailed slices are still
on Studio. No models or active-best file were changed.

Important diagnosis: on Sep–Oct, plain weekday profiles with half-life 14/28/56/
112 days have volume biases -15.84%/-14.44%/-10.83%/-7.94%. The medium network
retains -11.75% bias, and boosted profile residual -9.76%. Their summer-anchored
level is a concrete failure mode. The direct external-feature model has +0.71%
bias and score 0.899936. More steps already worsened June validation for both
networks. Consider a calendar-conditioned historical profile (school term versus
vacation, with partial pooling) as one further motivated experiment after the
current studies complete; do not simply extend an overfitting network blindly.

Operational note: Studio's login shell is fish, so Python heredocs need an
explicit `bash` wrapper. A bulk SSH tar snapshot of all predictions stalled at
low throughput and its local SSH process was terminated; the incomplete file
`/tmp/haktramv-night-snapshot.tar.gz` was NOT extracted. Reading the small scores
CSV directly over SSH succeeded. Prefer small report bundles first, then only
the selected model/candidate artifacts. Model/optimizer checkpoints remain intact
on Studio.

## Second heartbeat, approximately 03:32 Moscow

All three original studies finished successfully; no training processes remained
before launching the follow-up. Large Transformer selected step 500 and scored
0.836139 on Sep–Oct, below both the small network and the direct models. Reports,
status, metrics and detailed error slices are now downloaded. The diagnostics
recomputed **77 forecasts**, checking 14640 unique keys and every target against
the source hourly matrix. Larger/longer neural training did not improve the
relevant 61-day validation. Keep this result in the morning report.

One motivated additional study is running on Studio:

```sh
caffeinate -i .venv/bin/python -u night_calendar_analog_study.py --hours 1.5
```

Output `outputs/night_20260927/calendar_analog/`, log
`outputs/night_20260927/calendar_analog.log`. Python PID at launch **70710**,
own OS lock and 07:30 hard deadline. It passed a 12-tree smoke run and a
future-target perturbation check. Do not launch a duplicate. It compares
absolute and normalized-residual CatBoost models with causal calendar-matched
profiles: 112-day age decay, soft matching summer/school/public-holiday regimes,
weekday type, plus a second profile with soft temperature matching. The latter
uses permitted historical future weather, never future targets. Profiles retain
two pooled pseudo-observations. Selection on June/July/August chooses 500/1500/
3000 trees separately for the two families, followed by September control and
October-inclusive final refit. No mixing with the public-best submission.

`night_diagnostics.py` now also includes calendar_analog outputs. Rerun when
complete, compare controls/bias and selection stability before choosing morning
candidates. Direct prior model's Sep–Oct failures concentrate on route 7
(+8.79% volume bias), route 25 (-9.09%), and weekends (about -5%). Do not tune
post-hoc route multipliers on this inspected control and present them as unseen
validation gains.

Small report download succeeded using `ssh -C ... tar -czf -` with only JSON,
metrics and diagnostics. Archive `/tmp/haktramv-night-reports.tar.gz` was fully
validated and extracted. Models/optimizer checkpoints are still only on Studio;
the previous direct LightGBM candidate and its model are already local.

## Third heartbeat, approximately 04:05 Moscow

Calendar analog study completed normally. Its standalone term-matched baseline
scored **0.898480** on Sep–Oct (weather analog 0.898292), versus unconditioned
profile112 0.871264 and profile56 0.849782. Thus the calendar improvement at the
same 112-day half-life is 0.02722, not the full 0.04870 difference to profile56.
On July/August starts, baseline_term scored 0.790268/0.884569. The learned
corrections worsened autumn: absolute 0.878280, residual 0.885622. The new baseline
does not beat the prior direct LightGBM 0.899936 and has no public score yet.
Reports, candidate `calendar_analog/submission_baseline_term.csv`, all four driver
logs, and updated diagnostics (105 independently recomputed forecasts) are local.
Bundle `/tmp/haktramv-night-analog.tar.gz` validated before extraction.

One final focused neural check is now running, after checking that all earlier
training processes had exited:

```sh
caffeinate -i .venv/bin/python -u night_sequence_calendar.py --hours 1
```

Own output/lock `outputs/night_20260927/sequence_calendar/`, separate driver log
`outputs/night_20260927/sequence_calendar.log`, hard deadline 07:30. This reuses
the 3.69M medium Transformer but corrects its baseline to the calendar analog;
masked cancelled slots, LR 5e-5, checkpoints **0/50/150/500/1500**. Zero-step is a
real eligible choice, so there is no forced learned correction if it hurts.
The 8-update smoke run passed saved-model replay and initial-baseline equality;
causal profile perturbation check passed. Selection/control/final origins stay
151/181/212/243/304. This is not a single-factor ablation (baseline, masking and
optimization changed). `night_diagnostics.py` includes this new family.

Next: inspect this final focused run; do not start more broad searches merely
to occupy the night. Recompute diagnostics, verify/replay selected candidate
predictions, and prepare 1–3 deliverables. Defensible current choices are the
existing direct1600 candidate, calendar baseline (or new neural correction only
if validated), and an explicitly untested route-5 aggregate scenario applied to
the direct candidate. Preserve public best. Finish report and disable heartbeat
once deliverables are complete, by 09:00 at latest; there is no requirement to
keep an unproductive training process alive until morning.

## Completed, fourth heartbeat, approximately 04:35 Moscow

The final calendar Transformer completed and selected 500 updates. It improved
July but scored 0.883527 on Sep–Oct, below the uncorrected calendar baseline
0.898480. No further training is justified by this series. All five study
directories are complete and the process check found no remaining night jobs.

Final Russian report: `outputs/night_20260927/REPORT_RU.md`. Three deliverables
are in `outputs/night_20260927/deliverables/`: direct1600, calendar baseline, and
the separate direct1600 route-5 scenario. The compact 1600-tree inference model
is included. `night_package.py` replayed nine forecasts, verified the exact
14640-cell submission grid and scenario isolation. `night_infer_deliverables.py`
then independently regenerated all three CSVs byte-for-byte on Studio, without
training. Updated diagnostics verify 124 saved validation predictions across
four overlapping cutoffs. No public score improvement is established.

All chosen artifacts, reports and logs have been downloaded with hash checks.
The public-best local file still has the recorded f2bb611... hash. Studio's
top-level best file was an older different copy; it was also left untouched.
Packaging checked an explicit reference copied from the authoritative local best
instead of overwriting Studio's existing file.

After the final report is committed/published, disable heartbeat `studio`.
This is the end of the overnight study; do not restart any completed jobs.
