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
