# Education calendar experiment

Starts from the feature architecture of the saved public-score 0.88408 XGBoost,
including calendar, timetable/POI, past weather/operations, district weather and
transfer accessibility. Does not use the prior analog-forest architecture.

Three paired variants keep the same XGBoost hyperparameters and seed:

- `baseline_matched`: original features, refitted locally;
- `education`: adds typed school holidays, time inside and until the end of
  school/holiday periods, return-to-school indicators, university calendar
  proxies and route/hour exposure interactions;
- `education_transfer`: also removes raw annual date coordinates (month, week,
  day, day-of-year, annual harmonics and trend), allowing shared academic phases
  to generalize from spring to autumn.

School dates are inherited from `baseline/weather_daily.csv`. Their accuracy
for each school is not asserted. The student features are explicit generic
assumptions: July–August summer, late January–early February winter break,
January/June/late December exams, other periods teaching. They are not actual
verified calendars for all Moscow universities or matched individual campuses.

For context, HSE's official academic calendar demonstrates a different modular
schedule and late-December exams:
https://studyabroad.hse.ru/incoming/year
The page was consulted on 2026-09-25. Its exact dates are not backfilled as
universal facts into 2025 training or validation features. Semester-age and
attendance proxies are modeling assumptions, not observed student counts.

Training uses no target-period weather/transaction observations. Five early
rolling windows select the education variant and blend weight, then the
September–October control is evaluated. Control has been inspected in previous
experiments, and overlapping folds are correlated. New feature selection is
not proof of a causal student effect.

Both the standalone education submission and the blend selected on early
windows are saved, even if they fail the promotion check. `recommend` in the
report requires positive blend weight and better autumn performance than the
saved incumbent. A zero selected weight means the selected submission repeats
the incumbent; the standalone file remains a new experimental submission.

User reported public score 0.86746 at 2026-09-25 19:58 +03:00. Exact uploaded
file was not identified, so that score is not attached to a specific file hash.

Run with `/opt/anaconda3/bin/python3 experiments/education_calendar/run.py`.

## Completed results

37 education features added. Selected before inspecting the autumn scores:
`education_transfer`, weight 1.0 (no blend).

| Variant | Mean of five early windows | September–October |
|---|---:|---:|
| Saved incumbent | 0.804829 | 0.848292 |
| Education | 0.818548 | 0.863748 |
| Education plus removal of annual date coordinates | 0.827243 | 0.886449 |

Selected variant beats the saved incumbent on all six evaluated windows.
Autumn aggregate bias changes from -5.764% to +2.181%; September is still
overpredicted while October's aggregate prediction is close to actual.
This is local validation, not a new leaderboard score or causal attribution.

Recommended file: `output/submission_education.csv` (identical to the selected
file). SHA-256: `a36b4d10282c0a5d64bcdbddcab2e78a64adb893c1ebdc4cdad57ef447f39596`.
14,640 ordered template keys; no missing, duplicate or negative predictions;
integer predictions; route 5 zero. Calendar boundary/partition checks passed.
Final November–December total is 12,404,611 versus the incumbent's 12,645,014.
No upload was performed.
