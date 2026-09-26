# Complete external inputs and longer training

Experiments ran on Studio on 27 September 2026. The confirmed public best remains
**0.89402**, and `outputs/submission_best.csv` is unchanged. All new public scores
are unknown. Scores below are local development results.

## Inputs

The previous fleet experiment used 40 features. The assembled matrix has 268:
calendar, education, UAV/connectivity reports, service changes, schedules,
transfers, 75 stop-nearby POI features, hourly route-weighted weather, climate
normals/anomalies, nearby road crashes, and downloaded transport donor profiles
from Moscow and other cities.

The user explicitly authorized retrospective hidden-period external observations.
Actual weather and event reports for November–December enter prediction in the
same way as historical dates. The nine ECMWF IFS weather cells now cover all
2025 dates: 87,600 route-hour rows, no missing weather. Source and attribution:
[Open-Meteo / ECMWF / ERA5](https://open-meteo.com/en/docs/historical-weather-api).

Schedules and POI remain source snapshots, not independently verified daily
historical observations. See `FLEET_SCHEDULE_RESEARCH.md` for provenance. Assigning
a 2025 date to a repeating timetable does not establish historical validity.
Route 5's current minute timetable is not verified for December 2025.

The [DTP Map archive](https://dtp-stat.ru/opendata/) supplies 8,354 Moscow crash
records for 2025, including November–December. Features count events within
300/700/1500 metres of route stops, in current/trailing-three-hour windows.
These are reported crashes, **not congestion, vehicle speed or traffic volume**.
No usable open street-hour 2025 traffic-flow archive was located in the searched
sources. Search results and limitations: `data/complete_external_20260927/road_report.json`.
Only crash aggregates are published, not raw participant records.

Actual hidden-period Moscow vehicle and transaction statistics were not found.
Vehicle counts are therefore predicted. Known-fleet results are historical
diagnostics only. The feature manifest explicitly lists exclusions, source
coverage and missingness. Individual passenger transactions are not published.

## Controlled 800-tree comparison

Each column is a complete 61-day fixed-origin forecast. The windows overlap and
have been used in earlier experiments; they are not independent leaderboard tests.
Models within each comparison use the same training rows, weights and settings.

| Model | July–August | August–September | September–October |
|---|---:|---:|---:|
| Previous direct, 40 features | 0.78916 | 0.89144 | 0.89489 |
| Previous predicted-fleet model | 0.79756 | 0.88809 | 0.89455 |
| v8 × 1.03 | 0.79115 | 0.88421 | 0.89886 |
| All 268, direct | 0.81028 | 0.86900 | 0.86249 |
| All 268, predicted fleet | 0.81092 | 0.88091 | 0.86543 |
| Local 233, direct | 0.78869 | 0.89530 | 0.89975 |
| Local 233, predicted fleet | 0.79324 | 0.89311 | 0.89749 |

The 233-feature variant jointly removes 35 columns: numeric date/year trend and
transport donor profiles/indices, including Moscow monthly aggregates. Weather,
climate, holidays, education, UAV/connectivity, service changes, schedules, POI
and road crashes remain. This joint ablation does not identify the individual
effect of each removed feature.

On September–October, total direct forecast volume was 9.44% below truth with
268 features, versus 0.80% above with 233. This points to a transfer-of-level
problem, but does not establish which individual column caused it.

In the 233-feature variant, removing road crashes changes the autumn score from
0.89975 to 0.89929; removing UAV/connectivity changes it to 0.89949. Effects on
other windows are mixed: a large, stable benefit from either group is unproven.
Known historical fleet gives 0.90455 on the finite-fleet autumn subset; matched
subset controls are stored in `metrics.csv`.

## Longer training

The same models were continued from 800 trees, retaining identical first-800
predictions, rows, weights and learning rate 0.035. No validation target entered
fitting. Checkpoints were evaluated at 800, 1600 and 3200 trees.

| Trees | All 268 direct, autumn | All 268 fleet, autumn | Local 233 direct, autumn | Local 233 fleet, autumn |
|---:|---:|---:|---:|---:|
| 800 | 0.86249 | 0.86543 | 0.89975 | 0.89749 |
| 1600 | 0.86317 | 0.86671 | 0.89994 | 0.89715 |
| 3200 | 0.86383 | 0.86792 | 0.89961 | 0.89644 |

Four times as many trees slightly improve the all-feature variant, but do not
repair its regression. In the local direct model, 1600 slightly improves autumn
validation; 3200 regresses while training score keeps improving. This is
consistent with overfitting on this window, not evidence that other model
architectures could never improve.

Mean-of-three-window selection chooses 3200 for the local direct model and 800
for the local fleet model. A separate 1600-tree file selects the autumn maximum.
That is an additional exploratory choice on an already inspected window, not
a new independent test. Full training and validation curves are retained.

## Candidates

- `outputs/longer_moscow_external_20260927/submission_direct_1600_autumn.csv`:
  standalone local-external model, 1600 trees. Route 5 remains zero because it
  has no positive historical labels.
- `outputs/longer_moscow_external_20260927/submission_moscow_with_route5.csv`:
  identical other-nine-route predictions plus a separate route-5 scenario
  anchored to the official rounded 40,000 first-week passenger trips. Assumed
  dates are December 16–22. Reported trips may differ from competition
  validations, and the current timetable is not confirmed for that period.
  This correction has no historical validation score.
- `outputs/longer_moscow_external_20260927/submission_longer_direct.csv`:
  3200-tree direct model selected on the mean of all three windows.
- All 268-feature candidates are retained in `outputs/all_external_20260927/`
  and `outputs/longer_all_external_20260927/`.
- Source data, the assembled matrix, feature names, missingness and provenance:
  `data/complete_external_20260927/`.

Candidate forecast volumes differ from the confirmed best. Local improvement
does not establish a public-score improvement; the best was not replaced.

Saved-model predictions and WAPE calculations were replayed, and submission
keys, integer/nonnegative values and target exclusion checked. Each experiment
contains `verification.json`. The route-5 variant preserves all nine other
routes exactly and sums to 40,000 in the assumed first week.

## Reproduction on Studio

From `/Users/khan/tram-forecast-20260926`:

```sh
python -m pip install -r requirements-all-external.txt
python all_external_study.py
python all_external_study.py --moscow-only
python verify_all_external.py
python verify_all_external.py --moscow-only
python longer_all_external_study.py
python longer_all_external_study.py --moscow-only
python verify_longer_external.py
python verify_longer_external.py --moscow-only
python export_external_checkpoint.py
python attach_route5_to_external.py --base outputs/longer_moscow_external_20260927/submission_direct_1600_autumn.csv
```

Feature preparation is documented in `data/complete_external_20260927/README.md`.
It requires the supplied feature bundle, previous fleet-model artifacts and
schedule files. Moscow target values after each fitting cutoff are not used.
