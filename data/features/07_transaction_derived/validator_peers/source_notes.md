# Vehicle-peer validator experiment

The raw fields `garage_number` and `device_no` link validators to vehicles.
Aggregated 62,442,883 transactions into 3,994,491 vehicle/device/hour rows.
There are 490 observed vehicles and 3,080 device IDs. Vehicle is missing for
431,450 transactions, which are excluded from peer comparisons. Card hashes
are not read into outputs.

Vehicle/device baselines use the preceding 28 days. A device becomes expected
after being observed on at least three earlier days. Its normal share is its
own average positive share, not 1/N. On active vehicle-days, record missing
expected devices, very low shares, high failure rates, concentration of payments
and returns after absence. Vehicles need at least 50 transactions per day.

Route features are transaction-weighted means over vehicles and trailing
7/28/84-day summaries. Each forecast row uses only days preceding its simulated
forecast origin. Target-period observations never update the input state.
Original target labels are not filled or inflated for supposed lost payments.
Fully silent never-observed devices cannot be discovered from these logs.

## Findings

- 44,398 active vehicle-days; median observed/expected devices both 6.
- 12,678 vehicle-days have silent expected devices with active peers.
- 28 vehicle-days have a device with >=10 transactions and >=50% failures.
- October 31 snapshot (observations through October 30): 2,409 eligible
  vehicle/device pairs, of which 96 were unseen for seven days while peers
  were active on at least three days.

Absence can be replacement, relocation, uneven passenger choice or missing
logs. These are **not verified broken-validator labels**. Snapshot probability
is a smoothed observation-frequency proxy on the next peer-active day, not a
calibrated probability of hardware failure or a two-month breakdown forecast.

The raw-success audit agrees with 57,550 of 57,551 label rows. One route/hour
(route 1, July 3 at 08:00) has 1,639 raw successful rows versus label 1,638.
The supplied labels remain authoritative and unchanged. This one-record
discrepancy is recorded in `output/raw_label_discrepancies.csv`.

## Validation

Five early windows choose peer features and blend weight before autumn.
The selected narrow health features improve standalone XGBoost early mean
0.824435 → 0.826323 and autumn 0.889123 → 0.889396.

However, the chosen 50/50 incumbent blend is slightly worse than the previously
submitted blend: early 0.842895 → 0.842717; autumn 0.878166 → 0.877701. Thus the
peer experiment is not promoted as a replacement. It beats the original
incumbent locally, but adds no established improvement to the latest blend.
Public score 0.88284 for that earlier blend was reported by the user.

`output/submission_validator_peers.csv` is the experimental blend;
`output/submission_validator_peers_standalone.csv` is the pure peer model.
Both passed 14,640-row template/order, missing/duplicate, integer/non-negative
prediction and route-5 checks. No upload was performed.

Tests confirm unequal expected shares, detection of a missing known validator,
and invariance of past signals to adding a previously unknown future device.

Run `aggregate.py`, `features.py`, `run.py`, `snapshot.py`, and `audit.py` using
`/opt/anaconda3/bin/python3`. Run `test_features.py` for the causal checks.
