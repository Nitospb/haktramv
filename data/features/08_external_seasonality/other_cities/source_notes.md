# External transit transfer experiment

This experiment tests whether public multi-year ridership data improves the
Moscow route-hour forecast. Foreign counts are never appended to the Moscow
target. They are converted to within-day hourly shares and normalized calendar
indices, so foreign network scale cannot replace the Moscow scale.

Sources:

- Malaysia KTMB Komuter hourly origin-destination ridership, yearly parquet,
  2023 onward: https://data.gov.my/data-catalogue/ridership_od_komuter
- Malaysia daily public-transport ridership, 2019 onward:
  https://data.gov.my/data-catalogue/ridership_headline
- MTA Roosevelt Island Tram hourly ridership, 2020-2024, aggregated through
  the official Socrata API:
  https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s
- Moscow metro station passenger flow, official quarterly portal data. A
  comparable 2024-Q3 snapshot is joined to tram routes through metro stations
  within 1.2 km of their stops:
  https://data.mos.ru/opendata/62743
- Seoul bus route/stop/hour passenger counts, 24 official monthly files for
  2023-2024. The raw stop rows are aggregated to route-month-hour boarding and
  alighting totals:
  https://data.seoul.go.kr/dataList/OA-12913/S/1/datasetView.do

For every validation fold, source rows after that fold's forecast cutoff are
excluded. This prevents foreign future observations from leaking into a
historical evaluation. The baseline predictions are the saved soft-calendar
XGBoost outputs; only the candidate is retrained with the external features.

The Moscow portal has missing station-quarter rows. These are retained as
missingness/coverage features and median-imputed for magnitude; they are never
interpreted as zero passengers. The Moscow-metro candidate scored 0.831647 on
the five selection folds versus 0.835102 for the foreign profile interaction
model. Giving it 5% ensemble weight changed selection from 0.843762 to
0.843767, but reduced the autumn control score from 0.884038 to 0.882282.
Therefore `submission_external_ensemble.csv` deliberately remains the
pre-Moscow ensemble; `submission_moscow_candidate.csv` is diagnostic only.

The Seoul data produced 368,280 route-month-hour rows for 699 routes, covering
about 3.28 billion boardings. Its standalone selection mean was 0.833751. A 5%
ensemble weight improved selection only from 0.843762 to 0.843779 and reduced
the autumn control score from 0.884038 to 0.883920. It is therefore also kept
as a diagnostic (`submission_seoul_candidate.csv`) rather than replacing the
recommended submission.
