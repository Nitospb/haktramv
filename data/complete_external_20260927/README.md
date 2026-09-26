# Complete external features for the 2025 tram forecast

Prepared 27 September 2026. The user explicitly reports that organizers permit
retrospective observations from the hidden period. Weather, reported events and
foreign transport indices therefore use actual 2025 dates, including November
and December. Unknown Moscow boardings and vehicle counts are still unavailable.

`all_features_matrix.parquet` contains one row per route/date/hour, 268 numeric
features and the historical aggregated target. It contains no individual
transaction records. `feature_manifest.json` lists input sources and exclusions;
`feature_coverage.csv` distinguishes missing values from zeros. Scheduled POI
features remain missing when the original POI table has no matching row. This
includes uncovered route 5 and mostly hours without a positive stop-event
count, but there are also a few positive-schedule cells with missing POI data;
the two source tables are not assumed identical. The model uses -1 for missing
numeric fields.

## Weather

`route_hour_weather_2025.parquet` covers all 87,600 route-hour cells without gaps.
January–October comes from the supplied route-cell archive. November–December
was downloaded with the same explicit ECMWF IFS model and elevation disabled.
Nine approximately 9 km cells are weighted by the fraction of route stops in
each cell. These are gridded model data, not separate district weather stations.
Snow depth is the same city-centre ERA5-seamless source used in the original
archive. Historical zone climatology for 2022–2023 is a separate reference.

Source and attribution: [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
ECMWF IFS / ERA5, [CC BY 4.0](https://open-meteo.com/en/terms).
Raw November–December responses, parameters, coordinates, units and hashes are
retained for reproducibility in this directory.

## Road incidents

`route_hour_road_incidents_2025.parquet` counts reported Moscow crashes within
300, 700 and 1,500 metres of the nearest listed route stop. It includes current
hour and trailing three-hour counts. The window is a modelling hypothesis,
not a measured disruption duration. Timestamps without offset in the Moscow
source are interpreted as Moscow local time. The supplied stop snapshot is not
a historically versioned route geometry.

The source archive contains 8,354 Moscow crash records for 2025, covering every
month. [Карта ДТП](https://dtp-stat.ru/opendata/) republishes official GIBDD data
with some corrections, including coordinates. These records mainly describe
reported injury crashes. **They are not congestion, car speed or traffic volume
measurements.** Zero counts mean no matching record, not guaranteed clear roads.
Only route-hour aggregates are published here; raw participant-level data stays
outside the repository. The archive checksum and month counts are in
`road_report.json`.

No usable open 2025 street-hour congestion archive was found in the searched
sources. [ЦОДД](https://gucodd.mos.ru/) shows current city conditions;
[TomTom's free CSV list](https://www.tomtom.com/downloads/traffic-index/) does
not include Moscow; [Yandex documentation](https://www.yandex.ru/support/maps/ru/concept/stoppers)
describes current traffic layers; YMArchive's public archive ends in 2015.
None of these was substituted for observed November–December 2025 local traffic.

## Rebuild

From the repository root, with the supplied feature bundle and earlier fleet
study available:

```sh
python prepare_complete_weather.py
python prepare_road_incidents.py
python prepare_all_external.py
```

Run `all_external_study.py` and `verify_all_external.py` on Studio. External city
data are donor indices/profiles, not extra Moscow target observations. Full
coverage and feature inclusion do not imply that every feature improves accuracy.
