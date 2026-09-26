# Moscow disruption events, 2025

Public event timestamps aligned to the ten tram routes in the competition
dataset. The collection covers 2025; the train/test history uses January to
October, while November–December is retained for retrospective analysis only.

## Files

- `events_2025_clean.csv` — cleaned event-level records with Moscow timestamps,
  source links, original text, event type, explicit place names and precision.
- `route_hour_disruption_features_2025.csv` — complete
  `route × date × hour` grid: 87,600 rows for all ten routes and all of 2025.
- `known_connectivity_periods_2025.csv` — two policy/announcement periods whose
  precision is only a date or date range.
- `telegram_posts_2025.csv` — deduplicated source archive used by the builder.
- `mos_only/`, `mos_early/`, `mos_first_half/`, `rosaviatsiya/` — collection
  chunks retained for auditability.

## Sources

- Moscow Mayor Sergey Sobyanin, official Telegram channel:
  `https://t.me/s/mos_sobyanin`
- Moscow Oblast Governor Andrey Vorobyov, official Telegram channel:
  `https://t.me/s/vorobiev_live`
- Rosaviatsiya press service: `https://t.me/s/favt_info`
- Connectivity-monitor reports: `https://t.me/s/na_svyazi_helpdesk`
- 5–9 May warning: `https://www.mn.ru/short/operatory-predupredili-ob-ogranicheniyah-interneta-v-moskve-s-5-po-9-maya-chto-izvestno`
- Beeline whitelist launch on 5 September:
  `https://moskva.beeline.ru/customers/press/news/details/zapuskaem-belie-spiski/`

## What was found for January–October

- 259 official drone-related posts on 53 distinct dates.
- 300 airport bulletin/airport rows on 54 dates (279 starts or continued
  restriction bulletins and 21 explicit lift bulletins). These are a proxy for
  regional disruption, not a count of independent attacks.
- Two independent connectivity reports around the 5–9 May restriction period.
- Three Moscow city impact locations explicitly stated in the mayor's posts:
  Domodedovskaya Street (11 March), Kashirskoye Highway (6 May), and Vernadsky
  Avenue (29 May).

No official 2025 post in the collected sources identifies an incident inside a
specific Moscow administrative district traversed by one of the ten routes with
enough precision for a defensible route-local flag. Therefore regional reports
are copied as regional features for every route; they are **not** presented as
events on a particular tram line.

## Important limitations

- A post timestamp is the publication time, not necessarily the start of an
  alert. When Rosaviatsiya states an effective time explicitly, it is parsed
  into `effective_at_msk`; otherwise the publication time is retained.
- There is no public district-by-district log of mobile internet restrictions
  or whitelist activation. `whitelist_mechanism_available` means the mechanism
  had launched, not that it was active in that hour.
- `mobile_internet_restrictions_possible` marks the announced 5–9 May window;
  it does not prove continuous restriction in every district.
- Drone/airport facts occurring after a forecast origin are unavailable future
  information. They may be used for retrospective impact analysis, but must not
  be used directly to forecast November–December from an October origin.

Rebuild cleaned outputs with:

```bash
python3 external_features/build_moscow_disruption_features.py
```
