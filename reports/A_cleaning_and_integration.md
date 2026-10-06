# A — Data Cleaning & Integration

## A1 — Cleaning log (2 pts)

Source: EDA in `notebooks/01_cleaning_and_integration.ipynb`.  
Table: `reports/A1_cleaning_log.csv` (one row per issue).

| File | Distinct issues |
|------|-----------------|
| Trips (`ride_demand_train.csv`) | 7 (≥5 required) |
| Weather (`weather_hourly.csv`) | 5 (≥4 required) |
| Events (`events_calendar.csv`) | 6 (≥4 required) |

Columns: `file`, `columns`, `issue_type`, `rows_affected`, `pct_affected`, `fix`, `why`.

Fixes documented in the log are implemented in `src/cleaning.py` for the reproducible pipeline (A2+).

## A2 — Time & key standardization (2 pts)

Source: `notebooks/01_cleaning_and_integration.ipynb` (section A2). Helpers: `src/cleaning.py`.

### (a) Zone & event_type before → after
- **Trips zone:** 55 raw labels → 12 canonical (`CANONICAL_ZONES`).
- **Events zone:** 28 raw labels → same 12 (multi-zone & citywide expanded).
- **Weather:** no zone column (city-wide); joins on timestamp only.
- **event_type:** 20 raw → 8 snake_case types.

### (b) Timestamp parsing
- **Trips:** slash DMY / ISO space / ISO-T `+03:00` — 0 parse failures. Day>12 on slash proves day-first.
- **Weather:** UTC `...Z` / slash local — 0 parse failures.
- **Events:** slash / ISO / month-name — starts 0 failures; end failures = raw missing ends only.

### (c) Clock proof
- Trips: large slice explicitly tagged `+03:00` (Africa/Addis_Ababa).
- Weather `...Z`: mean temperature peaks ~15:00 after UTC→EAT, ~12:00 if Z is wrongly treated as local (+3h = UTC offset).
- All parsed timestamps localized/converted to **`Africa/Addis_Ababa`**.

## A3 — Join map & diagram (2 pts)

Source: notebook section A3. Figure: `figures/A3_join_map.png`.

| Join | Keys | Type | Cardinality |
|------|------|------|-------------|
| trips ← weather | `pickup_hour_eat` | left join | many-to-one |
| trips ← events | zone + hour ∈ `[window_start, window_end]` | interval match → aggregate | many-to-many → 1 row/zone-hour |

**Left table = trips** (forecast grain = zone × hour).

**Event windows** (confirmed only): football (2,2), concert (1,2), sports_run (1,1); conference/exhibition/road_closure/public_holiday/school_break (0,0).

## A4 — Join audit

From `scripts/build_masters.py` / notebook A4:

- Train rows before/after weather+events join: **84,614 → 84,614** (unchanged).
- Weather match rate before fill: **~89%**; missing keys = weather hour gaps after dedupe (~790 unique hours). Treatment: **ffill/bfill** on city hour → 0 null rain after fill.
- Events: confirmed ids matched ≥1 zone-hour **153**; unmatched **6**; cancelled event-zone rows excluded from features.

See `reports/A4_join_audit.csv`.

## A5 — Join proof

Notebook picks 3 zone-hours: rainy (`rain_mm≥2`), football window, public holiday — shows weather fields + matched event ids/features.

## A6 — Feature engineering

`reports/A6_feature_table.csv` — calendar / weather / events / lag features, all `known_at_forecast_time=yes`.

## A7 — Integrity checks

`reports/A7_integrity_checks.txt` — automated PASS/FAIL (duplicate keys, tz, zones, sentinels, train/test columns, no target on test, no ops leakage).

## A8 — Master tables

| File | Role |
|------|------|
| `data/processed/master_train.csv` | cleaned + joined + features + `trips` |
| `data/processed/master_test.csv` | same features, **no** `trips` |
| `data/processed/data_dictionary_master.csv` | column dictionary |

Rebuild: `python scripts/build_masters.py`
