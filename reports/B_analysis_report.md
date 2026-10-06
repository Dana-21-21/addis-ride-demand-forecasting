# B — Data Analysis Report

Source: `notebooks/02_analysis_report.ipynb` on `data/processed/master_train.csv`.
Each task: result + short interpretation (Deliverable B, 14 pts).

## B1 — Demand patterns

### B1.1 Volume by zone

Top zones by volume: Merkato (12.1%), Bole (11.7%), Megenagna (11.5%). City mean trips/zone-hour ranges 18.0–41.0. Ayat launches mid-March (not full-period), so its share understates eventual weight.

### B1.2 Hour-of-day profile by zone type

Zone types from weekday shapes — residential (morning ~07:00), business (evening ~18:00), market (midday ~13:00), transport hub (evening), nightlife/airport (early morning + weekend lift). business: peaks 18:00 (mean 77.5), quietest 02:00 | market: peaks 13:00 (mean 91.4), quietest 00:00 | nightlife_airport: peaks 06:00 (mean 58.3), quietest 02:00 | residential: peaks 07:00 (mean 48.0), quietest 02:00 | transport_hub: peaks 18:00 (mean 90.2), quietest 01:00

### B1.3 Weekday vs weekend

Weekend-busier: Bole (1.31), Gerji (1.08), Ayat (1.08). Weekend-collapse: Kazanchis (0.51), Piassa (0.49), Arat Kilo (0.49). Arat Kilo stands out — weekday office/commerce demand drops sharply Sat–Sun (means: Mon=24.2, Tue=24.6, Wed=23.9, Thu=22.9, Fri=25.1, Sat=11.9, Sun=11.5).

### B1.4 Trend

Weekly city trips rise from ~44,816 (early full weeks) to ~65,857 (late full weeks): +46.9%. For November forecasts, a positive trend term (or recent lag levels) is essential — a January baseline would under-predict by roughly that growth gap.

## B2 — Weather

### B2.1 Timezone check

Temperature peaks at 15:00 EAT — consistent with local afternoon heat, so weather is correctly on Africa/Addis_Ababa after UTC→EAT conversion. At 0h shift, corr(trips, rain)=0.174 and rainy/dry=1.78; at +5h these fall to -0.015 and 0.92. A 3h UTC mistake would materially weaken the rain feature.

### B2.2 Rain effect by zone type

Matched rainy (≥0.5 mm) vs dry same zone·weekday·hour: lift in transport_hub ×1.37, nightlife_airport ×1.34, residential ×1.33, business ×1.33; demand falls in market ×0.83. Rain does not increase demand everywhere — Merkato (market) softens, consistent with outdoor trading pausing in wet weather.

### B2.3 Rain dose-response

City-wide dose-response (vs dry baseline): none=1.00 → light=1.12 → moderate=1.26 → heavy=1.47. Light rain already lifts demand (+12%); moderate and heavy keep rising (heavy 1.47). Not a pure straight line — early step from none→light is large relative to rainfall amount; zone types differ in slope (see table).

## B3 — Events & calendar

### B3.1 Public holidays

Holiday city-day vs nearby same weekdays — rank (low→high): Genna (Ethiopian Christmas) (0.72); Good Friday (0.78); Eid al-Adha (0.82); Eid al-Fitr (0.84); Enkutatash (Ethiopian New Year) (0.84); Mawlid (0.85); Downfall of the Derg (0.85); International Labour Day (0.85); Patriots' Victory Day (0.88); Fasika (Ethiopian Easter) (0.97); Meskel (Finding of the True Cross) (0.99); Timkat (Epiphany) (1.00); Adwa Victory Day (1.02). 10/13 holidays clearly reduce demand; not all do — e.g. Adwa Victory Day hold flat/up. Local effects vary (see strongest_local_drop/rise): business zones often drop more than residential.

### B3.2 Football event windows

Football match-zone uplift — pre_2h ×2.17, during ×1.44, post_2h ×2.85 (vs same zone·dow·hour without football). Largest window: post_2h.

### B3.3 Event type ranking

Largest effects: football_match ×2.43, sports_run ×1.72, concert ×1.60. Weak / no clear lift: public_holiday ×1.03, road_closure ×0.74. Public holidays show ratio < 1 (demand drop), which is a real effect — just opposite sign.

### B3.4 Cancelled and unlisted events

(a) Cancelled events still show elevated demand in their nominal windows (mean ratio 1.41 on 26 hours) — some 'cancelled' rows may be mis-labeled or the crowd still came; we keep them out of features per status rules, but treat the label cautiously. (b) Unlisted spikes (≥3): Ayat 2025-07-20 (×2.25: possible local launch promo / estate move-in day); Megenagna 2025-09-10 (×1.79: possible transport disruption or unlisted stadium/hub surge); Lideta 2025-08-08 (×1.73: possible courthouse/office surge or unlisted conference).

## B4 — Operations & data quality

### B4.1 Operational variables vs demand

Correlations with trips: active_drivers=0.91, avg_wait_min=0.49, avg_fare_birr=0.00. Drivers co-move with demand (supply dispatched to busy zones) — not a clean cause. Wait time can rise when demand outstrips supply (positive) or fall when oversupplied. Fare mixes trip length/mix, not pure demand. None are known at forecast time for future hours, so they must not be model inputs (leakage / unavailable).

### B4.2 Gaps and outages

Shared platform outages: 42 hours across ≥10 early zones (1 contiguous block(s)). Ayat late launch from 2025-03-15 00:00:00+03:00 explains 1752 missing Ayat hours before go-live. Remaining ~682 sporadic zone-hour gaps look like random missing records. Treatment: do not invent zero-demand for outages/pre-launch; keep master at observed zone-hours; forecast features rely on lags/calendar rather than filling gaps with zeros.

### B4.3 Pay-period effect

Payday window (day-of-month ≤3 or ≥28): raw mean daily trips ratio 1.066 vs other days; after linear detrend + day-of-week removal, residual lift ≈ 412 trips/day (5.1% of mean day). Effect is large enough to keep `is_payday_window` as a calendar feature.
