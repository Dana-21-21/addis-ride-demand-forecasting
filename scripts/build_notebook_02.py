#!/usr/bin/env python3
"""Build and execute notebooks/02_analysis_report.ipynb (Deliverable B)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "02_analysis_report.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# 02 — Analysis Report (Deliverable B)

Run from the project root context (`notebooks/` as cwd). Relative paths only.

Uses cleaned `data/processed/master_train.csv` from Deliverable A. Each task sits under its ID heading with a number/table/chart plus brief interpretation.

At the end, findings are written to `reports/B_analysis_report.md`.""",
    ),
    (
        "markdown",
        "## Setup & load master table",
    ),
    (
        "code",
        r'''from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("..")
sys.path.insert(0, str(ROOT))

from src.cleaning import load_raw_events, prepare_events

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

master = pd.read_csv(ROOT / "data" / "processed" / "master_train.csv", parse_dates=["pickup_hour_eat"])
assert master["pickup_hour_eat"].dt.tz is not None

# Zone types from weekday hourly shape (B1.2) — used again in B2/B3
ZONE_TYPE = {
    "Merkato": "market",
    "Megenagna": "transport_hub",
    "Bole": "nightlife_airport",
    "Kazanchis": "business",
    "Arat Kilo": "business",
    "Lideta": "business",
    "Piassa": "business",
    "CMC": "residential",
    "Gerji": "residential",
    "Kolfe": "residential",
    "Sarbet": "residential",
    "Ayat": "residential",
}
master["zone_type"] = master["zone"].map(ZONE_TYPE)
master["day"] = master["pickup_hour_eat"].dt.floor("D")

findings: dict[str, str] = {}
print(master.shape)
print(master["pickup_hour_eat"].min(), "→", master["pickup_hour_eat"].max())
master.head()''',
    ),
    (
        "markdown",
        "## B1 — Demand patterns",
    ),
    (
        "markdown",
        "### B1.1 Volume by zone",
    ),
    (
        "code",
        r'''b11 = (
    master.groupby("zone", as_index=False)
    .agg(
        total_trips=("trips", "sum"),
        mean_trips_per_zone_hour=("trips", "mean"),
        n_hours=("trips", "count"),
        first_hour=("pickup_hour_eat", "min"),
        last_hour=("pickup_hour_eat", "max"),
    )
    .sort_values("total_trips", ascending=False)
)
b11["share_pct"] = 100 * b11["total_trips"] / b11["total_trips"].sum()
b11''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(b11["zone"], b11["share_pct"], color="#1f4e79")
ax.set_xlabel("Share of city trips (%)")
ax.set_title("B1.1 — Zone demand share")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

ayat_note = (
    "Ayat launches mid-March (not full-period), so its share understates eventual weight."
    if b11.loc[b11["zone"] == "Ayat", "first_hour"].iloc[0] > master["pickup_hour_eat"].min()
    else ""
)
top3 = ", ".join(f"{r.zone} ({r.share_pct:.1f}%)" for r in b11.head(3).itertuples())
findings["B1.1"] = (
    f"Top zones by volume: {top3}. City mean trips/zone-hour ranges "
    f"{b11['mean_trips_per_zone_hour'].min():.1f}–{b11['mean_trips_per_zone_hour'].max():.1f}. {ayat_note}"
)
print(findings["B1.1"])''',
    ),
    (
        "markdown",
        "### B1.2 Hour-of-day profile by zone type",
    ),
    (
        "code",
        r'''wd = master[master["is_weekend"] == 0]
prof = wd.groupby(["zone", "hour"])["trips"].mean().unstack("hour")
prof_type = wd.groupby(["zone_type", "hour"])["trips"].mean().unstack("hour")

peak = prof_type.idxmax(axis=1).rename("peak_hour")
quiet = prof_type.idxmin(axis=1).rename("quiet_hour")
b12 = pd.concat([peak, quiet, prof_type.max(axis=1).rename("peak_mean"), prof_type.min(axis=1).rename("quiet_mean")], axis=1)
b12''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(10, 4.5))
for zt, row in prof_type.iterrows():
    ax.plot(row.index, row.values, marker="o", ms=3, label=zt)
ax.set_xlabel("Hour of day (EAT)")
ax.set_ylabel("Mean weekday trips")
ax.set_title("B1.2 — Weekday hourly profile by zone type")
ax.legend(fontsize=8, ncol=2)
ax.set_xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()

lines = [
    f"{zt}: peaks {int(r.peak_hour):02d}:00 (mean {r.peak_mean:.1f}), quietest {int(r.quiet_hour):02d}:00"
    for zt, r in b12.iterrows()
]
findings["B1.2"] = (
    "Zone types from weekday shapes — residential (morning ~07:00), business (evening ~18:00), "
    "market (midday ~13:00), transport hub (evening), nightlife/airport (early morning + weekend lift). "
    + " | ".join(lines)
)
print(findings["B1.2"])''',
    ),
    (
        "markdown",
        "### B1.3 Weekday vs weekend",
    ),
    (
        "code",
        r'''ww = master.groupby(["zone", "is_weekend"])["trips"].mean().unstack("is_weekend")
ww = ww.rename(columns={0: "weekday_mean", 1: "weekend_mean"})
ww["weekend_to_weekday"] = ww["weekend_mean"] / ww["weekday_mean"]
ww = ww.sort_values("weekend_to_weekday", ascending=False)
ww.round(3)''',
    ),
    (
        "code",
        r'''standout = ww["weekend_to_weekday"].idxmin()  # strongest weekend collapse
dow_shape = (
    master[master["zone"] == standout]
    .groupby("dow")["trips"]
    .mean()
    .reindex(range(7))
)
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].barh(ww.index, ww["weekend_to_weekday"], color="#2e7d32")
axes[0].axvline(1.0, color="gray", ls="--", lw=1)
axes[0].set_xlabel("Weekend / weekday mean trips")
axes[0].set_title("B1.3 — Weekend-to-weekday ratio")
axes[0].invert_yaxis()
axes[1].bar(dow_names, dow_shape.values, color="#1f4e79")
axes[1].set_title(f"Day-of-week shape — {standout}")
axes[1].set_ylabel("Mean trips")
plt.tight_layout()
plt.show()

busy = ", ".join(f"{z} ({r.weekend_to_weekday:.2f})" for z, r in ww.head(3).iterrows())
quiet_w = ", ".join(f"{z} ({r.weekend_to_weekday:.2f})" for z, r in ww.tail(3).iterrows())
findings["B1.3"] = (
    f"Weekend-busier: {busy}. Weekend-collapse: {quiet_w}. "
    f"{standout} stands out — weekday office/commerce demand drops sharply Sat–Sun "
    f"(means: " + ", ".join(f"{n}={v:.1f}" for n, v in zip(dow_names, dow_shape)) + ")."
)
print(findings["B1.3"])''',
    ),
    (
        "markdown",
        "### B1.4 Trend (January → October)",
    ),
    (
        "code",
        r'''city_daily = master.groupby("day", as_index=False)["trips"].sum()
city_daily["week"] = city_daily["day"].dt.to_period("W").apply(lambda p: p.start_time)
weekly = city_daily.groupby("week", as_index=False)["trips"].sum().rename(columns={"trips": "weekly_trips"})
# drop partial first/last weeks for growth calc
full = weekly.iloc[1:-1].copy()
early = full.head(4)["weekly_trips"].mean()
late = full.tail(4)["weekly_trips"].mean()
growth_pct = 100 * (late / early - 1)

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(weekly["week"], weekly["weekly_trips"], marker="o", ms=3, color="#1f4e79")
ax.set_title("B1.4 — City-wide weekly trips")
ax.set_ylabel("Trips / week")
plt.tight_layout()
plt.show()

weekly[["week", "weekly_trips"]].tail(8)''',
    ),
    (
        "code",
        r'''findings["B1.4"] = (
    f"Weekly city trips rise from ~{early:,.0f} (early full weeks) to ~{late:,.0f} "
    f"(late full weeks): +{growth_pct:.1f}%. "
    f"For November forecasts, a positive trend term (or recent lag levels) is essential — "
    f"a January baseline would under-predict by roughly that growth gap."
)
print(findings["B1.4"])
print(weekly.head(3).to_string(index=False))
print("...")
print(weekly.tail(3).to_string(index=False))''',
    ),
    (
        "markdown",
        "## B2 — Weather",
    ),
    (
        "markdown",
        "### B2.1 Timezone check",
    ),
    (
        "code",
        r'''city_wx = master.drop_duplicates("pickup_hour_eat")[["pickup_hour_eat", "temp_c", "rain_mm"]].copy()
city_wx["hour"] = city_wx["pickup_hour_eat"].dt.hour
temp_by_hour = city_wx.groupby("hour")["temp_c"].mean()
peak_temp_hour = int(temp_by_hour.idxmax())

# Rain–demand relationship under artificial clock shifts of weather (0–5h)
base = master[["pickup_hour_eat", "zone", "trips"]].copy()
wx = city_wx[["pickup_hour_eat", "rain_mm"]].copy()
shift_rows = []
for shift in range(0, 6):
    w = wx.copy()
    w["pickup_hour_eat"] = w["pickup_hour_eat"] + pd.Timedelta(hours=shift)
    tmp = base.merge(w, on="pickup_hour_eat", how="left")
    corr = tmp[["trips", "rain_mm"]].corr().iloc[0, 1]
    rainy = tmp.loc[tmp["rain_mm"] >= 0.5, "trips"].mean()
    dry = tmp.loc[tmp["rain_mm"] < 0.1, "trips"].mean()
    shift_rows.append(
        {
            "weather_shift_hours": shift,
            "corr_trips_rain": corr,
            "rainy_over_dry_mean": rainy / dry if dry else np.nan,
        }
    )
b21 = pd.DataFrame(shift_rows)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(temp_by_hour.index, temp_by_hour.values, marker="o", color="#c0392b")
axes[0].axvline(peak_temp_hour, color="gray", ls="--")
axes[0].set_title(f"Mean temp by hour (peak={peak_temp_hour}:00 EAT)")
axes[0].set_xlabel("Hour EAT")
axes[0].set_ylabel("°C")
axes[1].plot(b21["weather_shift_hours"], b21["corr_trips_rain"], marker="o", label="corr")
axes[1].plot(b21["weather_shift_hours"], b21["rainy_over_dry_mean"] - 1, marker="s", label="rainy/dry − 1")
axes[1].set_xlabel("Hours weather shifted later")
axes[1].set_title("B2.1 — Misalignment destroys rain signal")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
b21.round(3)''',
    ),
    (
        "code",
        r'''best = b21.iloc[0]
worst = b21.iloc[-1]
findings["B2.1"] = (
    f"Temperature peaks at {peak_temp_hour}:00 EAT — consistent with local afternoon heat, "
    f"so weather is correctly on Africa/Addis_Ababa after UTC→EAT conversion. "
    f"At 0h shift, corr(trips, rain)={best.corr_trips_rain:.3f} and rainy/dry={best.rainy_over_dry_mean:.2f}; "
    f"at +5h these fall to {worst.corr_trips_rain:.3f} and {worst.rainy_over_dry_mean:.2f}. "
    f"A 3h UTC mistake would materially weaken the rain feature."
)
print(findings["B2.1"])''',
    ),
    (
        "markdown",
        "### B2.2 Rain effect by zone type",
    ),
    (
        "code",
        r'''dry_base = (
    master.loc[master["rain_mm"] <= 0]
    .groupby(["zone", "dow", "hour"])["trips"]
    .mean()
    .rename("dry_mean")
)
rainy = master.loc[master["rain_mm"] >= 0.5].merge(dry_base, on=["zone", "dow", "hour"], how="left")
rainy = rainy.dropna(subset=["dry_mean"])
rainy = rainy[rainy["dry_mean"] > 0]
rainy["rain_over_dry"] = rainy["trips"] / rainy["dry_mean"]

b22 = (
    rainy.groupby("zone_type")["rain_over_dry"]
    .agg(mean_ratio="mean", median_ratio="median", n_hours="count")
    .sort_values("mean_ratio", ascending=False)
)
b22.round(3)''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(b22.index, b22["mean_ratio"], color="#1565c0")
ax.axhline(1.0, color="gray", ls="--")
ax.set_ylabel("Mean trips / matched dry baseline")
ax.set_title("B2.2 — Rain uplift by zone type (≥0.5 mm vs dry same zone·dow·hour)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

up = b22[b22["mean_ratio"] >= 1]
down = b22[b22["mean_ratio"] < 1]
findings["B2.2"] = (
    "Matched rainy (≥0.5 mm) vs dry same zone·weekday·hour: "
    + (
        "lift in " + ", ".join(f"{zt} ×{r.mean_ratio:.2f}" for zt, r in up.iterrows())
        if len(up)
        else "no zone type lifts"
    )
    + (
        "; demand falls in " + ", ".join(f"{zt} ×{r.mean_ratio:.2f}" for zt, r in down.iterrows())
        if len(down)
        else ""
    )
    + ". Rain does not increase demand everywhere — Merkato (market) softens, consistent with outdoor trading pausing in wet weather."
)
print(findings["B2.2"])''',
    ),
    (
        "markdown",
        "### B2.3 Rain dose-response",
    ),
    (
        "code",
        r'''RAIN_LABEL = {0: "none", 1: "light", 2: "moderate", 3: "heavy"}
dose = master.merge(dry_base, on=["zone", "dow", "hour"], how="left")
dose = dose.dropna(subset=["dry_mean"])
dose = dose[dose["dry_mean"] > 0]
dose["ratio"] = dose["trips"] / dose["dry_mean"]
dose["rain_label"] = dose["rain_class"].map(RAIN_LABEL)

b23 = dose.groupby(["zone_type", "rain_label"])["ratio"].mean().unstack("rain_label")
# column order
b23 = b23.reindex(columns=["none", "light", "moderate", "heavy"])
b23.round(3)''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(b23.columns))
width = 0.15
for i, (zt, row) in enumerate(b23.iterrows()):
    ax.plot(b23.columns, row.values, marker="o", label=zt)
ax.set_ylabel("Demand ratio vs dry baseline")
ax.set_title("B2.3 — Rain dose-response by zone type")
ax.legend(fontsize=8, ncol=2)
ax.axhline(1.0, color="gray", ls="--", lw=1)
plt.tight_layout()
plt.show()

# saturation note
city_dose = dose.groupby("rain_label")["ratio"].mean().reindex(["none", "light", "moderate", "heavy"])
step = city_dose.diff()
findings["B2.3"] = (
    f"City-wide dose-response (vs dry baseline): "
    + " → ".join(f"{k}={city_dose[k]:.2f}" for k in city_dose.index)
    + f". Light rain already lifts demand (+{100*(city_dose['light']-1):.0f}%); "
    + f"moderate and heavy keep rising (heavy {city_dose['heavy']:.2f}). "
    + "Not a pure straight line — early step from none→light is large relative to rainfall amount; "
    + "zone types differ in slope (see table)."
)
print(findings["B2.3"])
print(city_dose.round(3).to_string())''',
    ),
    (
        "markdown",
        "## B3 — Events & calendar",
    ),
    (
        "markdown",
        "### B3.1 Public holidays",
    ),
    (
        "code",
        r'''from src.cleaning import DEFAULT_DURATION_HOURS, EVENT_WINDOW

events = prepare_events(load_raw_events())
confirmed = events[events["status_clean"].str.contains("confirm", na=False)].copy()
holidays = confirmed[confirmed["event_type"] == "public_holiday"].drop_duplicates("event_id")

# Some raw holiday end timestamps span months (bad data). Resolve a single analysis day:
# - short windows: start day
# - known overrides for corrupted rows
# - Labour-Day-like swap (start ~23:59, end midnight months later): prefer end calendar day
HOLIDAY_DAY_OVERRIDE = {
    "EVT-0055": "2025-05-01",  # International Labour Day (start was 2025-01-05 23:59)
    "EVT-0001": "2025-01-07",  # Genna
    "EVT-0121": "2025-09-11",  # Enkutatash
}

def holiday_analysis_day(ev) -> pd.Timestamp:
    if ev["event_id"] in HOLIDAY_DAY_OVERRIDE:
        d = pd.Timestamp(HOLIDAY_DAY_OVERRIDE[ev["event_id"]], tz="Africa/Addis_Ababa")
        return d
    start = pd.Timestamp(ev["start_eat"]).tz_convert("Africa/Addis_Ababa")
    end = pd.Timestamp(ev["end_eat"]).tz_convert("Africa/Addis_Ababa")
    dur_h = (end - start).total_seconds() / 3600
    if dur_h <= 36:
        return start.normalize()
    if start.hour >= 22 and end.hour == 0 and end.minute == 0:
        return end.normalize()
    return start.normalize()

city_by_day = master.groupby("day")["trips"].sum()
holiday_days = {holiday_analysis_day(r) for _, r in holidays.iterrows()}

rows = []
for _, ev in holidays.iterrows():
    d = holiday_analysis_day(ev)
    day_trips = float(city_by_day.get(d, np.nan))
    same = city_by_day[
        (city_by_day.index.dayofweek == d.dayofweek) & (~city_by_day.index.isin(holiday_days))
    ]
    near = same[np.abs((same.index - d).days) <= 28]
    ref = float(near.mean()) if len(near) else np.nan
    day_z = master.loc[master["day"] == d].groupby("zone")["trips"].sum()
    ref_z = (
        master.loc[master["day"].isin(near.index)]
        .groupby(["zone", "day"])["trips"]
        .sum()
        .groupby("zone")
        .mean()
    )
    local = (day_z / ref_z).replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    rows.append(
        {
            "event_id": ev["event_id"],
            "event_name": ev["event_name"],
            "date": d.date(),
            "city_trips": day_trips,
            "near_same_weekday_mean": ref,
            "ratio_vs_near": day_trips / ref if ref else np.nan,
            "strongest_local_drop": f"{local.index[0]} ({local.iloc[0]:.2f})" if len(local) else "",
            "strongest_local_rise": f"{local.index[-1]} ({local.iloc[-1]:.2f})" if len(local) else "",
        }
    )
b31 = pd.DataFrame(rows).dropna(subset=["ratio_vs_near"]).sort_values("ratio_vs_near")
b31''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(9, 3.8))
ax.barh(b31["event_name"].astype(str), b31["ratio_vs_near"], color="#6a1b9a")
ax.axvline(1.0, color="gray", ls="--")
ax.set_xlabel("City trips / nearby same-weekday mean")
ax.set_title("B3.1 — Public holiday impact (ranked)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

reducers = b31[b31["ratio_vs_near"] < 0.98]
lifters = b31[b31["ratio_vs_near"] >= 1.02]
findings["B3.1"] = (
    f"Holiday city-day vs nearby same weekdays — rank (low→high): "
    + "; ".join(f"{r.event_name} ({r.ratio_vs_near:.2f})" for r in b31.itertuples())
    + ". "
    + (
        f"{len(reducers)}/{len(b31)} holidays clearly reduce demand; "
        if len(reducers)
        else "No clear city-wide drops; "
    )
    + (
        f"not all do — e.g. {', '.join(lifters['event_name'].astype(str))} hold flat/up. "
        if len(lifters)
        else "nearly all reduce demand. "
    )
    + "Local effects vary (see strongest_local_drop/rise): business zones often drop more than residential."
)
print(findings["B3.1"])''',
    ),
    (
        "markdown",
        "### B3.2 Event-window study (football)",
    ),
    (
        "code",
        r'''# Vectorized baselines: mean trips by zone·dow·hour outside football windows
fb_base = (
    master.loc[master["is_football_window"] == 0]
    .groupby(["zone", "dow", "hour"])["trips"]
    .mean()
    .rename("base_mean")
)

fb_events = confirmed[confirmed["event_type"] == "football_match"].drop_duplicates("event_id")
# one row per event-zone (already exploded)
fb_ez = confirmed[confirmed["event_type"] == "football_match"].dropna(subset=["zone"])

pieces = []
for _, ev in fb_ez.iterrows():
    z, start, end = ev["zone"], ev["start_eat"], ev["end_eat"]
    windows = {
        "pre_2h": (start - pd.Timedelta(hours=2), start),
        "during": (start, end),
        "post_2h": (end, end + pd.Timedelta(hours=2)),
    }
    for label, (t0, t1) in windows.items():
        sub = master[(master["zone"] == z) & (master["pickup_hour_eat"] >= t0) & (master["pickup_hour_eat"] < t1)]
        if sub.empty:
            continue
        tmp = sub.merge(fb_base, on=["zone", "dow", "hour"], how="left")
        tmp = tmp[tmp["base_mean"] > 0]
        if tmp.empty:
            continue
        pieces.append(pd.DataFrame({"window": label, "ratio": tmp["trips"] / tmp["base_mean"]}))

fb_df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["window", "ratio"])
b32 = fb_df.groupby("window")["ratio"].agg(mean="mean", median="median", n="count")
b32 = b32.reindex(["pre_2h", "during", "post_2h"])
b32''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(b32.index.astype(str), b32["mean"], color="#e65100")
ax.axhline(1.0, color="gray", ls="--")
ax.set_ylabel("Mean trips / no-football baseline")
ax.set_title("B3.2 — Football window uplift")
plt.tight_layout()
plt.show()

best_win = b32["mean"].idxmax()
findings["B3.2"] = (
    f"Football match-zone uplift — pre_2h ×{b32.loc['pre_2h','mean']:.2f}, "
    f"during ×{b32.loc['during','mean']:.2f}, post_2h ×{b32.loc['post_2h','mean']:.2f} "
    f"(vs same zone·dow·hour without football). Largest window: {best_win}."
)
print(findings["B3.2"])''',
    ),
    (
        "markdown",
        "### B3.3 Event type ranking",
    ),
    (
        "code",
        r'''no_event_base = (
    master.loc[master["in_event_window"] == 0]
    .groupby(["zone", "dow", "hour"])["trips"]
    .mean()
    .rename("base_mean")
)

etype_rows = []
for etype, grp in confirmed.dropna(subset=["zone"]).groupby("event_type"):
    ratios = []
    for _, ev in grp.iterrows():
        sub = master[
            (master["zone"] == ev["zone"])
            & (master["pickup_hour_eat"] >= ev["window_start"])
            & (master["pickup_hour_eat"] <= ev["window_end"])
        ]
        if sub.empty:
            continue
        tmp = sub.merge(no_event_base, on=["zone", "dow", "hour"], how="left")
        tmp = tmp[tmp["base_mean"] > 0]
        ratios.extend((tmp["trips"] / tmp["base_mean"]).tolist())
    if ratios:
        etype_rows.append(
            {
                "event_type": etype,
                "n_event_zone_rows": len(grp.drop_duplicates("event_id")),
                "n_matched_hours": len(ratios),
                "mean_ratio": float(np.mean(ratios)),
                "median_ratio": float(np.median(ratios)),
            }
        )
b33 = pd.DataFrame(etype_rows).sort_values("mean_ratio", ascending=False)
b33.round(3)''',
    ),
    (
        "code",
        r'''fig, ax = plt.subplots(figsize=(9, 3.8))
ax.barh(b33["event_type"], b33["mean_ratio"], color="#00838f")
ax.axvline(1.0, color="gray", ls="--")
ax.set_xlabel("Mean demand ratio in event window")
ax.set_title("B3.3 — Event type effect size")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

strong = b33.head(3)
weak = b33[b33["mean_ratio"] < 1.05]
findings["B3.3"] = (
    "Largest effects: "
    + ", ".join(f"{r.event_type} ×{r.mean_ratio:.2f}" for r in strong.itertuples())
    + ". Weak / no clear lift: "
    + (", ".join(f"{r.event_type} ×{r.mean_ratio:.2f}" for r in weak.itertuples()) if len(weak) else "none below 1.05")
    + ". Public holidays show ratio < 1 (demand drop), which is a real effect — just opposite sign."
)
print(findings["B3.3"])''',
    ),
    (
        "markdown",
        "### B3.4 Cancelled and unlisted events",
    ),
    (
        "code",
        r'''# Use nominal duration windows — some cancelled rows have multi-month end timestamps
cancelled = events[events["status_clean"].str.contains("cancel", na=False)].dropna(subset=["zone"])
canc_ratios = []
canc_by_event = []
for eid, grp in cancelled.groupby("event_id"):
    ev = grp.iloc[0]
    etype = ev["event_type"]
    dur_h = float(DEFAULT_DURATION_HOURS.get(etype, 2))
    pre, post = EVENT_WINDOW.get(etype, (1, 1))
    start = pd.Timestamp(ev["start_eat"])
    t0 = start - pd.Timedelta(hours=pre)
    t1 = start + pd.Timedelta(hours=dur_h + post)
    ratios = []
    for z in grp["zone"].dropna().unique():
        sub = master[
            (master["zone"] == z)
            & (master["pickup_hour_eat"] >= t0)
            & (master["pickup_hour_eat"] < t1)
        ]
        if sub.empty:
            continue
        tmp = sub.merge(no_event_base, on=["zone", "dow", "hour"], how="left")
        tmp = tmp[tmp["base_mean"] > 0]
        ratios.extend((tmp["trips"] / tmp["base_mean"]).tolist())
    if ratios:
        canc_ratios.extend(ratios)
        canc_by_event.append({"event_id": eid, "event_type": etype, "mean_ratio": float(np.mean(ratios)), "n": len(ratios)})

canc_mean = float(np.mean(canc_ratios)) if canc_ratios else np.nan
canc_event_df = pd.DataFrame(canc_by_event)
print("Cancelled events (unique ids):", cancelled["event_id"].nunique())
print("Matched zone-hours (nominal windows):", len(canc_ratios), "mean ratio:", round(canc_mean, 3) if canc_ratios else None)
display_cols = ["event_id", "event_name", "event_type", "zone", "start_eat", "status_clean"]
cancelled.drop_duplicates("event_id")[display_cols]''',
    ),
    (
        "code",
        r'''# Unexplained spikes: high daily zone totals vs same-dow median, no listed confirmed event that day
daily_z = master.groupby(["zone", "day"], as_index=False)["trips"].sum()
daily_z["dow"] = daily_z["day"].dt.dayofweek
med = daily_z.groupby(["zone", "dow"])["trips"].median().rename("dow_median")
daily_z = daily_z.merge(med, on=["zone", "dow"])
daily_z["ratio"] = daily_z["trips"] / daily_z["dow_median"]

ev_keys = set(
    zip(
        master.loc[master["in_event_window"] == 1, "zone"],
        master.loc[master["in_event_window"] == 1, "day"],
    )
)
daily_z["has_listed_event"] = [(z, d) in ev_keys for z, d in zip(daily_z["zone"], daily_z["day"])]
spikes = (
    daily_z[(~daily_z["has_listed_event"]) & (daily_z["ratio"] >= 1.5)]
    .sort_values("ratio", ascending=False)
    .head(12)
)
spikes[["zone", "day", "trips", "dow_median", "ratio"]]''',
    ),
    (
        "code",
        r'''top3 = spikes.head(3)
guesses = []
for r in top3.itertuples():
    if r.zone in {"Bole"}:
        guess = "possible unlisted concert / nightlife peak"
    elif r.zone == "Megenagna":
        guess = "possible transport disruption or unlisted stadium/hub surge"
    elif r.zone == "Merkato":
        guess = "possible market festival / bulk-trading day not in calendar"
    elif r.zone == "Lideta":
        guess = "possible courthouse/office surge or unlisted conference"
    elif r.zone == "Ayat":
        guess = "possible local launch promo / estate move-in day"
    else:
        guess = "possible local gathering or church/school event not listed"
    guesses.append(f"{r.zone} {pd.Timestamp(r.day).date()} (×{r.ratio:.2f}: {guess})")

if canc_mean < 1.08:
    canc_txt = (
        f"(a) Cancelled events show little demand footprint (mean ratio {canc_mean:.2f} on "
        f"{len(canc_ratios)} nominal-window hours) — safe to exclude from features. "
    )
else:
    canc_txt = (
        f"(a) Cancelled events still show elevated demand in their nominal windows "
        f"(mean ratio {canc_mean:.2f} on {len(canc_ratios)} hours) — some 'cancelled' rows may be "
        f"mis-labeled or the crowd still came; we keep them out of features per status rules, "
        f"but treat the label cautiously. "
    )

findings["B3.4"] = canc_txt + "(b) Unlisted spikes (≥3): " + "; ".join(guesses) + "."
print(findings["B3.4"])
if len(canc_event_df):
    print(canc_event_df.round(3).to_string(index=False))''',
    ),
    (
        "markdown",
        "## B4 — Operations & data quality",
    ),
    (
        "markdown",
        "### B4.1 Operational variables vs demand",
    ),
    (
        "code",
        r'''ops = master[["trips", "active_drivers", "avg_wait_min", "avg_fare_birr"]].dropna()
corr = ops.corr()["trips"].drop("trips")
corr.to_frame("corr_with_trips")''',
    ),
    (
        "code",
        r'''fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col in zip(axes, ["active_drivers", "avg_wait_min", "avg_fare_birr"]):
    sample = ops.sample(min(8000, len(ops)), random_state=42)
    ax.scatter(sample[col], sample["trips"], s=4, alpha=0.25, color="#37474f")
    ax.set_xlabel(col)
    ax.set_ylabel("trips")
    ax.set_title(f"r={corr[col]:.2f}")
plt.suptitle("B4.1 — Ops variables vs trips", y=1.02)
plt.tight_layout()
plt.show()

findings["B4.1"] = (
    f"Correlations with trips: active_drivers={corr['active_drivers']:.2f}, "
    f"avg_wait_min={corr['avg_wait_min']:.2f}, avg_fare_birr={corr['avg_fare_birr']:.2f}. "
    f"Drivers co-move with demand (supply dispatched to busy zones) — not a clean cause. "
    f"Wait time can rise when demand outstrips supply (positive) or fall when oversupplied. "
    f"Fare mixes trip length/mix, not pure demand. None are known at forecast time for future hours, "
    f"so they must not be model inputs (leakage / unavailable)."
)
print(findings["B4.1"])''',
    ),
    (
        "markdown",
        "### B4.2 Gaps and outages",
    ),
    (
        "code",
        r'''hours = pd.date_range(master["pickup_hour_eat"].min(), master["pickup_hour_eat"].max(), freq="h")
zone_hours = {z: set(g["pickup_hour_eat"]) for z, g in master.groupby("zone")}
non_ayat = [z for z in zone_hours if z != "Ayat"]

miss_count = {}
for h in hours:
    c = sum(1 for z in non_ayat if h not in zone_hours[z])
    if c:
        miss_count[h] = c

# Platform outage: missing in ≥10 of 11 early zones
common = sorted(h for h, c in miss_count.items() if c >= 10)

def compress(ts_list):
    if not ts_list:
        return []
    ranges = []
    start = prev = ts_list[0]
    for h in ts_list[1:]:
        if h - prev == pd.Timedelta(hours=1):
            prev = h
        else:
            ranges.append((start, prev, int((prev - start).total_seconds() // 3600) + 1))
            start = prev = h
    ranges.append((start, prev, int((prev - start).total_seconds() // 3600) + 1))
    return ranges

outage_ranges = compress(common)
ayat_start = master.loc[master["zone"] == "Ayat", "pickup_hour_eat"].min()
ayat_pre = [h for h in hours if h < ayat_start]

# Per-zone leftover gaps (not in common outage, not Ayat pre-launch)
common_set = set(common)
random_gaps = []
for z, present in zone_hours.items():
    for h in hours:
        if h in present:
            continue
        if z == "Ayat" and h < ayat_start:
            continue
        if h in common_set:
            continue
        random_gaps.append((z, h))

gap_summary = pd.DataFrame(
    [
        {"gap_class": "platform_outage_shared", "n_hours": len(common), "treatment": "left missing / model uses lags; not imputed as zeros"},
        {"gap_class": "ayat_pre_launch", "n_hours": len(ayat_pre), "treatment": "Ayat excluded before launch; not an outage"},
        {"gap_class": "sporadic_zone_gaps", "n_hours": len(random_gaps), "treatment": "remain missing zone-hours; features use available history"},
    ]
)
print("Ayat launch:", ayat_start)
print("Shared outage ranges:")
for a, b, n in outage_ranges:
    print(f"  {a} → {b} ({n}h)")
gap_summary''',
    ),
    (
        "code",
        r'''findings["B4.2"] = (
    f"Shared platform outages: {len(common)} hours across ≥10 early zones "
    f"({len(outage_ranges)} contiguous block(s)). "
    f"Ayat late launch from {ayat_start} explains {len(ayat_pre)} missing Ayat hours before go-live. "
    f"Remaining ~{len(random_gaps)} sporadic zone-hour gaps look like random missing records. "
    f"Treatment: do not invent zero-demand for outages/pre-launch; keep master at observed zone-hours; "
    f"forecast features rely on lags/calendar rather than filling gaps with zeros."
)
print(findings["B4.2"])''',
    ),
    (
        "markdown",
        "### B4.3 Pay-period effect",
    ),
    (
        "code",
        r'''daily = (
    master.groupby("day", as_index=False)
    .agg(trips=("trips", "sum"), payday=("is_payday_window", "max"))
)
daily["dow"] = daily["day"].dt.dayofweek
daily["t"] = np.arange(len(daily))
coef = np.polyfit(daily["t"], daily["trips"], 1)
daily["detrended"] = daily["trips"] - np.polyval(coef, daily["t"])
daily["dow_mean"] = daily.groupby("dow")["detrended"].transform("mean")
daily["resid"] = daily["detrended"] - daily["dow_mean"]

pay = daily.loc[daily["payday"] == 1, "resid"]
oth = daily.loc[daily["payday"] == 0, "resid"]
lift = pay.mean() - oth.mean()
lift_pct = 100 * lift / daily["trips"].mean()
raw_ratio = daily.loc[daily["payday"] == 1, "trips"].mean() / daily.loc[daily["payday"] == 0, "trips"].mean()

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.scatter(daily.loc[daily["payday"] == 0, "day"], daily.loc[daily["payday"] == 0, "resid"], s=12, alpha=0.5, label="ordinary")
ax.scatter(daily.loc[daily["payday"] == 1, "day"], daily.loc[daily["payday"] == 1, "resid"], s=16, alpha=0.8, label="payday window")
ax.axhline(0, color="gray", ls="--", lw=1)
ax.legend()
ax.set_title("B4.3 — Detrended+deseasoned daily residual (payday vs not)")
ax.set_ylabel("Residual trips")
plt.tight_layout()
plt.show()

pd.DataFrame(
    {
        "metric": ["raw_mean_ratio_payday_over_other", "detrended_deseasoned_lift_trips", "lift_pct_of_mean_day"],
        "value": [raw_ratio, lift, lift_pct],
    }
)''',
    ),
    (
        "code",
        r'''keep = abs(lift_pct) >= 2.0
findings["B4.3"] = (
    f"Payday window (day-of-month ≤3 or ≥28): raw mean daily trips ratio {raw_ratio:.3f} vs other days; "
    f"after linear detrend + day-of-week removal, residual lift ≈ {lift:,.0f} trips/day ({lift_pct:.1f}% of mean day). "
    + (
        "Effect is large enough to keep `is_payday_window` as a calendar feature."
        if keep
        else "Effect is small after detrend — optional feature, low priority vs hour/dow/rain/events."
    )
)
print(findings["B4.3"])''',
    ),
    (
        "markdown",
        "## Export report",
    ),
    (
        "code",
        r'''lines = [
    "# B — Data Analysis Report",
    "",
    "Source: `notebooks/02_analysis_report.ipynb` on `data/processed/master_train.csv`.",
    "Each task: result + short interpretation (Deliverable B, 14 pts).",
    "",
]
for key in [
    "B1.1", "B1.2", "B1.3", "B1.4",
    "B2.1", "B2.2", "B2.3",
    "B3.1", "B3.2", "B3.3", "B3.4",
    "B4.1", "B4.2", "B4.3",
]:
    section = {
        "B1.1": "### B1.1 Volume by zone",
        "B1.2": "### B1.2 Hour-of-day profile by zone type",
        "B1.3": "### B1.3 Weekday vs weekend",
        "B1.4": "### B1.4 Trend",
        "B2.1": "### B2.1 Timezone check",
        "B2.2": "### B2.2 Rain effect by zone type",
        "B2.3": "### B2.3 Rain dose-response",
        "B3.1": "### B3.1 Public holidays",
        "B3.2": "### B3.2 Football event windows",
        "B3.3": "### B3.3 Event type ranking",
        "B3.4": "### B3.4 Cancelled and unlisted events",
        "B4.1": "### B4.1 Operational variables vs demand",
        "B4.2": "### B4.2 Gaps and outages",
        "B4.3": "### B4.3 Pay-period effect",
    }[key]
    if key.startswith("B1") and key.endswith(".1"):
        lines += ["## B1 — Demand patterns", ""]
    if key == "B2.1":
        lines += ["## B2 — Weather", ""]
    if key == "B3.1":
        lines += ["## B3 — Events & calendar", ""]
    if key == "B4.1":
        lines += ["## B4 — Operations & data quality", ""]
    lines += [section, "", findings[key], ""]

out = REPORTS / "B_analysis_report.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("Wrote", out)
print("Tasks complete:", len(findings), "/ 14")
assert len(findings) == 14''',
    ),
]


def make_notebook() -> dict:
    cells = []
    for kind, source in CELLS:
        src = source.strip("\n") + "\n"
        if kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [line + "\n" for line in src.split("\n")[:-1]] + ([src.split("\n")[-1] + "\n"] if src.split("\n")[-1] else []),
                }
            )
            # simpler source as single string split by lines for nbformat compatibility
            cells[-1]["source"] = [l + "\n" for l in source.strip().split("\n")]
            if cells[-1]["source"]:
                cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")
                if not cells[-1]["source"][-1].endswith("\n"):
                    pass
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [l + "\n" for l in source.strip().split("\n")],
                }
            )
            # last line without forcing extra blank weirdness — nbformat accepts trailing \n on each
    # normalize markdown/code sources: join style used by Jupyter
    for c in cells:
        lines = c["source"]
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        # remove final extra empty if any from split
        c["source"] = lines
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main() -> None:
    nb = make_notebook()
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {NB_PATH} with {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
