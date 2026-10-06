"""Addis Ababa ride-demand forecast demo (Deliverable E).

    streamlit run app/app.py

The user picks a zone and a date in 1–14 November 2025. Weather and events are
looked up from the cleaned tables in app/assets/ (built by
scripts/build_app_assets.py); the bundled LightGBM model returns the hourly
forecast.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
TZ = "Africa/Addis_Ababa"
FIRST_DAY = dt.date(2025, 11, 1)
LAST_DAY = dt.date(2025, 11, 14)
TRIPS_PER_DRIVER_HOUR = 1.3
RAIN_NOTE_MM = 0.5

EVENT_LABELS = {
    "football_match": "football match",
    "concert": "concert",
    "conference": "conference",
    "exhibition": "exhibition",
    "sports_run": "sports run",
    "road_closure": "road closure",
    "public_holiday": "public holiday",
    "school_break": "school break",
}


# --------------------------------------------------------------------------- data


@st.cache_resource
def load_model() -> dict:
    return joblib.load(ASSETS / "model.joblib")


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    feats = pd.read_parquet(ASSETS / "forecast_features.parquet")
    feats["pickup_hour_eat"] = pd.to_datetime(feats["pickup_hour_eat"]).dt.tz_convert(TZ)

    weather = pd.read_csv(ASSETS / "weather_hourly_clean.csv")
    weather["timestamp_eat"] = pd.to_datetime(weather["timestamp_eat"], utc=True).dt.tz_convert(TZ)

    events = pd.read_csv(ASSETS / "events_clean.csv")
    for col in ["start_eat", "end_eat", "window_start", "window_end"]:
        events[col] = pd.to_datetime(events[col], utc=True).dt.tz_convert(TZ)

    return {
        "features": feats,
        "weather": weather,
        "events": events,
        "profile": pd.read_csv(ASSETS / "zone_profile.csv"),
        "fares": pd.read_csv(ASSETS / "zone_fares.csv").set_index("zone")["avg_fare_birr"],
    }


def day_bounds(day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day, tz=TZ)
    return start, start + pd.Timedelta(days=1)


def lookup_weather(weather: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    start, end = day_bounds(day)
    w = weather[(weather["timestamp_eat"] >= start) & (weather["timestamp_eat"] < end)].copy()
    w["hour"] = w["timestamp_eat"].dt.hour
    return w.set_index("hour").reindex(range(24))


def lookup_events(events: pd.DataFrame, zone: str, day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Confirmed events whose demand window touches the day: (this zone, other zones, not confirmed)."""
    start, end = day_bounds(day)
    touching = events[(events["window_end"] >= start) & (events["window_start"] < end)]
    confirmed = touching["status_clean"].str.contains("confirm", na=False)
    here = touching[confirmed & (touching["zone"] == zone)].sort_values("start_eat")
    elsewhere = touching[confirmed & (touching["zone"] != zone)].sort_values("start_eat")
    elsewhere = elsewhere.drop_duplicates(["event_name", "venue", "start_eat"])
    dropped = touching[~confirmed & (touching["zone"] == zone)]
    return here, elsewhere, dropped


def forecast_day(tables: dict, bundle: dict, zone: str, day: dt.date) -> pd.DataFrame:
    start, end = day_bounds(day)
    f = tables["features"]
    rows = f[(f["zone"] == zone) & (f["pickup_hour_eat"] >= start) & (f["pickup_hour_eat"] < end)]
    rows = rows.sort_values("pickup_hour_eat")

    pred = np.clip(bundle["model"].predict(rows[bundle["features"]]), 0, None)
    band = bundle["residual_bands"].get(zone, {"q10": 0.0, "q90": 0.0})
    fare = float(tables["fares"].get(zone, np.nan))

    out = pd.DataFrame(
        {
            "hour": rows["pickup_hour_eat"].dt.hour.to_numpy(),
            "forecast_trips": pred,
            "low_80": np.clip(pred + band["q10"], 0, None),
            "high_80": pred + band["q90"],
        }
    )
    out["drivers_needed"] = np.ceil(out["forecast_trips"] / TRIPS_PER_DRIVER_HOUR).astype(int)
    out["expected_fares_birr"] = out["forecast_trips"] * fare

    prof = tables["profile"]
    dow = pd.Timestamp(day).dayofweek
    typical = prof[(prof["zone"] == zone) & (prof["dow"] == dow)].set_index("hour")["typical_trips"]
    out["typical_trips"] = out["hour"].map(typical)
    return out


# --------------------------------------------------------------------------- text


def fmt_hour(h: int) -> str:
    return f"{h:02d}:00"


def fmt_span(start: pd.Timestamp, end: pd.Timestamp, day: dt.date) -> str:
    def part(t: pd.Timestamp) -> str:
        return t.strftime("%H:%M") if t.date() == day else t.strftime("%d %b %H:%M")

    return f"{part(start)}–{part(end)}"


def weather_sentence(w: pd.DataFrame) -> str:
    if w["temp_c"].isna().all():
        return "no weather forecast found for this date"
    lo, hi = w["temp_c"].min(), w["temp_c"].max()
    rain = w["rain_mm"].fillna(0)
    total = rain.sum()
    if total < RAIN_NOTE_MM:
        return f"dry, {lo:.0f}–{hi:.0f} °C"
    wet = rain[rain >= RAIN_NOTE_MM]
    peak_h = int(rain.idxmax())
    hours = ", ".join(fmt_hour(int(h)) for h in wet.index[:6]) + (" …" if len(wet) > 6 else "")
    return (
        f"rain {rain.max():.1f} mm at {fmt_hour(peak_h)} ({total:.1f} mm over the day; wet hours {hours}), "
        f"{lo:.0f}–{hi:.0f} °C"
    )


def event_phrase(ev: pd.Series, day: dt.date) -> str:
    label = EVENT_LABELS.get(ev["event_type"], str(ev["event_type"]).replace("_", " "))
    where = f" at {ev['venue']}" if isinstance(ev["venue"], str) and ev["venue"].strip() else ""
    crowd = f", ~{ev['attendance']:,.0f} expected" if pd.notna(ev["attendance"]) else ""
    return f"{label}{where} {fmt_span(ev['start_eat'], ev['end_eat'], day)}{crowd}"


# --------------------------------------------------------------------------- chart


def forecast_chart(fc: pd.DataFrame, here: pd.DataFrame, w: pd.DataFrame, day: dt.date, zone: str) -> go.Figure:
    fig = go.Figure()
    start, end = day_bounds(day)

    for _, ev in here.iterrows():
        s = max(ev["start_eat"], start)
        e = min(ev["end_eat"], end - pd.Timedelta(hours=1))
        x0 = (s - start).total_seconds() / 3600 - 0.5
        x1 = (e - start).total_seconds() / 3600 + 0.5
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="orange",
            opacity=0.18,
            line_width=0,
            annotation_text=EVENT_LABELS.get(ev["event_type"], ev["event_type"]),
            annotation_position="top left",
        )

    rain = w["rain_mm"].fillna(0)
    if rain.sum() >= RAIN_NOTE_MM:
        fig.add_bar(
            x=list(range(24)),
            y=rain.to_numpy(),
            name="rain (mm)",
            marker_color="rgba(70,130,180,0.35)",
            yaxis="y2",
        )

    fig.add_scatter(
        x=fc["hour"], y=fc["high_80"], mode="lines", line_width=0, showlegend=False, hoverinfo="skip"
    )
    fig.add_scatter(
        x=fc["hour"],
        y=fc["low_80"],
        mode="lines",
        line_width=0,
        fill="tonexty",
        fillcolor="rgba(31,119,180,0.15)",
        name="80% range",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=fc["hour"],
        y=fc["typical_trips"],
        mode="lines",
        name=f"typical {pd.Timestamp(day).day_name()} (last 8 weeks)",
        line=dict(color="gray", dash="dash"),
    )
    fig.add_scatter(
        x=fc["hour"],
        y=fc["forecast_trips"],
        mode="lines+markers",
        name="forecast trips",
        line=dict(color="#1f77b4", width=3),
    )

    fig.update_layout(
        title=f"{zone} — {day:%A %d %B %Y}",
        xaxis=dict(title="hour of day (EAT)", tickmode="linear", dtick=2, range=[-0.5, 23.5]),
        yaxis=dict(title="trips per hour", rangemode="tozero"),
        yaxis2=dict(title="rain (mm)", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=10),
        height=460,
        hovermode="x unified",
    )
    return fig


# --------------------------------------------------------------------------- page


st.set_page_config(page_title="Addis Ride Demand Forecast", page_icon="🚕", layout="wide")
st.title("Addis Ababa ride-demand forecast")
st.caption(
    "Pick a zone and a day between 1 and 14 November 2025. Weather and events are looked up "
    "automatically from the bundled cleaned tables."
)

try:
    bundle = load_model()
    tables = load_tables()
except FileNotFoundError as exc:
    st.error(
        f"App data is missing ({Path(exc.filename).name}). From the project root run "
        "`python scripts/build_app_assets.py`, then reload this page."
    )
    st.stop()

zones = sorted(tables["features"]["zone"].unique())
c1, c2 = st.columns(2)
zone = c1.selectbox("Zone", zones, index=zones.index("Bole") if "Bole" in zones else 0)
day = c2.date_input("Date", value=FIRST_DAY, format="DD/MM/YYYY")

if not isinstance(day, dt.date) or not (FIRST_DAY <= day <= LAST_DAY):
    st.warning(
        f"Forecasts are available for **{FIRST_DAY:%d %B} – {LAST_DAY:%d %B %Y}** only — the period "
        "covered by the weather forecast and event calendar. Please pick a date in that range."
    )
    st.stop()

fc = forecast_day(tables, bundle, zone, day)
if fc.empty:
    st.warning(f"No forecast rows found for {zone} on {day:%d %B %Y}.")
    st.stop()

weather = lookup_weather(tables["weather"], day)
here, elsewhere, dropped = lookup_events(tables["events"], zone, day)

lookups = [weather_sentence(weather)]
lookups += [event_phrase(ev, day) for _, ev in here.iterrows()] or ["no events in this zone"]
st.info("**Looked up:** " + "; ".join(lookups) + ".")
if not elsewhere.empty:
    st.caption(
        "Elsewhere in the city that day: "
        + "; ".join(f"{event_phrase(ev, day)} ({ev['zone']})" for _, ev in elsewhere.iterrows())
    )
if not dropped.empty:
    st.caption(
        "Ignored (not confirmed): "
        + "; ".join(f"{event_phrase(ev, day)} — {ev['status_clean']}" for _, ev in dropped.iterrows())
    )

peak = fc.loc[fc["forecast_trips"].idxmax()]
total_trips = fc["forecast_trips"].sum()
typical_total = fc["typical_trips"].sum()
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Trips forecast (day)",
    f"{total_trips:,.0f}",
    f"{(total_trips / typical_total - 1):+.0%} vs typical" if typical_total > 0 else None,
)
m2.metric("Peak hour", f"{fmt_hour(int(peak['hour']))}", f"{peak['forecast_trips']:.0f} trips", delta_color="off")
m3.metric("Drivers needed at peak", f"{int(peak['drivers_needed'])}", f"{fc['drivers_needed'].sum():,} driver-hours/day", delta_color="off")
m4.metric(
    "Expected gross fares",
    f"{fc['expected_fares_birr'].sum():,.0f} Birr",
    f"avg fare {tables['fares'].get(zone, np.nan):.0f} Birr/trip",
    delta_color="off",
)

st.plotly_chart(forecast_chart(fc, here, weather, day, zone), width="stretch")

table = pd.DataFrame(
    {
        "Hour": [fmt_hour(h) for h in fc["hour"]],
        "Forecast trips": fc["forecast_trips"].round(1),
        "80% range": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(fc["low_80"], fc["high_80"])],
        "Typical trips": fc["typical_trips"].round(1),
        "Drivers needed": fc["drivers_needed"],
        "Expected fares (Birr)": fc["expected_fares_birr"].round(0).astype(int),
        "Temp (°C)": weather["temp_c"].round(1).to_numpy(),
        "Rain (mm)": weather["rain_mm"].round(1).to_numpy(),
    }
)
st.subheader("Hourly forecast")
st.dataframe(table, hide_index=True, width="stretch", height=880)
st.download_button(
    "Download as CSV",
    table.to_csv(index=False).encode(),
    file_name=f"forecast_{zone.replace(' ', '_')}_{day:%Y-%m-%d}.csv",
    mime="text/csv",
)

with st.expander("How this works"):
    st.markdown(
        f"""
- **Model:** {bundle.get('label', 'LightGBM')}, trained on hourly trips up to {bundle['trained_through'][:10]}, using
  zone, calendar, demand two or more weeks earlier, weather and event features. Only history from at least
  14 days before each hour is used, so every day of the 1–14 Nov horizon is forecast the same way.
- **Accuracy:** on held-out 14-day validation fortnights the average error was
  **{bundle['holdout_mae']:.1f} trips per zone-hour**. The shaded band covers 80% of the errors seen in that
  zone on 18–31 Oct.
- **Drivers needed** = forecast trips ÷ {TRIPS_PER_DRIVER_HOUR} trips per driver-hour, rounded up.
- **Expected gross fares** = forecast trips × the zone's average fare in the training history.
- **Typical profile** = mean trips for the same zone, weekday and hour over the last 8 weeks of history.
"""
    )
