"""Feature engineering for the master tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SPEC = [
    # calendar (≥3)
    {
        "name": "hour",
        "formula": "hour of pickup_hour_eat",
        "source": "pickup_hour_eat",
        "why": "Demand has strong daily seasonality",
        "known_at_forecast_time": "yes",
        "group": "calendar",
    },
    {
        "name": "dow",
        "formula": "day of week 0=Mon … 6=Sun",
        "source": "pickup_hour_eat",
        "why": "Weekday vs weekend patterns differ by zone",
        "known_at_forecast_time": "yes",
        "group": "calendar",
    },
    {
        "name": "is_weekend",
        "formula": "1 if dow in {5,6}",
        "source": "dow",
        "why": "Weekend demand shifts (markets, leisure)",
        "known_at_forecast_time": "yes",
        "group": "calendar",
    },
    {
        "name": "is_payday_window",
        "formula": "1 if day_of_month <=3 or >=28",
        "source": "pickup_hour_eat",
        "why": "Pay-period spending may lift trips",
        "known_at_forecast_time": "yes",
        "group": "calendar",
    },
    {
        "name": "is_public_holiday",
        "formula": "1 if hour overlaps a confirmed public_holiday window",
        "source": "events",
        "why": "Holidays change city-wide demand",
        "known_at_forecast_time": "yes",
        "group": "calendar",
    },
    # weather (≥2)
    {
        "name": "rain_mm",
        "formula": "cleaned rain in the hour",
        "source": "weather.rain_mm",
        "why": "Rain often increases ride requests",
        "known_at_forecast_time": "yes",
        "group": "weather",
    },
    {
        "name": "rain_last_3h",
        "formula": "rain_mm + lag1 + lag2 (by city hour)",
        "source": "weather.rain_mm",
        "why": "Wet conditions persist beyond a single hour",
        "known_at_forecast_time": "yes",
        "group": "weather",
    },
    {
        "name": "rain_class",
        "formula": "0 none / 1 light / 2 moderate / 3 heavy",
        "source": "rain_mm",
        "why": "Non-linear rain response",
        "known_at_forecast_time": "yes",
        "group": "weather",
    },
    # events (≥3)
    {
        "name": "in_event_window",
        "formula": "1 if any confirmed event window covers the hour",
        "source": "events",
        "why": "Events create local demand spikes",
        "known_at_forecast_time": "yes",
        "group": "events",
    },
    {
        "name": "is_football_window",
        "formula": "1 if football_match window (±2h) covers the hour",
        "source": "events",
        "why": "Matches are among the largest local shocks",
        "known_at_forecast_time": "yes",
        "group": "events",
    },
    {
        "name": "hours_to_event_start",
        "formula": "hours until nearest matching event start (NaN if none)",
        "source": "events.start_eat",
        "why": "Pre-event arrival traffic differs from during/after",
        "known_at_forecast_time": "yes",
        "group": "events",
    },
    {
        "name": "event_attendance",
        "formula": "max expected attendance among matching events",
        "source": "events.expected_attendance",
        "why": "Bigger crowds → bigger demand lift",
        "known_at_forecast_time": "yes",
        "group": "events",
    },
    # lag / trend (≥1)
    {
        "name": "trips_lag_168",
        "formula": "same zone trips 168 hours earlier (1 week)",
        "source": "trips",
        "why": "Weekly seasonality baseline per zone-hour",
        "known_at_forecast_time": "yes",
        "group": "lag",
    },
    {
        "name": "trend_day",
        "formula": "days since 2025-01-01",
        "source": "pickup_hour_eat",
        "why": "Captures gradual growth through the year",
        "known_at_forecast_time": "yes",
        "group": "lag",
    },
]


def _rain_class(x: float) -> int:
    if pd.isna(x) or x <= 0:
        return 0
    if x < 1:
        return 1
    if x < 5:
        return 2
    return 3


def add_features(df: pd.DataFrame, train_trips_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add engineered features. Lag uses train history (+ current df for train)."""
    out = df.copy()
    ts = out["pickup_hour_eat"]
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["day_of_month"] = ts.dt.day
    out["is_payday_window"] = ((out["day_of_month"] <= 3) | (out["day_of_month"] >= 28)).astype(int)
    out["trend_day"] = (ts.dt.tz_localize(None) - pd.Timestamp("2025-01-01")).dt.days

    # rain rolling on city timeline
    weather_hour = (
        out[["pickup_hour_eat", "rain_mm"]]
        .drop_duplicates("pickup_hour_eat")
        .sort_values("pickup_hour_eat")
    )
    weather_hour["rain_last_3h"] = (
        weather_hour["rain_mm"].fillna(0)
        + weather_hour["rain_mm"].fillna(0).shift(1)
        + weather_hour["rain_mm"].fillna(0).shift(2)
    )
    out = out.merge(
        weather_hour[["pickup_hour_eat", "rain_last_3h"]],
        on="pickup_hour_eat",
        how="left",
    )
    out["rain_class"] = out["rain_mm"].map(_rain_class)

    # weekly lag trips
    hist = train_trips_history
    if hist is None:
        hist = out[["zone", "pickup_hour_eat", "trips"]].copy()
    else:
        hist = hist[["zone", "pickup_hour_eat", "trips"]].copy()
        if "trips" in out.columns:
            hist = pd.concat([hist, out[["zone", "pickup_hour_eat", "trips"]]], ignore_index=True)
            hist = hist.drop_duplicates(["zone", "pickup_hour_eat"], keep="last")

    lag = hist.copy()
    lag["pickup_hour_eat"] = lag["pickup_hour_eat"] + pd.Timedelta(hours=168)
    lag = lag.rename(columns={"trips": "trips_lag_168"})
    out = out.merge(lag[["zone", "pickup_hour_eat", "trips_lag_168"]], on=["zone", "pickup_hour_eat"], how="left")

    # fill lag with zone-hour median from history where possible
    if hist["trips"].notna().any():
        tmp = hist.copy()
        tmp["hour"] = tmp["pickup_hour_eat"].dt.hour
        tmp["dow"] = tmp["pickup_hour_eat"].dt.dayofweek
        med = tmp.groupby(["zone", "dow", "hour"])["trips"].median()
        need = out["trips_lag_168"].isna()
        keys = pd.MultiIndex.from_frame(out.loc[need, ["zone", "dow", "hour"]])
        out.loc[need, "trips_lag_168"] = keys.map(med)
    out["trips_lag_168"] = out["trips_lag_168"].fillna(out["trips_lag_168"].median() if out["trips_lag_168"].notna().any() else 0)

    return out


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(FEATURE_SPEC)
