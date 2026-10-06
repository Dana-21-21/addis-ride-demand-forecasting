"""Feature engineering for the master tables."""

import pandas as pd


def add_calendar_features(df: pd.DataFrame, time_col: str = "pickup_hour") -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[time_col])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["month"] = ts.dt.month
    out["day_of_month"] = ts.dt.day
    # Rough payday flag: last 3 / first 3 days of month
    out["is_payday_window"] = ((out["day_of_month"] <= 3) | (out["day_of_month"] >= 28)).astype(int)
    return out


# TODO: rain classes, event window flags, lag/trend features (fit on train only)
