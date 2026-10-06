"""Modeling helpers for Deliverable D (and the demo app).

All lag features look back at least 336 hours (14 days), so every hour of a
14-day forecast horizon can be built the same way for train, validation and test.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cleaning import CANONICAL_ZONES, load_raw_events, prepare_events

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

SEED = 42
HORIZON_H = 336
TRIPS_PER_DRIVER_HOUR = 1.3
# 99.5% of zone-hours run at <= 1.8 trips per active driver; a separate cluster at 8–15
# looks like trip counts inflated ~10x while the driver count stayed normal
MAX_TRIPS_PER_DRIVER = 3.0

SHOCK_TYPES = [
    "football_match",
    "concert",
    "conference",
    "exhibition",
    "sports_run",
    "road_closure",
]

ZONE_FEATURES = ["zone_id"]
CALENDAR_FEATURES = ["hour", "dow", "is_weekend", "day_of_month", "is_payday_window"]
TREND_LAG_FEATURES = [
    "trend_day",
    "lag_336",
    "lag_504",
    "lag_wk_mean",
    "zone_level_2w",
    "zone_hour_profile_4w",
]
WEATHER_FEATURES = ["temp_c", "rain_mm", "rain_last_3h", "rain_class", "humidity_pct", "wind_kmh"]
EVENT_FEATURES = [
    "is_public_holiday",
    "is_school_break",
    "in_event_window",
    "n_events",
    "event_attendance",
    "hours_to_event_start",
    "ev_pre",
    "ev_during",
    "ev_post",
] + [f"ev_{t}" for t in SHOCK_TYPES]

BASE_FEATURES = ZONE_FEATURES + CALENDAR_FEATURES + TREND_LAG_FEATURES
ALL_FEATURES = BASE_FEATURES + WEATHER_FEATURES + EVENT_FEATURES

LEAKY_COLUMNS = ["active_drivers", "avg_wait_min", "avg_fare_birr"]


# --------------------------------------------------------------------------- data


def load_masters() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED / "master_train.csv")
    test = pd.read_csv(PROCESSED / "master_test.csv")
    for df in (train, test):
        df["pickup_hour_eat"] = pd.to_datetime(df["pickup_hour_eat"], utc=True).dt.tz_convert(
            "Africa/Addis_Ababa"
        )
        df["matched_event_ids"] = df["matched_event_ids"].fillna("")
    return train, test


def event_lookup() -> pd.DataFrame:
    ev = prepare_events(load_raw_events())
    ev = ev[ev["status_clean"].str.contains("confirm", na=False)]
    return ev[["event_id", "zone", "event_type", "start_eat", "end_eat"]].astype({"event_id": str})


def add_event_detail(df: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-type flags and pre / during / post phase for matched shock events."""
    events = event_lookup() if events is None else events
    out = df.copy()
    for col in ["ev_pre", "ev_during", "ev_post"] + [f"ev_{t}" for t in SHOCK_TYPES]:
        out[col] = 0

    pairs = out.loc[out["matched_event_ids"] != "", ["zone", "pickup_hour_eat", "matched_event_ids"]]
    if pairs.empty:
        return out
    pairs = pairs.assign(event_id=pairs["matched_event_ids"].str.split(",")).explode("event_id")
    pairs = pairs.reset_index().merge(events, on=["event_id", "zone"], how="inner")
    pairs = pairs[pairs["event_type"].isin(SHOCK_TYPES)]

    t = pairs["pickup_hour_eat"]
    pairs["ev_pre"] = (t < pairs["start_eat"]).astype(int)
    pairs["ev_post"] = (t > pairs["end_eat"]).astype(int)
    pairs["ev_during"] = ((t >= pairs["start_eat"]) & (t <= pairs["end_eat"])).astype(int)
    for typ in SHOCK_TYPES:
        pairs[f"ev_{typ}"] = (pairs["event_type"] == typ).astype(int)

    flags = pairs.groupby("index")[["ev_pre", "ev_during", "ev_post"] + [f"ev_{t}" for t in SHOCK_TYPES]].max()
    out.loc[flags.index, flags.columns] = flags.to_numpy()
    return out


def add_horizon_lags(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Lag features that only use trips at least 14 days before the target hour.

    `history` holds known trips (zone, pickup_hour_eat, trips). Gaps (outages,
    pre-launch hours) stay NaN rather than being treated as zero demand.
    """
    hist = history[["zone", "pickup_hour_eat", "trips"]].dropna()
    grid_start = hist["pickup_hour_eat"].min()
    grid_end = max(hist["pickup_hour_eat"].max(), df["pickup_hour_eat"].max())
    hours = pd.date_range(grid_start, grid_end, freq="h")

    wide = hist.pivot_table(index="pickup_hour_eat", columns="zone", values="trips").reindex(hours)
    wide = wide.reindex(columns=sorted(set(wide.columns) | set(df["zone"].unique())))

    feats = {
        "lag_336": wide.shift(336),
        "lag_504": wide.shift(504),
        "lag_wk_mean": pd.concat([wide.shift(h) for h in (336, 504, 672, 840)]).groupby(level=0).mean(),
        "zone_level_2w": wide.shift(336).rolling(168, min_periods=24).mean(),
    }
    # mean of the same hour-of-day over the 4 weeks ending 14 days earlier (all weekdays)
    by_hour = wide.shift(336)
    feats["zone_hour_profile_4w"] = (
        pd.concat([by_hour.shift(24 * d) for d in range(28)]).groupby(level=0).mean()
    )

    out = df.copy()
    keys = pd.MultiIndex.from_arrays([out["pickup_hour_eat"], out["zone"]])
    for name, frame in feats.items():
        out[name] = frame.stack(future_stack=True).reindex(keys).to_numpy()
    return out


def build_model_table(df: pd.DataFrame, history: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    out = add_horizon_lags(df, history)
    out = add_event_detail(out, events)
    out["zone_id"] = out["zone"].map({z: i for i, z in enumerate(CANONICAL_ZONES)}).astype(int)
    out["rain_last_3h"] = out["rain_last_3h"].fillna(out["rain_mm"])
    if {"trips", "active_drivers"} <= set(out.columns):
        out["suspect_target"] = out["trips"] / out["active_drivers"].clip(lower=1) > MAX_TRIPS_PER_DRIVER
    return out


# --------------------------------------------------------------------------- splits & metrics


def tz_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="Africa/Addis_Ababa")


def split_by_cutoff(df: pd.DataFrame, cutoff: str, days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = tz_ts(cutoff)
    tr = df[df["pickup_hour_eat"] < c]
    va = df[(df["pickup_hour_eat"] >= c) & (df["pickup_hour_eat"] < c + pd.Timedelta(days=days))]
    return tr, va


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


# --------------------------------------------------------------------------- baselines


def mean_baseline(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return np.full(len(valid), train["trips"].mean())


def seasonal_naive(train: pd.DataFrame, valid: pd.DataFrame, weeks: int | None = 4) -> np.ndarray:
    """Mean trips for the same zone × dow × hour (optionally last `weeks` weeks only)."""
    tr = train
    if weeks is not None:
        tr = train[train["pickup_hour_eat"] >= train["pickup_hour_eat"].max() - pd.Timedelta(weeks=weeks)]
    prof = tr.groupby(["zone", "dow", "hour"])["trips"].mean()
    full = train.groupby(["zone", "dow", "hour"])["trips"].mean()
    keys = pd.MultiIndex.from_frame(valid[["zone", "dow", "hour"]])
    pred = pd.Series(keys.map(prof), index=valid.index)
    pred = pred.fillna(pd.Series(keys.map(full), index=valid.index))
    return pred.fillna(train["trips"].mean()).to_numpy()


# --------------------------------------------------------------------------- models


def make_model(name: str, params: dict | None = None):
    from lightgbm import LGBMRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    params = params or {}
    if name == "ridge":
        return "ridge", lambda feats: make_pipeline(
            ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        [f for f in feats if f in ("zone_id", "hour", "dow")],
                    ),
                    (
                        "num",
                        make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                        [f for f in feats if f not in ("zone_id", "hour", "dow")],
                    ),
                ]
            ),
            Ridge(alpha=params.get("alpha", 1.0)),
        )
    if name == "random_forest":
        return "random_forest", lambda feats: make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(
                n_estimators=params.get("n_estimators", 200),
                min_samples_leaf=params.get("min_samples_leaf", 3),
                max_features=params.get("max_features", 0.5),
                n_jobs=N_JOBS,
                random_state=SEED,
            ),
        )
    if name == "hist_gbm":
        return "hist_gbm", lambda feats: HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 600),
            learning_rate=params.get("learning_rate", 0.05),
            categorical_features=[feats.index("zone_id")] if "zone_id" in feats else None,
            random_state=SEED,
        )
    if name == "lightgbm":
        defaults = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
        )
        defaults.update(params)
        # n_jobs=-1 oversubscribes threads on this machine and is ~5x slower
        return "lightgbm", lambda feats: LGBMRegressor(**defaults, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    raise ValueError(name)


N_JOBS = 4


def fit_model(name: str, train: pd.DataFrame, feats: list[str], params: dict | None = None, y=None):
    _, factory = make_model(name, params)
    model = factory(feats)
    y = train["trips"] if y is None else y
    t0 = time.perf_counter()
    if name == "lightgbm":
        model.fit(train[feats], y, categorical_feature=["zone_id"] if "zone_id" in feats else "auto")
    else:
        model.fit(train[feats], y)
    return model, time.perf_counter() - t0


def fit_predict(name: str, train: pd.DataFrame, valid: pd.DataFrame, feats: list[str], params: dict | None = None):
    model, secs = fit_model(name, train, feats, params)
    pred = np.clip(model.predict(valid[feats]), 0, None)
    return model, pred, secs


# --------------------------------------------------------------------------- final-model config


def add_lag_168(df: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Same zone-hour one week earlier; NaN when that hour is not in `history`."""
    lag = history[["zone", "pickup_hour_eat", "trips"]].dropna().copy()
    lag["pickup_hour_eat"] = lag["pickup_hour_eat"] + pd.Timedelta(hours=168)
    lag = lag.rename(columns={"trips": "lag_168"})
    out = df.drop(columns=["lag_168"], errors="ignore")
    return out.merge(lag, on=["zone", "pickup_hour_eat"], how="left")


def cap_targets(train: pd.DataFrame, q: float) -> pd.Series:
    caps = train.groupby("zone")["trips"].quantile(q)
    return train["trips"].clip(upper=train["zone"].map(caps))


def fit_final(train: pd.DataFrame, cutoff: str, config: dict) -> dict:
    """Fit the configured model(s) on rows before `cutoff`.

    config keys: params (LightGBM), features, horizon_split (bool), cap_q (float | None),
    drop_suspect (bool: train without rows flagged by `suspect_target`).
    With horizon_split, a second model that also sees `lag_168` serves the first
    7 days after the cutoff — the only days whose one-week lag is already known.
    """
    if config.get("drop_suspect") and "suspect_target" in train.columns:
        train = train[~train["suspect_target"]]
    feats = list(config["features"])
    y = cap_targets(train, config["cap_q"]) if config.get("cap_q") else train["trips"]
    bundle = {"cutoff": cutoff, "config": config, "features": feats, "models": {}}
    bundle["models"]["default"], _ = fit_model("lightgbm", train, feats, config["params"], y=y)
    if config.get("horizon_split"):
        f1 = feats + ["lag_168"]
        bundle["features_week1"] = f1
        bundle["models"]["week1"], _ = fit_model("lightgbm", train, f1, config["params"], y=y)
    return bundle


def predict_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    pred = bundle["models"]["default"].predict(df[bundle["features"]])
    if "week1" in bundle["models"]:
        c = tz_ts(bundle["cutoff"])
        wk1 = (df["pickup_hour_eat"] < c + pd.Timedelta(days=7)).to_numpy() & df["lag_168"].notna().to_numpy()
        if wk1.any():
            pred[wk1] = bundle["models"]["week1"].predict(df.loc[wk1, bundle["features_week1"]])
    return np.clip(pred, 0, None)


def rolling_eval(full: pd.DataFrame, cutoffs: list[str], predict_fn) -> pd.DataFrame:
    """predict_fn(train, valid, cutoff) -> predictions. Returns per-fold scores with OOF preds attached."""
    rows, oof = [], []
    for c in cutoffs:
        tr, va = split_by_cutoff(full, c)
        p = predict_fn(tr, va, c)
        row = {"cutoff": c, "n_valid": len(va), "rmse": rmse(va["trips"], p), "mae": mae(va["trips"], p)}
        if "suspect_target" in va.columns:
            ok = ~va["suspect_target"].to_numpy()
            row.update(rmse_clean=rmse(va["trips"][ok], p[ok]), mae_clean=mae(va["trips"][ok], p[ok]))
        rows.append(row)
        oof.append(va.assign(pred=p, fold=c))
    res = pd.DataFrame(rows)
    res.attrs["oof"] = pd.concat(oof)
    return res
