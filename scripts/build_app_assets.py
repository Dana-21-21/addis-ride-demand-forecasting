"""Bundle everything the demo app needs into app/assets/ (Deliverable E).

Run once from the project root after the master tables exist:

    .venv/bin/python scripts/build_app_assets.py

Outputs
    model.joblib            final LightGBM + feature list + validation error bands
    forecast_features.parquet  model-ready rows for 1–14 Nov 2025 (12 zones x 336 h)
    weather_hourly_clean.csv   cleaned city weather (forecast rows for the horizon)
    events_clean.csv           cleaned, zone-expanded event calendar
    zone_profile.csv           typical trips per zone x weekday x hour (last 8 weeks)
    zone_fares.csv             trip-weighted average fare per zone
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cleaning import load_raw_events, load_raw_weather, prepare_events, prepare_weather  # noqa: E402
from src.modeling import (  # noqa: E402
    ALL_FEATURES,
    SEED,
    build_model_table,
    event_lookup,
    load_masters,
    mae,
    make_model,
    split_by_cutoff,
)

ASSETS = ROOT / "app" / "assets"
D_MODEL = ROOT / "models" / "final_model.joblib"
VALID_CUTOFF = "2025-10-18"
HORIZON_START = pd.Timestamp("2025-11-01", tz="Africa/Addis_Ababa")
HORIZON_END = pd.Timestamp("2025-11-15", tz="Africa/Addis_Ababa")


def fit_lightgbm(df: pd.DataFrame):
    _, factory = make_model("lightgbm")
    model = factory(ALL_FEATURES)
    model.fit(df[ALL_FEATURES], df["trips"], categorical_feature=["zone_id"])
    return model


def main() -> None:
    np.random.seed(SEED)
    ASSETS.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    def step(msg: str) -> None:
        print(f"{time.perf_counter() - t0:7.1f}s  {msg}", flush=True)

    train, test = load_masters()
    events = event_lookup()
    history = train[train["trips"].notna()]
    step("loaded masters")
    train_tbl = build_model_table(history, history, events)
    step("train features")
    test_tbl = build_model_table(test, history, events)
    step("test features")

    # residual bands from a 14-day holdout, same horizon as the real forecast
    tr, va = split_by_cutoff(train_tbl, VALID_CUTOFF)
    holdout_model = fit_lightgbm(tr)
    va = va.assign(pred=np.clip(holdout_model.predict(va[ALL_FEATURES]), 0, None))
    va["resid"] = va["trips"] - va["pred"]
    bands = va.groupby("zone")["resid"].quantile([0.1, 0.9]).unstack()
    bands.columns = ["q10", "q90"]
    valid_mae = mae(va["trips"], va["pred"])
    print(f"holdout {VALID_CUTOFF} +14d  MAE={valid_mae:.2f}")

    step("holdout model")

    # ship the Deliverable D model when it exists so the demo matches the report and submission
    if D_MODEL.exists():
        d = joblib.load(D_MODEL)
        final_model, features = d["models"]["default"], d["features"]
        label = f"Deliverable D final ({d.get('label', 'lightgbm')})"
        valid_mae = float(d.get("validation", {}).get("rolling_mae_mean", valid_mae))
        missing = set(features) - set(test_tbl.columns)
        assert not missing, f"D model needs features the app table lacks: {missing}"
    else:
        final_model, features, label = fit_lightgbm(train_tbl), ALL_FEATURES, "LightGBM (app build)"
    step(f"final model: {label}")
    joblib.dump(
        {
            "model": final_model,
            "features": features,
            "label": label,
            "trained_through": str(history["pickup_hour_eat"].max()),
            "holdout_mae": valid_mae,
            "residual_bands": bands.to_dict(orient="index"),
        },
        ASSETS / "model.joblib",
    )

    keep = ["zone", "pickup_hour_eat", "row_id"] + list(dict.fromkeys(ALL_FEATURES + features))
    test_tbl[keep].to_parquet(ASSETS / "forecast_features.parquet", index=False)

    weather = prepare_weather(load_raw_weather())
    weather = weather[
        (weather["timestamp_eat"] >= HORIZON_START - pd.Timedelta(days=1))
        & (weather["timestamp_eat"] < HORIZON_END + pd.Timedelta(days=1))
    ]
    weather[["timestamp_eat", "temp_c", "rain_mm", "humidity_pct", "wind_kmh", "data_type"]].to_csv(
        ASSETS / "weather_hourly_clean.csv", index=False
    )

    ev = prepare_events(load_raw_events())
    ev = ev.drop_duplicates(["zone", "event_type", "start_eat", "end_eat", "venue"])
    ev[
        [
            "event_id",
            "event_name",
            "event_type",
            "venue",
            "zone",
            "start_eat",
            "end_eat",
            "window_start",
            "window_end",
            "attendance",
            "status_clean",
        ]
    ].to_csv(ASSETS / "events_clean.csv", index=False)

    recent = history[history["pickup_hour_eat"] >= history["pickup_hour_eat"].max() - pd.Timedelta(weeks=8)]
    recent.groupby(["zone", "dow", "hour"])["trips"].mean().rename("typical_trips").reset_index().to_csv(
        ASSETS / "zone_profile.csv", index=False
    )

    fares = history.dropna(subset=["avg_fare_birr"])
    fares = (
        fares.assign(rev=fares["trips"] * fares["avg_fare_birr"])
        .groupby("zone")[["rev", "trips"]]
        .sum()
    )
    (fares["rev"] / fares["trips"]).rename("avg_fare_birr").reset_index().to_csv(
        ASSETS / "zone_fares.csv", index=False
    )

    for p in sorted(ASSETS.glob("*")):
        if p.name != ".gitkeep":
            print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
