"""Cleaning helpers for trips, weather, and events (Deliverable A)."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Canonical 12 zone labels — expand/adjust during cleaning after inspecting raw values
CANONICAL_ZONES = [
    "Ayat",
    "Bole",
    "CMC",
    "Gerji",
    "Kazanchis",
    "Lideta",
    "Megenagna",
    "Mexico",
    "Piassa",
    "Saris",
    "Summit",
    "Tor Hailoch",
]


def load_raw_trips(split: str = "train") -> pd.DataFrame:
    name = "ride_demand_train.csv" if split == "train" else "ride_demand_test.csv"
    return pd.read_csv(RAW / name)


def load_raw_weather() -> pd.DataFrame:
    return pd.read_csv(RAW / "weather_hourly.csv")


def load_raw_events() -> pd.DataFrame:
    return pd.read_csv(RAW / "events_calendar.csv")


def normalize_zone(series: pd.Series) -> pd.Series:
    """Placeholder: map messy zone spellings to CANONICAL_ZONES."""
    return series.astype(str).str.strip().str.title()


# TODO: parse timestamps, convert clocks to Africa/Addis_Ababa, join weather + events
