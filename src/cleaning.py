"""Cleaning helpers for trips, weather, and events (Deliverable A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CANONICAL_ZONES = [
    "Arat Kilo",
    "Ayat",
    "Bole",
    "CMC",
    "Gerji",
    "Kazanchis",
    "Kolfe",
    "Lideta",
    "Megenagna",
    "Merkato",
    "Piassa",
    "Sarbet",
]

ZONE_ALIASES = {
    "arat kilo": "Arat Kilo",
    "arat  kilo": "Arat Kilo",
    "ayat": "Ayat",
    "bole": "Bole",
    "bole rd": "Bole",
    "cmc": "CMC",
    "c.m.c": "CMC",
    "gerji": "Gerji",
    "kazanchis": "Kazanchis",
    "kazanches": "Kazanchis",
    "kazanchis (kirkos)": "Kazanchis",
    "kolfe": "Kolfe",
    "kolfe keranio": "Kolfe",
    "lideta": "Lideta",
    "megenagna": "Megenagna",
    "megenaga": "Megenagna",
    "merkato": "Merkato",
    "mercato": "Merkato",
    "piassa": "Piassa",
    "piazza": "Piassa",
    "piassa (arada)": "Piassa",
    "sarbet": "Sarbet",
}

CITYWIDE_TOKENS = {"all", "all zones", "citywide", "city-wide", "city wide"}

# hours before start / after end by event type
EVENT_WINDOW = {
    "football_match": (2, 2),
    "concert": (1, 2),
    "sports_run": (1, 1),
    "conference": (0, 0),
    "exhibition": (0, 0),
    "road_closure": (0, 0),
    "public_holiday": (0, 0),
    "school_break": (0, 0),
}
DEFAULT_WINDOW = (1, 1)

DEFAULT_DURATION_HOURS = {
    "football_match": 2,
    "concert": 3,
    "sports_run": 3,
    "conference": 8,
    "exhibition": 8,
    "road_closure": 10,
    "public_holiday": 24,
    "school_break": 24,
}


def load_raw_trips(split: str = "train") -> pd.DataFrame:
    name = "ride_demand_train.csv" if split == "train" else "ride_demand_test.csv"
    return pd.read_csv(RAW / name)


def load_raw_weather() -> pd.DataFrame:
    return pd.read_csv(RAW / "weather_hourly.csv")


def load_raw_events() -> pd.DataFrame:
    return pd.read_csv(RAW / "events_calendar.csv")


def _zone_key(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def normalize_zone_label(x):
    if pd.isna(x):
        return None
    key = _zone_key(x)
    if "(" in key:
        key = key.split("(")[0].strip()
    if key in CITYWIDE_TOKENS:
        return "__CITYWIDE__"
    if key in ZONE_ALIASES:
        return ZONE_ALIASES[key]
    titled = " ".join(w.capitalize() for w in key.split())
    if titled == "Cmc":
        titled = "CMC"
    return titled if titled in CANONICAL_ZONES else None


def expand_event_zones(zone_raw):
    if pd.isna(zone_raw):
        return []
    text = str(zone_raw).strip()
    parts = [p.strip() for p in text.replace(" and ", "&").split("&")]
    out = []
    for p in parts:
        z = normalize_zone_label(p)
        if z == "__CITYWIDE__":
            return list(CANONICAL_ZONES)
        if z is not None:
            out.append(z)
    return sorted(set(out))


def normalize_event_type(x):
    key = " ".join(str(x).strip().lower().replace("-", "_").split()).replace(" ", "_")
    mapping = {
        "football_match": "football_match",
        "footballmatch": "football_match",
        "road_closure": "road_closure",
        "roadclosure": "road_closure",
        "public_holiday": "public_holiday",
        "publicholiday": "public_holiday",
        "sports_run": "sports_run",
        "sportsrun": "sports_run",
        "conference": "conference",
        "concert": "concert",
        "exhibition": "exhibition",
        "school_break": "school_break",
        "schoolbreak": "school_break",
    }
    return mapping.get(key, key)


def parse_trip_hour(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, Africa/Addis_Ababa]")
    m_slash = s.str.contains("/")
    m_iso_tz = s.str.contains(r"\+0?3:00")
    m_naive = ~m_slash & ~m_iso_tz
    if m_slash.any():
        out.loc[m_slash] = (
            pd.to_datetime(s.loc[m_slash], dayfirst=True, errors="coerce")
            .dt.tz_localize("Africa/Addis_Ababa")
        )
    if m_iso_tz.any():
        out.loc[m_iso_tz] = (
            pd.to_datetime(s.loc[m_iso_tz], errors="coerce")
            .dt.tz_convert("Africa/Addis_Ababa")
        )
    if m_naive.any():
        out.loc[m_naive] = (
            pd.to_datetime(s.loc[m_naive], errors="coerce")
            .dt.tz_localize("Africa/Addis_Ababa")
        )
    return out


def parse_weather_timestamp(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, Africa/Addis_Ababa]")
    m_z = s.str.endswith("Z")
    m_slash = s.str.contains("/")
    if m_z.any():
        out.loc[m_z] = (
            pd.to_datetime(s.loc[m_z], utc=True, errors="coerce")
            .dt.tz_convert("Africa/Addis_Ababa")
        )
    if m_slash.any():
        out.loc[m_slash] = (
            pd.to_datetime(s.loc[m_slash], dayfirst=True, errors="coerce")
            .dt.tz_localize("Africa/Addis_Ababa")
        )
    return out


EVENT_DATETIME_FORMATS = {
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$": "%Y-%m-%d %H:%M",
    r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$": "%d/%m/%Y %H:%M",
    r"^[A-Za-z]{3} \d{2}, \d{4} \d{2}:\d{2} [AP]M$": "%b %d, %Y %I:%M %p",
}


def parse_event_datetime(series: pd.Series) -> pd.Series:
    # One explicit format per pattern: dayfirst must apply to slash dates only,
    # otherwise ISO "2025-05-01" is misread as 5 January.
    s = series.astype(str).str.strip()
    naive = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for pattern, fmt in EVENT_DATETIME_FORMATS.items():
        m = s.str.match(pattern)
        if m.any():
            naive.loc[m] = pd.to_datetime(s.loc[m], format=fmt, errors="coerce")
    return naive.dt.tz_localize("Africa/Addis_Ababa")


def parse_attendance(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    digits = s.str.replace(",", "", regex=False).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(digits, errors="coerce")


def prepare_trips(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["zone"] = out["zone"].map(normalize_zone_label)
    out["pickup_hour_eat"] = parse_trip_hour(out["pickup_hour"])
    out["pickup_hour_eat"] = out["pickup_hour_eat"].dt.floor("h")
    if is_train:
        out.loc[out["trips"] < 0, "trips"] = np.nan
        out.loc[out["avg_wait_min"] < 0, "avg_wait_min"] = np.nan
        agg = (
            out.groupby(["zone", "pickup_hour_eat"], as_index=False)
            .agg(
                trips=("trips", "mean"),
                avg_fare_birr=("avg_fare_birr", "mean"),
                avg_wait_min=("avg_wait_min", "mean"),
                active_drivers=("active_drivers", "mean"),
                record_id=("record_id", "first"),
            )
        )
        # impute trips with zone×hour×dow median
        agg["dow"] = agg["pickup_hour_eat"].dt.dayofweek
        agg["hour"] = agg["pickup_hour_eat"].dt.hour
        med = (
            agg.dropna(subset=["trips"])
            .groupby(["zone", "dow", "hour"])["trips"]
            .median()
        )
        fill = agg.set_index(["zone", "dow", "hour"]).index.map(med)
        agg["trips"] = agg["trips"].fillna(pd.Series(fill, index=agg.index))
        agg["trips"] = agg["trips"].fillna(agg["trips"].median())
        return agg.drop(columns=["dow", "hour"])
    # test
    out = out.rename(columns={"row_id": "row_id"})
    id_col = "row_id" if "row_id" in out.columns else "record_id"
    agg = (
        out.groupby(["zone", "pickup_hour_eat"], as_index=False)
        .agg(**{id_col: (id_col, "first")})
    )
    return agg


def prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_eat"] = parse_weather_timestamp(out["timestamp"]).dt.floor("h")
    out.loc[out["rain_mm"] == -9999, "rain_mm"] = np.nan
    f_mask = out["temp_c"].between(50, 80)
    out.loc[f_mask, "temp_c"] = (out.loc[f_mask, "temp_c"] - 32) * 5 / 9
    # prefer observed over forecast when deduping
    out["_obs"] = (out["data_type"].astype(str).str.lower() == "observed").astype(int)
    out = out.sort_values(["timestamp_eat", "_obs"], ascending=[True, False])
    out = out.drop_duplicates("timestamp_eat", keep="first")
    out = out.sort_values("timestamp_eat")
    for col in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        out[col] = out[col].interpolate(limit_direction="both")
    return out.drop(columns=["_obs"]).reset_index(drop=True)


def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_type"] = out["event_type"].map(normalize_event_type)
    out["status_clean"] = out["status"].astype(str).str.strip().str.lower()
    out["start_eat"] = parse_event_datetime(out["start_datetime"])
    out["end_eat"] = parse_event_datetime(out["end_datetime"])
    # swap inverted windows
    bad = out["end_eat"].notna() & out["start_eat"].notna() & (out["end_eat"] < out["start_eat"])
    out.loc[bad, ["start_eat", "end_eat"]] = out.loc[bad, ["end_eat", "start_eat"]].to_numpy()
    # impute missing ends
    miss = out["end_eat"].isna() & out["start_eat"].notna()
    dur = out.loc[miss, "event_type"].map(DEFAULT_DURATION_HOURS).fillna(2)
    out.loc[miss, "end_eat"] = out.loc[miss, "start_eat"] + pd.to_timedelta(dur, unit="h")
    out["attendance"] = parse_attendance(out["expected_attendance"])
    out["zones_list"] = out["zone"].map(expand_event_zones)
    out = out.drop(columns=["zone"]).explode("zones_list").rename(columns={"zones_list": "zone"})
    out = out.dropna(subset=["zone"]).reset_index(drop=True)
    pre = out["event_type"].map(lambda t: EVENT_WINDOW.get(t, DEFAULT_WINDOW)[0])
    post = out["event_type"].map(lambda t: EVENT_WINDOW.get(t, DEFAULT_WINDOW)[1])
    out["window_start"] = out["start_eat"] - pd.to_timedelta(pre, unit="h")
    out["window_end"] = out["end_eat"] + pd.to_timedelta(post, unit="h")
    return out.reset_index(drop=True)


def join_weather(trips: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather[
        ["timestamp_eat", "temp_c", "rain_mm", "humidity_pct", "wind_kmh", "data_type"]
    ].rename(columns={"timestamp_eat": "pickup_hour_eat"})
    before = len(trips)
    merged = trips.merge(w, on="pickup_hour_eat", how="left")
    assert len(merged) == before, "Weather join changed row count — check duplicate hours"
    # fill small gaps from nearest earlier/later city weather
    weather_cols = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]
    filled = (
        merged[["pickup_hour_eat"] + weather_cols]
        .drop_duplicates("pickup_hour_eat")
        .sort_values("pickup_hour_eat")
    )
    filled[weather_cols] = filled[weather_cols].ffill().bfill()
    merged = merged.drop(columns=weather_cols).merge(
        filled, on="pickup_hour_eat", how="left"
    )
    assert len(merged) == before
    return merged


def join_events(trips: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event features via interval match (zone + time window).

    Loops over events (small) and masks trips — avoids huge cross joins.
    """
    confirmed = events[events["status_clean"].str.contains("confirm", na=False)].copy()
    out = trips.copy().reset_index(drop=True)
    n = len(out)

    in_window = np.zeros(n, dtype=int)
    is_holiday = np.zeros(n, dtype=int)
    is_football = np.zeros(n, dtype=int)
    is_school = np.zeros(n, dtype=int)
    n_events = np.zeros(n, dtype=int)
    attendance = np.full(n, np.nan)
    hours_to = np.full(n, np.nan)
    matched: list[list[str]] = [[] for _ in range(n)]

    if confirmed.empty:
        out = out.assign(
            in_event_window=0,
            is_public_holiday=0,
            is_football_window=0,
            is_school_break=0,
            n_events=0,
            event_attendance=np.nan,
            hours_to_event_start=np.nan,
            matched_event_ids="",
        )
        return out

    shock_types = {
        "football_match",
        "concert",
        "sports_run",
        "conference",
        "exhibition",
        "road_closure",
        "public_holiday",
    }
    hours = out["pickup_hour_eat"]
    zones = out["zone"].to_numpy()

    for ev in confirmed.itertuples(index=False):
        mask = (
            (zones == ev.zone)
            & (hours >= ev.window_start).to_numpy()
            & (hours <= ev.window_end).to_numpy()
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue

        n_events[idx] += 1
        if ev.event_type == "public_holiday":
            is_holiday[idx] = 1
        if ev.event_type == "football_match":
            is_football[idx] = 1
        if ev.event_type == "school_break":
            is_school[idx] = 1
        if ev.event_type in shock_types:
            in_window[idx] = 1

        hrs = (ev.start_eat - hours.iloc[idx]).dt.total_seconds().to_numpy() / 3600.0
        prev = hours_to[idx]
        take = np.isnan(prev) | (hrs < prev)
        hours_to[idx[take]] = hrs[take]

        if pd.notna(ev.attendance):
            prev_a = attendance[idx]
            attendance[idx] = np.where(
                np.isnan(prev_a), ev.attendance, np.maximum(prev_a, ev.attendance)
            )

        eid = str(ev.event_id)
        for i in idx:
            matched[i].append(eid)

    out["in_event_window"] = in_window
    out["is_public_holiday"] = is_holiday
    out["is_football_window"] = is_football
    out["is_school_break"] = is_school
    out["n_events"] = n_events
    out["event_attendance"] = attendance
    out["hours_to_event_start"] = hours_to
    out["matched_event_ids"] = [",".join(sorted(set(xs))) if xs else "" for xs in matched]
    return out

def build_joined_tables():
    """Prepare + join train/test. Returns dict of frames for audit/export."""
    trips_train = prepare_trips(load_raw_trips("train"), is_train=True)
    trips_test = prepare_trips(load_raw_trips("test"), is_train=False)
    weather = prepare_weather(load_raw_weather())
    events = prepare_events(load_raw_events())

    train_n = len(trips_train)
    test_n = len(trips_test)

    train_w = join_weather(trips_train, weather)
    test_w = join_weather(trips_test, weather)
    assert len(train_w) == train_n
    assert len(test_w) == test_n

    train_we = join_events(train_w, events)
    test_we = join_events(test_w, events)
    assert len(train_we) == train_n
    assert len(test_we) == test_n

    return {
        "trips_train": trips_train,
        "trips_test": trips_test,
        "weather": weather,
        "events": events,
        "train_joined": train_we,
        "test_joined": test_we,
    }