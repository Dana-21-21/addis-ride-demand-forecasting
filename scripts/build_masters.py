"""Build master_train / master_test / data dictionary (A4–A8 pipeline)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleaning import CANONICAL_ZONES, build_joined_tables  # noqa: E402
from features import FEATURE_SPEC, add_features, feature_dictionary  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def _to_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pickup_hour_eat" in out.columns:
        out["pickup_hour_eat"] = out["pickup_hour_eat"].dt.tz_convert("Africa/Addis_Ababa").dt.strftime(
            "%Y-%m-%d %H:%M:%S%z"
        )
    return out


def validate(master_train: pd.DataFrame, master_test: pd.DataFrame, train_n: int, test_n: int) -> list[str]:
    lines = []

    def check(name: str, ok: bool):
        status = "PASS" if ok else "FAIL"
        lines.append(f"[{status}] {name}")
        print(lines[-1])

    check("train one row per zone-hour", not master_train.duplicated(["zone", "pickup_hour_eat"]).any())
    check("test one row per zone-hour", not master_test.duplicated(["zone", "pickup_hour_eat"]).any())
    check("train row count unchanged by joins", len(master_train) == train_n)
    check("test row count unchanged by joins", len(master_test) == test_n)
    check("only 12 canonical zones (train)", set(master_train["zone"].unique()) <= set(CANONICAL_ZONES))
    check("only 12 canonical zones (test)", set(master_test["zone"].unique()) <= set(CANONICAL_ZONES))
    check("no negative trips left", not ((master_train["trips"] < 0).any()))
    check("no rain sentinel -9999", not ((master_train["rain_mm"] == -9999).any() or (master_test["rain_mm"] == -9999).any()))
    check(
        "timestamps on Africa/Addis_Ababa",
        str(master_train["pickup_hour_eat"].dt.tz) == "Africa/Addis_Ababa"
        and str(master_test["pickup_hour_eat"].dt.tz) == "Africa/Addis_Ababa",
    )

    feat_cols = [f["name"] for f in FEATURE_SPEC]
    check("train has all engineered features", all(c in master_train.columns for c in feat_cols))
    check("test has all engineered features", all(c in master_test.columns for c in feat_cols))
    check("test has no target column", "trips" not in master_test.columns)

    leak_ops = {"avg_fare_birr", "avg_wait_min", "active_drivers"}
    check(
        "no ops leakage features in model feature list",
        leak_ops.isdisjoint(set(feat_cols)),
    )
    return lines


def main():
    print("Building joined tables...")
    parts = build_joined_tables()
    trips_train = parts["trips_train"]
    weather = parts["weather"]
    events = parts["events"]
    train_joined = parts["train_joined"]
    test_joined = parts["test_joined"]

    # ---- A4 audit numbers ----
    w_keys = set(weather["timestamp_eat"])
    miss_w = (~trips_train["pickup_hour_eat"].isin(w_keys)).sum()
    confirmed = events[events["status_clean"].str.contains("confirm", na=False)]
    cancelled = events[~events["status_clean"].str.contains("confirm", na=False)]
    matched_ids = set()
    for s in train_joined.loc[train_joined["n_events"] > 0, "matched_event_ids"]:
        if s:
            matched_ids.update(s.split(","))
    all_conf = set(confirmed["event_id"].astype(str))
    audit = {
        "train_rows_before": len(trips_train),
        "train_rows_after_weather": len(train_joined),
        "weather_match_rate_pct": round(100 * (1 - miss_w / len(trips_train)), 2),
        "zone_hours_no_weather_before_fill": int(miss_w),
        "confirmed_event_ids": int(confirmed["event_id"].nunique()),
        "cancelled_event_rows": int(len(cancelled)),
        "events_matched_ge1": int(len(matched_ids)),
        "events_unmatched": int(len(all_conf - matched_ids)),
    }
    print("A4 audit:", audit)

    print("Adding features...")
    master_train = add_features(train_joined, train_trips_history=trips_train)
    master_test = add_features(test_joined, train_trips_history=trips_train)
    # test must not carry target
    master_test = master_test.drop(columns=["trips"], errors="ignore")

    print("Validating...")
    lines = validate(master_train, master_test, len(trips_train), len(parts["trips_test"]))

    # data dictionary
    dd_rows = [
        {"column": "zone", "type": "category", "source": "trips", "description": "Canonical zone label", "derived": "normalize_zone_label"},
        {"column": "pickup_hour_eat", "type": "datetime(tz)", "source": "trips", "description": "Hour in Africa/Addis_Ababa", "derived": "parse_trip_hour"},
        {"column": "trips", "type": "float", "source": "trips", "description": "Target (train only)", "derived": "clean + impute"},
        {"column": "temp_c", "type": "float", "source": "weather", "description": "Temperature C", "derived": "join_weather"},
        {"column": "rain_mm", "type": "float", "source": "weather", "description": "Rain mm", "derived": "join_weather"},
        {"column": "humidity_pct", "type": "float", "source": "weather", "description": "Humidity %", "derived": "join_weather"},
        {"column": "wind_kmh", "type": "float", "source": "weather", "description": "Wind km/h", "derived": "join_weather"},
        {"column": "data_type", "type": "category", "source": "weather", "description": "observed/forecast", "derived": "join_weather"},
        {"column": "n_events", "type": "int", "source": "events", "description": "Matched confirmed events", "derived": "join_events"},
        {"column": "matched_event_ids", "type": "string", "source": "events", "description": "Matched event ids", "derived": "join_events"},
    ]
    for f in FEATURE_SPEC:
        dd_rows.append(
            {
                "column": f["name"],
                "type": "engineered",
                "source": f["source"],
                "description": f["why"],
                "derived": f["formula"],
            }
        )
    data_dictionary = pd.DataFrame(dd_rows)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    _to_export(master_train).to_csv(PROCESSED / "master_train.csv", index=False)
    _to_export(master_test).to_csv(PROCESSED / "master_test.csv", index=False)
    data_dictionary.to_csv(PROCESSED / "data_dictionary_master.csv", index=False)
    feature_dictionary().to_csv(REPORTS / "A6_feature_table.csv", index=False)
    pd.DataFrame([audit]).to_csv(REPORTS / "A4_join_audit.csv", index=False)
    (REPORTS / "A7_integrity_checks.txt").write_text("\n".join(lines) + "\n")

    print("Wrote:")
    print(" ", PROCESSED / "master_train.csv", master_train.shape)
    print(" ", PROCESSED / "master_test.csv", master_test.shape)
    print(" ", PROCESSED / "data_dictionary_master.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
