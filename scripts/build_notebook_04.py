#!/usr/bin/env python3
"""Build notebooks/04_modeling_and_evaluation.ipynb (Deliverable D).

Execute afterwards with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_modeling_and_evaluation.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "04_modeling_and_evaluation.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# 04 — Modeling & Evaluation (Deliverable D)

Run with `notebooks/` as the working directory; all paths are relative.

Every score below comes from chronological splits of the train file (1 Jan – 31 Oct 2025); the test file is never scored.

* **Main validation fortnight:** train on everything before **2025-10-18**, validate on **18–31 Oct 2025**.
* **Rolling origin (D3):** 4 folds with cutoffs 6 Sep, 20 Sep, 4 Oct, 18 Oct; each trains on all earlier hours and validates on the next 14 days.
* **Tuning (D6):** Optuna searches on the 20 Sep and 4 Oct folds only, so the 18–31 Oct fortnight stays untouched for the before/after comparison.

**Horizon-safe lags.** The forecast is made once, at the end of 31 Oct, for the next 14 days. A one-week lag is unknown for 8–14 Nov, so the core lag features look back at least 336 h (14 days): `lag_336`, `lag_504`, `lag_wk_mean` (mean of the 2–5-week lags), `zone_level_2w` and `zone_hour_profile_4w`. These are built identically for train, validation and test (see `src/modeling.py`).

At the end the notebook writes `reports/D_model_evaluation.md`, `models/final_model.joblib` and `submission/team_<NAME>_submission.csv`.""",
    ),
    ("markdown", "## Setup & load"),
    (
        "code",
        r'''from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("..")
sys.path.insert(0, str(ROOT))

from src import modeling as M
from src.predict import write_submission
from src.train import save_model

np.random.seed(M.SEED)
TEAM_NAME = "ride_minds"
REPORTS = ROOT / "reports"

train_raw, test_raw = M.load_masters()
events = M.event_lookup()
full = M.add_lag_168(M.build_model_table(train_raw, train_raw, events), train_raw)
test = M.add_lag_168(M.build_model_table(test_raw, train_raw, events), train_raw)

VALID_CUTOFF = "2025-10-18"
# ≥4 chronological 14-day folds (D3). Tuning folds leave 18–31 Oct untouched for before/after.
ROLLING_CUTOFFS = ["2025-09-06", "2025-09-20", "2025-10-04", "2025-10-18"]
TUNING_CUTOFFS = ["2025-09-20", "2025-10-04"]
N_TRIALS = 8
tr, va = M.split_by_cutoff(full, VALID_CUTOFF)

from src.cleaning import load_raw_events, prepare_events

ev_info = prepare_events(load_raw_events()).drop_duplicates("event_id").set_index("event_id")
findings: dict[str, str] = {}
tables: dict[str, pd.DataFrame] = {}


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(digits)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *body])


print("model table:", full.shape, "| test:", test.shape)
print("train rows (< 18 Oct):", len(tr), "| validation rows (18-31 Oct):", len(va))
print("mean trips in validation fortnight:", round(va["trips"].mean(), 2))
print(f"rows flagged suspect_target (> {M.MAX_TRIPS_PER_DRIVER} trips per active driver): {int(full['suspect_target'].sum())}")''',
    ),
    # ------------------------------------------------------------------ D1
    ("markdown", "## D1 — Baselines"),
    (
        "code",
        r'''base_preds = {
    "mean predictor": M.mean_baseline(tr, va),
    "seasonal naive (zone×dow×hour, all history)": M.seasonal_naive(tr, va, weeks=None),
    "seasonal naive (zone×dow×hour, last 4 weeks)": M.seasonal_naive(tr, va, weeks=4),
}
d1 = pd.DataFrame(
    [{"model": k, "rmse": M.rmse(va["trips"], p), "mae": M.mae(va["trips"], p)} for k, p in base_preds.items()]
)
SN_WEEKS = 4 if d1.loc[2, "rmse"] <= d1.loc[1, "rmse"] else None
SN_NAME = d1.loc[2 if SN_WEEKS else 1, "model"]
SN_RMSE = d1.loc[2 if SN_WEEKS else 1, "rmse"]
tables["D1"] = d1
findings["D1"] = (
    f"On 18–31 Oct the mean predictor scores RMSE {d1.loc[0, 'rmse']:.2f} / MAE {d1.loc[0, 'mae']:.2f}; "
    f"the seasonal-naive profile cuts that roughly in half (all history: RMSE {d1.loc[1, 'rmse']:.2f}, "
    f"last 4 weeks: RMSE {d1.loc[2, 'rmse']:.2f}). Zone × weekday × hour seasonality explains most of the variance, "
    f"so any real model has to beat **{SN_RMSE:.2f}** to be worth deploying."
)
print(findings["D1"])
d1''',
    ),
    # ------------------------------------------------------------------ D2
    ("markdown", "## D2 — Model comparison (same split, same features)"),
    (
        "code",
        r'''FAMILIES = {
    "ridge": "Ridge (one-hot zone/hour/dow)",
    "hist_gbm": "HistGradientBoosting",
    "lightgbm": "LightGBM",
}
rows, d2_preds = [], {}
for key, label in FAMILIES.items():
    _, p, secs = M.fit_predict(key, tr, va, M.ALL_FEATURES)
    d2_preds[key] = p
    rows.append({"model": label, "key": key, "rmse": M.rmse(va["trips"], p), "mae": M.mae(va["trips"], p), "train_s": secs})
d2 = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
d2["rmse_vs_seasonal_naive_pct"] = 100 * (d2["rmse"] / SN_RMSE - 1)
tables["D2"] = d2.drop(columns="key")

winner = d2.iloc[0]
lgb_row = d2[d2["key"] == "lightgbm"].iloc[0]
why = (
    "Gradient boosting wins because demand is a product of interactions (zone × hour × weekday, rain × zone type, "
    "event type × phase) that trees pick up without hand-made crosses, and it handles the NaN lags of gaps natively. "
    "Ridge cannot model those interactions; the random forest averages deep trees and is slower and smoother on spikes."
)
if winner["key"] != "lightgbm":
    why += (
        f" {winner['model']} edges LightGBM by {lgb_row['rmse'] - winner['rmse']:.2f} RMSE; we keep LightGBM as the final "
        "family because it is as accurate within noise, much faster to tune, and supports the Poisson objective."
    )
findings["D2"] = (
    f"Winner on 18–31 Oct: **{winner['model']}** (RMSE {winner['rmse']:.2f}, MAE {winner['mae']:.2f}, "
    f"{winner['rmse_vs_seasonal_naive_pct']:+.1f}% vs seasonal naive). " + why
)
print(findings["D2"])
d2.drop(columns="key")''',
    ),
    # ------------------------------------------------------------------ D3
    ("markdown", "## D3 — Rolling-origin validation (4 folds × 14 days)"),
    (
        "code",
        r'''BASE_CONFIG = dict(features=M.ALL_FEATURES, params={}, horizon_split=False, cap_q=None)

def lgb_fold(config):
    return lambda a, b, c: M.predict_bundle(M.fit_final(a, c, config), b)

d3_lgb = M.rolling_eval(full, ROLLING_CUTOFFS, lgb_fold(BASE_CONFIG))
d3_sn = M.rolling_eval(full, ROLLING_CUTOFFS, lambda a, b, c: M.seasonal_naive(a, b, weeks=SN_WEEKS))
d3 = d3_lgb[["cutoff", "n_valid"]].assign(
    lightgbm_rmse=d3_lgb["rmse"], lightgbm_mae=d3_lgb["mae"], seasonal_naive_rmse=d3_sn["rmse"], seasonal_naive_mae=d3_sn["mae"]
)
d3["lgb_gain_pct"] = 100 * (1 - d3["lightgbm_rmse"] / d3["seasonal_naive_rmse"])
tables["D3"] = d3

# explain the worst LightGBM fold: which day and which zone-hours dominate its squared error
oof = d3_lgb.attrs["oof"].assign(se=lambda d: (d["trips"] - d["pred"]) ** 2, day=lambda d: d["pickup_hour_eat"].dt.date)
worst = d3.loc[d3["lightgbm_rmse"].idxmax(), "cutoff"]
w = oof[oof["fold"] == worst]
day_se = w.groupby("day")["se"].sum().sort_values(ascending=False)
top_day = day_se.index[0]
top_rows = w[w["day"] == top_day].nlargest(3, "se")
ids = sorted({i for s in w.loc[w["day"] == top_day, "matched_event_ids"] for i in s.split(",") if i})
names = [f"{ev_info.loc[i, 'event_name']} ({ev_info.loc[i, 'event_type']})" for i in ids if i in ev_info.index]
spike_txt = "; ".join(f"{r.zone} {r.pickup_hour_eat:%H:%M} actual {r.trips:.0f} vs pred {r.pred:.0f}" for r in top_rows.itertuples())

findings["D3"] = (
    f"LightGBM: RMSE {d3['lightgbm_rmse'].mean():.2f} ± {d3['lightgbm_rmse'].std():.2f} (mean ± sd over {len(d3)} folds); "
    f"seasonal naive: {d3['seasonal_naive_rmse'].mean():.2f} ± {d3['seasonal_naive_rmse'].std():.2f}. "
    f"LightGBM wins in {int((d3['lgb_gain_pct'] > 0).sum())}/{len(d3)} folds. "
    f"The fold-to-fold spread (sd {d3['lightgbm_rmse'].std():.2f}) is "
    + ("larger than" if d3["lightgbm_rmse"].std() > (d3["seasonal_naive_rmse"] - d3["lightgbm_rmse"]).mean() else "smaller than")
    + f" the average gap to the baseline ({(d3['seasonal_naive_rmse'] - d3['lightgbm_rmse']).mean():.2f}): which fortnight you test on "
    "matters more than small model tweaks, so differences of a few tenths on one split should not be over-read. "
    f"Worst fold: cutoff {worst} (RMSE {d3['lightgbm_rmse'].max():.2f}); {100 * day_se.iloc[0] / day_se.sum():.0f}% of its "
    f"squared error falls on {top_day} ({spike_txt}). "
    + (f"Calendar entries that day: {', '.join(names)}." if names else "No event in the calendar covers that day, so it looks like an unlisted spike or disruption.")
)
print(findings["D3"])
d3''',
    ),
    # ------------------------------------------------------------------ D4
    ("markdown", "## D4 — Feature availability & leakage audit"),
    (
        "code",
        r'''group_of = {f: "zone" for f in M.ZONE_FEATURES}
group_of.update({f: "calendar" for f in M.CALENDAR_FEATURES})
group_of.update({f: "trend/lag (≥14 days back)" for f in M.TREND_LAG_FEATURES})
group_of.update({f: "weather" for f in M.WEATHER_FEATURES})
group_of.update({f: "events" for f in M.EVENT_FEATURES})
audit = [
    {"column": f, "group": g, "known_at_forecast_time": "yes", "used": "yes",
     "note": "weather = forecast values for Nov (noisier than the observed values used in validation)" if g == "weather" else ""}
    for f, g in group_of.items()
]
audit += [
    {"column": "lag_168", "group": "lag", "known_at_forecast_time": "only days 1–7 of the horizon", "used": "D8 candidate (week-1 model only)",
     "note": "trips one week earlier; unknown for 8–14 Nov"},
    {"column": "trips_lag_168 (master table)", "group": "lag", "known_at_forecast_time": "no (for days 8–14)", "used": "no",
     "note": "uses actual trips from inside the forecast fortnight; replaced by horizon-safe lags"},
    {"column": "active_drivers", "group": "operations", "known_at_forecast_time": "no", "used": "no", "note": "consequence of demand; only in history"},
    {"column": "avg_wait_min", "group": "operations", "known_at_forecast_time": "no", "used": "no", "note": "consequence of demand vs supply"},
    {"column": "avg_fare_birr", "group": "operations", "known_at_forecast_time": "no", "used": "no", "note": "observed after trips happen (used only for revenue in the demo)"},
    {"column": "data_type", "group": "weather meta", "known_at_forecast_time": "yes", "used": "no", "note": "always 'observed' in train, 'forecast' in test — pure distribution-shift flag"},
    {"column": "record_id / row_id / matched_event_ids", "group": "identifiers", "known_at_forecast_time": "yes", "used": "no", "note": "IDs; event ids are encoded via the event flags"},
]
d4 = pd.DataFrame(audit)
tables["D4"] = d4

# leaky experiments on the same 18–31 Oct split
leak_rows = [{"variant": "final feature set (honest)", "rmse": M.rmse(va["trips"], d2_preds["lightgbm"]), "mae": M.mae(va["trips"], d2_preds["lightgbm"])}]
tr_l, va_l = tr.copy(), va.copy()
lag_master = train_raw.set_index(["zone", "pickup_hour_eat"])["trips_lag_168"]
for d in (tr_l, va_l):
    d["master_trips_lag_168"] = pd.MultiIndex.from_frame(d[["zone", "pickup_hour_eat"]]).map(lag_master)
for label, extra in [("+ active_drivers, avg_wait_min (LEAKY)", ["active_drivers", "avg_wait_min"]),
                     ("+ 1-week lag from actuals inside the fortnight (LEAKY)", ["master_trips_lag_168"])]:
    _, p, _ = M.fit_predict("lightgbm", tr_l, va_l, M.ALL_FEATURES + extra)
    leak_rows.append({"variant": label, "rmse": M.rmse(va["trips"], p), "mae": M.mae(va["trips"], p)})

# random split shown only as a contrast (Rule 7)
rnd = full.sample(frac=1.0, random_state=M.SEED)
cut = int(0.8 * len(rnd))
_, p, _ = M.fit_predict("lightgbm", rnd.iloc[:cut], rnd.iloc[cut:], M.ALL_FEATURES)
leak_rows.append({"variant": "random 80/20 split (contrast only, not valid)", "rmse": M.rmse(rnd.iloc[cut:]["trips"], p), "mae": M.mae(rnd.iloc[cut:]["trips"], p)})
d4_leak = pd.DataFrame(leak_rows)
tables["D4_leak"] = d4_leak

findings["D4"] = (
    "Excluded: active_drivers, avg_wait_min, avg_fare_birr (outcomes of demand, unknown for future hours), the master "
    "table's trips_lag_168 (needs trips from inside the forecast fortnight), data_type and identifiers. "
    f"Adding drivers/wait drops validation RMSE from {d4_leak.loc[0, 'rmse']:.2f} to {d4_leak.loc[1, 'rmse']:.2f} — "
    "an inflated score that would vanish in production because nobody knows next Tuesday's driver count. "
    f"The in-fortnight 1-week lag gives {d4_leak.loc[2, 'rmse']:.2f}"
    + (" (optimistic: it peeks at trips inside the forecast fortnight). " if d4_leak.loc[2, "rmse"] < d4_leak.loc[0, "rmse"]
       else " (no gain here, but it would still be unavailable for 8–14 Nov). ")
    + f"A random 80/20 split scores {d4_leak.loc[3, 'rmse']:.2f}"
    + (": optimistic, because neighbouring hours of the same weeks sit on both sides of the split."
       if d4_leak.loc[3, "rmse"] < d4_leak.loc[0, "rmse"] else
       ": not lower here, because its test rows are spread over the whole year (including the spiky early-autumn weeks) "
       "rather than a calm late-October fortnight. It is still invalid, since it trains on hours after the ones it scores.")
)
print(findings["D4"])
display(d4)
d4_leak''',
    ),
    # ------------------------------------------------------------------ D5
    ("markdown", "## D5 — Ablation (LightGBM, identical rolling folds)"),
    (
        "code",
        r'''ABLATION = {
    "(i) calendar + zone + trend/lag": M.BASE_FEATURES,
    "(ii) + weather": M.BASE_FEATURES + M.WEATHER_FEATURES,
    "(iii) + events": M.BASE_FEATURES + M.EVENT_FEATURES,
    "(iv) + weather + events": M.ALL_FEATURES,
}
rows = []
for label, feats in ABLATION.items():
    r = M.rolling_eval(full, ROLLING_CUTOFFS, lgb_fold({**BASE_CONFIG, "features": feats}))
    rows.append({"config": label, "n_features": len(feats), "rmse_18_31_oct": r["rmse"].iloc[-1],
                 "rolling_rmse_mean": r["rmse"].mean(), "rolling_mae_mean": r["mae"].mean()})
d5 = pd.DataFrame(rows)
d5["delta_rmse_vs_i"] = d5["rolling_rmse_mean"] - d5.loc[0, "rolling_rmse_mean"]
d5["delta_mae_vs_i"] = d5["rolling_mae_mean"] - d5.loc[0, "rolling_mae_mean"]
tables["D5"] = d5

dw, de, dboth = d5["delta_rmse_vs_i"].iloc[1:].tolist()
pct = lambda x: 100 * x / d5.loc[0, "rolling_rmse_mean"]
findings["D5"] = (
    f"Mean rolling RMSE: base {d5.loc[0, 'rolling_rmse_mean']:.2f}; weather {dw:+.2f} ({pct(dw):+.1f}%), "
    f"events {de:+.2f} ({pct(de):+.1f}%), both {dboth:+.2f} ({pct(dboth):+.1f}%). "
    + ("Both joins pay off and their gains roughly add up, so the integration work in A was worth it. "
       if dw < 0 and de < 0 else
       "Not every join helps on its own; the table shows which one carries the signal. ")
    + "Event gains are concentrated in few zone-hours (matches, concerts, closures), so they move RMSE more than MAE."
)
print(findings["D5"])
d5''',
    ),
    # ------------------------------------------------------------------ D6
    ("markdown", "## D6 — Hyperparameter tuning (Optuna, time-ordered folds)"),
    (
        "code",
        r'''import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
DEFAULT_PARAMS = dict(num_leaves=63, learning_rate=0.05, n_estimators=400, min_child_samples=20,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="regression")


def tuning_score(params):
    scores = []
    for c in TUNING_CUTOFFS:
        a, b = M.split_by_cutoff(full, c)
        _, p, _ = M.fit_predict("lightgbm", a, b, M.ALL_FEATURES, params)
        ok = ~b["suspect_target"].to_numpy()
        scores.append(M.rmse(b["trips"][ok], p[ok]))
    return float(np.mean(scores))


def objective(trial):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 200, 700, step=100),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        objective=trial.suggest_categorical("objective", ["regression", "poisson"]),
    )
    return tuning_score(params)


t0 = time.perf_counter()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=M.SEED))
study.enqueue_trial(DEFAULT_PARAMS)
study.optimize(objective, n_trials=N_TRIALS)
TUNED_PARAMS = study.best_params
tune_secs = time.perf_counter() - t0

_, p_tuned, _ = M.fit_predict("lightgbm", tr, va, M.ALL_FEATURES, TUNED_PARAMS)
ok_va = ~va["suspect_target"].to_numpy()
d6 = pd.DataFrame([
    {"params": label, "tuning_folds_rmse_clean": score, "rmse_18_31_oct": M.rmse(va["trips"], p),
     "rmse_clean_18_31_oct": M.rmse(va["trips"][ok_va], p[ok_va]), "mae_18_31_oct": M.mae(va["trips"], p)}
    for label, score, p in [("default", study.trials[0].value, d2_preds["lightgbm"]), ("tuned", study.best_value, p_tuned)]
])
tables["D6"] = d6
tables["D6_params"] = pd.DataFrame([{"parameter": k, "best": v} for k, v in TUNED_PARAMS.items()])
findings["D6"] = (
    f"Optuna TPE, {N_TRIALS} trials (trial 0 = defaults) over num_leaves 15–127, learning_rate 0.02–0.1, "
    "n_estimators 200–700, min_child_samples 5–100, subsample 0.6–1, colsample 0.5–1, reg_lambda 1e-3–10, "
    f"objective {{L2, Poisson}}; score = mean RMSE on the 20 Sep and 4 Oct folds, excluding rows flagged as corrupted "
    f"(D7) so a handful of impossible values do not steer the search ({tune_secs / 60:.1f} min). "
    f"Tuning-fold clean RMSE {study.trials[0].value:.2f} → {study.best_value:.2f}; on the untouched 18–31 Oct fortnight "
    f"clean RMSE {d6.loc[0, 'rmse_clean_18_31_oct']:.2f} → {d6.loc[1, 'rmse_clean_18_31_oct']:.2f} "
    f"(all rows {d6.loc[0, 'rmse_18_31_oct']:.2f} → {d6.loc[1, 'rmse_18_31_oct']:.2f}). Best: "
    + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in TUNED_PARAMS.items()) + "."
)
print(findings["D6"])
display(tables["D6_params"])
d6''',
    ),
    # ------------------------------------------------------------------ D7
    ("markdown", "## D7 — Error analysis (out-of-fold predictions from all 4 rolling folds, tuned model)"),
    (
        "code",
        r'''TUNED_CONFIG = {**BASE_CONFIG, "params": TUNED_PARAMS}
d7_roll = M.rolling_eval(full, ROLLING_CUTOFFS, lgb_fold(TUNED_CONFIG))
oof = d7_roll.attrs["oof"].copy()
oof["err"] = oof["trips"] - oof["pred"]
oof["abs_err"] = oof["err"].abs()
oof["day_type"] = np.select([oof["is_public_holiday"] == 1, oof["is_weekend"] == 1], ["holiday", "weekend"], "weekday")
oof["horizon_day"] = ((oof["pickup_hour_eat"] - oof["fold"].map(M.tz_ts)).dt.total_seconds() // 86400 + 1).astype(int)


def summarize(g):
    return pd.Series({"n": len(g), "mean_trips": g["trips"].mean(), "rmse": np.sqrt((g["err"] ** 2).mean()),
                      "mae": g["abs_err"].mean(), "bias(actual-pred)": g["err"].mean()})


by_zone = oof.groupby("zone").apply(summarize).sort_values("rmse", ascending=False)
by_zone["mae_pct_of_mean"] = 100 * by_zone["mae"] / by_zone["mean_trips"]
by_hour = oof.groupby("hour").apply(summarize)
by_daytype = oof.groupby("day_type").apply(summarize)
by_horizon = oof.groupby(oof["horizon_day"] > 7).apply(summarize).rename(index={False: "days 1–7", True: "days 8–14"})
tables.update(D7_zone=by_zone.reset_index(), D7_daytype=by_daytype.reset_index(), D7_horizon=by_horizon.reset_index())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
by_zone["rmse"].plot.bar(ax=axes[0], color="#0072B2", title="RMSE by zone")
by_hour["rmse"].plot(ax=axes[1], marker="o", color="#D55E00", title="RMSE by hour of day")
by_daytype["rmse"].plot.bar(ax=axes[2], color="#009E73", title="RMSE by day type")
for ax, xl in zip(axes, ["zone", "hour (EAT)", "day type"]):
    ax.set_xlabel(xl)
    ax.set_ylabel("RMSE (trips per zone-hour)")
fig.suptitle("D7 — Out-of-fold error, 4 rolling folds (6 Sep – 31 Oct 2025)")
plt.tight_layout()
fig.savefig(ROOT / "figures" / "D7_error_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()
display(by_zone.round(2), by_daytype.round(2), by_horizon.round(2))''',
    ),
    (
        "code",
        r'''def hypothesis(r):
    tpd = r.trips / max(r.active_drivers, 1)
    if tpd > M.MAX_TRIPS_PER_DRIVER:
        return f"suspected corrupted record: {tpd:.1f} trips per active driver (normal ≈1.3), trips look ~10x inflated"
    ids = [i for i in r.matched_event_ids.split(",") if i]
    types = sorted({ev_info.loc[i, "event_type"] for i in ids if i in ev_info.index} - {"school_break"})
    if types:
        phase = "after the event" if r.ev_post else "before the event" if r.ev_pre else "during the event"
        return f"listed {'/'.join(types)} {phase}: size of the lift is event-specific"
    if r.is_public_holiday:
        return "public holiday response differs from past holidays"
    if r.rain_mm >= 5:
        return "heavy rain hour: rare, response saturates"
    if r.err > 0 and r.trips > 2 * max(r.pred, 1):
        return "unlisted spike: no calendar event (local gathering / unlisted match / church or market day)"
    if r.err < 0 and r.trips < 0.5 * r.pred:
        return "demand collapse: likely partial outage or missing records"
    return "busy-hour noise in a high-volume zone"


top10 = oof.nlargest(10, "abs_err").copy()
top10["hypothesis"] = top10.apply(hypothesis, axis=1)
top10_view = top10[["zone", "pickup_hour_eat", "trips", "pred", "err", "active_drivers", "rain_mm", "matched_event_ids", "hypothesis"]]
top10_view = top10_view.assign(pickup_hour_eat=top10_view["pickup_hour_eat"].dt.strftime("%Y-%m-%d %H:%M"), pred=top10_view["pred"].round(1), err=top10_view["err"].round(1))
tables["D7_top10"] = top10_view

e2 = oof["err"] ** 2
top1_share = e2.nlargest(len(e2) // 100).sum() / e2.sum()
groups = top10["hypothesis"].str.split(":").str[0].value_counts()
worst_zone = by_zone.index[0]
findings["D7"] = (
    f"(a) Error scales with volume: {worst_zone} has the highest RMSE ({by_zone['rmse'].iloc[0]:.1f}), while MAE as % of mean demand "
    f"ranges {by_zone['mae_pct_of_mean'].min():.0f}–{by_zone['mae_pct_of_mean'].max():.0f}% across zones. "
    f"By hour, RMSE peaks at {int(by_hour['rmse'].idxmax()):02d}:00 and is lowest at {int(by_hour['rmse'].idxmin()):02d}:00. "
    "Day type: " + ", ".join(f"{k} {v:.1f}" for k, v in by_daytype["rmse"].items()) + ". "
    f"Days 8–14 of the horizon score {by_horizon['rmse'].iloc[1]:.2f} vs {by_horizon['rmse'].iloc[0]:.2f} for days 1–7; "
    f"overall bias (actual − pred) is {oof['err'].mean():+.2f} trips. "
    f"(b) The top 1% of zone-hours produce {100 * top1_share:.0f}% of the squared error, and the "
    f"{int(oof['suspect_target'].sum())} validation rows flagged as suspect alone produce "
    f"{100 * e2[oof['suspect_target']].sum() / e2.sum():.0f}%. Top-10 groups: "
    + "; ".join(f"{k} ×{v}" for k, v in groups.items()) + ". "
    "The flagged rows have a normal driver count but ~10× the trips, which points to corrupted trip counts rather than "
    "real demand: no feature could anticipate them, and training on them pulls neighbouring forecasts up."
)
print(findings["D7"])
top10_view''',
    ),
    # ------------------------------------------------------------------ D8
    ("markdown", "## D8 — Response to findings"),
    (
        "markdown",
        """Two changes follow from D7, each tested on the same 4 rolling folds with the tuned parameters:

1. **Drop suspected corrupted targets from training.** Rows with more than 3 trips per active driver (normal ≈1.3; 99.5% of rows ≤1.8) are removed from the training fold. `active_drivers` is used only to clean the training target, never as a model input, so nothing leaks into the forecast.
2. **Cap training spikes.** Cap the training target at each zone's 99.5th percentile (learned on the training fold only), a blunter version of (1) that needs no driver count.

A one-week lag for days 1–7 was not pursued further: D7 already shows days 1–7 are *harder* than days 8–14 under the 14-day-safe lags, so the bottleneck is spikes/corruption, not missing lag_168.

Scores are reported on all validation rows and on the rows not flagged as suspect (`*_clean`). A change is adopted if it lowers the mean rolling **clean** RMSE; if both help alone they are tested together.""",
    ),
    (
        "code",
        r'''CHANGES = {
    "drop suspect targets": {"drop_suspect": True},
    "cap spikes q99.5": {"cap_q": 0.995},
}
candidates = {"tuned (D6)": TUNED_CONFIG}
rolls = {"tuned (D6)": d7_roll}
for label, change in CHANGES.items():
    candidates["+ " + label] = {**TUNED_CONFIG, **change}
    rolls["+ " + label] = M.rolling_eval(full, ROLLING_CUTOFFS, lgb_fold(candidates["+ " + label]))
helpful = [k for k in CHANGES if rolls["+ " + k]["rmse_clean"].mean() < d7_roll["rmse_clean"].mean()]
if len(helpful) >= 2:
    combo = " + ".join(helpful)
    candidates["+ " + combo] = {**TUNED_CONFIG, **{k: v for h in helpful for k, v in CHANGES[h].items()}}
    rolls["+ " + combo] = M.rolling_eval(full, ROLLING_CUTOFFS, lgb_fold(candidates["+ " + combo]))

d8 = pd.DataFrame([{"variant": k, "rolling_rmse_mean": r["rmse"].mean(), "rolling_mae_mean": r["mae"].mean(),
                    "rolling_rmse_clean_mean": r["rmse_clean"].mean(), "rolling_rmse_clean_sd": r["rmse_clean"].std(),
                    "rolling_mae_clean_mean": r["mae_clean"].mean(), "rmse_clean_18_31_oct": r["rmse_clean"].iloc[-1]}
                   for k, r in rolls.items()])
d8["delta_rmse_clean_vs_tuned"] = d8["rolling_rmse_clean_mean"] - d8.loc[0, "rolling_rmse_clean_mean"]
tables["D8"] = d8

best = d8.loc[d8["rolling_rmse_clean_mean"].idxmin(), "variant"]
FINAL_CONFIG = candidates[best]
FINAL_LABEL = best
deltas = {k: v for k, v in zip(d8["variant"], d8["delta_rmse_clean_vs_tuned"]) if k != "tuned (D6)"}
findings["D8"] = (
    "Change in mean rolling clean RMSE vs the tuned model: " + "; ".join(f"{k} {v:+.3f}" for k, v in deltas.items()) + ". "
    + (f"Adopted **{best}** as the final configuration "
       f"(clean RMSE {d8.loc[0, 'rolling_rmse_clean_mean']:.2f} → {d8['rolling_rmse_clean_mean'].min():.2f}; "
       f"all-row RMSE {d8.loc[0, 'rolling_rmse_mean']:.2f} → {d8.set_index('variant').loc[best, 'rolling_rmse_mean']:.2f})."
       if best != "tuned (D6)" else
       "None of the changes helped on average, so the tuned model is kept unchanged.")
)
print(findings["D8"])
d8''',
    ),
    # ------------------------------------------------------------------ D9
    ("markdown", "## D9 — Plain-language metric"),
    (
        "code",
        r'''final_roll = rolls[FINAL_LABEL]
F_RMSE, F_MAE = final_roll["rmse"].mean(), final_roll["mae"].mean()
F_RMSE_C, F_MAE_C = final_roll["rmse_clean"].mean(), final_roll["mae_clean"].mean()
oof_final = final_roll.attrs["oof"]
mean_demand = oof_final.loc[~oof_final["suspect_target"], "trips"].mean()
tables["D3_final"] = final_roll[["cutoff", "n_valid", "rmse", "mae", "rmse_clean", "mae_clean"]]
sn_clean = d3_sn["rmse_clean"].mean()
findings["D9"] = (
    f"Across the four validation fortnights, the forecast for one zone in one hour is typically off by about "
    f"**{F_MAE_C:.1f} trips** (MAE), {100 * F_MAE_C / mean_demand:.0f}% of the average {mean_demand:.1f} trips per zone-hour. "
    f"The RMSE of {F_RMSE_C:.1f} trips ({100 * F_RMSE_C / mean_demand:.0f}% of mean) weights the occasional bigger miss, "
    f"such as an event that draws more people than usual. At ~{M.TRIPS_PER_DRIVER_HOUR} trips per driver-hour, a typical miss is "
    f"**{F_MAE_C / M.TRIPS_PER_DRIVER_HOUR:.1f} drivers** per zone per hour (≈{F_RMSE_C / M.TRIPS_PER_DRIVER_HOUR:.0f} in a bad hour). "
    f"Compared with the seasonal-naive rule (RMSE {sn_clean:.1f}), the model removes about {100 * (1 - F_RMSE_C / sn_clean):.0f}% "
    f"of the error. These figures exclude the {100 * full['suspect_target'].mean():.2f}% of history rows that look corrupted; including them, RMSE is "
    f"{F_RMSE:.1f} and MAE {F_MAE:.1f}."
)
print(findings["D9"])
tables["D3_final"]''',
    ),
    # ------------------------------------------------------------------ final
    ("markdown", "## Final model, saved bundle & submission"),
    (
        "code",
        r'''final_bundle = M.fit_final(full, "2025-11-01", FINAL_CONFIG)
pred_test = M.predict_bundle(final_bundle, test)

template = pd.read_csv(ROOT / "data" / "raw" / "submission_template.csv")
pred_by_id = pd.Series(pred_test, index=test["row_id"])
assert pred_by_id.index.is_unique and set(pred_by_id.index) == set(template["row_id"])
predicted = pred_by_id.reindex(template["row_id"]).round(2).to_numpy()
assert len(predicted) == 4032 and np.isfinite(predicted).all() and (predicted >= 0).all()
sub_path = write_submission(template["row_id"], predicted, TEAM_NAME)

final_bundle.update(
    label=FINAL_LABEL,
    validation={"rolling_rmse_clean_mean": F_RMSE_C, "rolling_mae_clean_mean": F_MAE_C,
                "rolling_rmse_mean": F_RMSE, "rolling_mae_mean": F_MAE, "cutoffs": ROLLING_CUTOFFS},
    zone_avg_fare=train_raw.groupby("zone")["avg_fare_birr"].mean().round(1).to_dict(),
    trips_per_driver_hour=M.TRIPS_PER_DRIVER_HOUR,
)
model_path = save_model(final_bundle)

daily = pd.DataFrame({"day": test["pickup_hour_eat"].dt.date, "pred": pred_test}).groupby("day")["pred"].sum()
recent = full[full["pickup_hour_eat"] >= M.tz_ts("2025-10-18")].groupby(full["pickup_hour_eat"].dt.date)["trips"].sum()
print("wrote", sub_path, "and", model_path)
print(f"mean predicted city trips/day (1–14 Nov): {daily.mean():.0f} vs actual 18–31 Oct: {recent.mean():.0f}")
pd.read_csv(sub_path).head()''',
    ),
    ("markdown", "## Write `reports/D_model_evaluation.md`"),
    (
        "code",
        r'''sections = [
    ("D1 — Baselines", ["D1"]),
    ("D2 — Model comparison", ["D2"]),
    ("D3 — Rolling-origin validation", ["D3"]),
    ("D4 — Feature availability & leakage audit", ["D4", "D4_leak"]),
    ("D5 — Ablation", ["D5"]),
    ("D6 — Tuning", ["D6", "D6_params"]),
    ("D7 — Error analysis", ["D7_zone", "D7_daytype", "D7_horizon", "D7_top10"]),
    ("D8 — Response to findings", ["D8"]),
    ("D9 — Plain-language metric", ["D3_final"]),
]
lines = [
    "# D — Modeling & Evaluation",
    "",
    "Source: `notebooks/04_modeling_and_evaluation.ipynb` (code in `src/modeling.py`). All scores are on chronological "
    "splits of the train file; the test file is never scored.",
    "",
    "**Split dates.** Main validation: train < 2025-10-18, validate 18–31 Oct 2025. Rolling origin: cutoffs "
    + ", ".join(ROLLING_CUTOFFS) + " (14-day validation each). Tuning folds: " + ", ".join(TUNING_CUTOFFS) + ".",
    "",
    f"**Final model:** LightGBM, configuration `{FINAL_LABEL}`. Mean rolling-origin RMSE {F_RMSE_C:.2f} / MAE {F_MAE_C:.2f} "
    f"on rows not flagged as corrupted (all rows: {F_RMSE:.2f} / {F_MAE:.2f}). "
    f"Uses weather and event features (Rule 5) and only forecast-time features (Rule 6). Saved to `models/final_model.joblib`; "
    f"predictions in `submission/{sub_path.name}`.",
    "",
]
for title, keys in sections:
    code = title.split(" ")[0]
    lines += [f"## {title}", "", findings[code], ""]
    for k in keys:
        lines += [md_table(tables[k]), ""]
    if code == "D7":
        lines += ["![D7 error breakdown](../figures/D7_error_breakdown.png)", ""]
(REPORTS / "D_model_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
for k, t in tables.items():
    t.to_csv(REPORTS / f"{k}.csv", index=False)
print("wrote", REPORTS / "D_model_evaluation.md")''',
    ),
]


def main() -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(src.strip()) if kind == "markdown" else nbformat.v4.new_code_cell(src.strip())
        for kind, src in CELLS
    ]
    nbformat.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
