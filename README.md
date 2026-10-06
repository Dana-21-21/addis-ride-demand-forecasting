# Ride Minds — Addis Ride Demand Forecasting

Qiyas Data Science & AI Hackathon — forecast hourly ride trips for 12 zones in Addis Ababa (1–14 Nov 2025).

## Team

| | |
|---|---|
| **Team name** | Ride Minds |
| **Members** | Danayit Girma Admase; Zenaw Negatu Abay; Samuel Melaku Bekele; Kalkidan Mihretie Mengestie; Rejeb Dendir Bireda |

## Summary

We clean and join trip history, hourly weather, and city events into master tables, then forecast zone-hour trips for 1–14 November 2025 with LightGBM (calendar, 14-day-safe lags, weather, and event features). Validation uses chronological rolling fortnights only.

**Final validation (4 rolling folds, clean rows):** RMSE **9.60** / MAE **6.20** trips per zone-hour.  
**Submission:** `submission/team_ride_minds_submission.csv`

## What the demo does

An operations manager picks a **zone** and a **date**. The app returns:
- hourly trip forecast for that day
- peak hour
- estimated drivers needed (~ trips / 1.3)
- expected fare revenue
- weather/event context used for that day

## Project layout

```text
team_ride_minds/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # original CSVs (do not edit)
│   └── processed/           # master_train / master_test / dictionary
├── notebooks/
│   ├── 01_cleaning_and_integration.ipynb   # A
│   ├── 02_analysis_report.ipynb            # B
│   ├── 03_visualizations.ipynb             # C
│   └── 04_modeling_and_evaluation.ipynb    # D
├── src/                     # reusable pipeline code
├── models/                  # final_model.joblib
├── figures/                 # fig01 ... fig12 + captions
├── reports/                 # A / B / D write-ups
├── app/                     # Streamlit demo
├── presentation/            # 5 slides
└── submission/              # scored CSV
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order

1. `python scripts/build_masters.py` (or `notebooks/01_cleaning_and_integration.ipynb`) → `data/processed/`
2. `notebooks/02_analysis_report.ipynb` → `reports/B_analysis_report.md`
3. `notebooks/03_visualizations.ipynb` → `figures/`
4. `notebooks/04_modeling_and_evaluation.ipynb` → `models/`, `reports/D_…`, submission
5. Bundle demo assets (if needed): `python scripts/build_app_assets.py`
6. Demo:

```bash
streamlit run app/app.py
```

## Demo link

**Live app:** https://addis-ride-demand-forecasting-mjc8skkfdzkgtw6nstthdp.streamlit.app/

_Local (same app):_ `streamlit run app/app.py`

## Notes

- Never edit files in `data/raw/`.
- Use relative paths only.
- Validate with time-ordered splits (never random split for reported scores).
- Final model uses weather and event features, only information known at forecast time.
