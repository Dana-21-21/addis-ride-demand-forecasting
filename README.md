# Addis Ride Demand Forecasting

Qiyas Data Science & AI Hackathon — forecast hourly ride trips for 12 zones in Addis Ababa (1–14 Nov 2025).

## What this app does (user view)

An operations manager picks a **zone** and a **date**. The demo returns:
- hourly trip forecast for that day
- peak hour
- estimated drivers needed (~ trips / 1.3)
- expected fare revenue
- weather/event context used for that day

## Project layout

```text
team_NAME/
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

1. `notebooks/01_cleaning_and_integration.ipynb` → writes `data/processed/`
2. `notebooks/02_analysis_report.ipynb`
3. `notebooks/03_visualizations.ipynb` → writes `figures/`
4. `notebooks/04_modeling_and_evaluation.ipynb` → writes `models/` + submission
5. Demo:

```bash
streamlit run app/app.py
```

## Team roles (suggested)

| Role | Owns |
|------|------|
| Data | notebooks/01, data/processed, reports/A |
| Analysis + Viz | notebooks/02–03, figures, reports/B |
| Modeling | notebooks/04, models, submission, reports/D |
| Demo + packaging | app/, README, presentation |

## Demo link

_Local for now:_ `streamlit run app/app.py`  
_Public URL:_ _(add when hosted)_

## Notes

- Never edit files in `data/raw/`.
- Use relative paths only.
- Validate with time-ordered splits (never random split for reported scores).
- Final model must use ≥1 weather feature and ≥1 events feature.
