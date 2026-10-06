"""Train and save the final forecasting model."""

from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def save_model(model, path: Path | None = None) -> Path:
    path = path or (MODELS / "final_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | None = None):
    path = path or (MODELS / "final_model.joblib")
    return joblib.load(path)


# TODO: baselines, LightGBM training, rolling-origin validation, ablation
