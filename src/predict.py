"""Generate predictions for the test master table / submission file."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
TEMPLATE = ROOT / "data" / "raw" / "submission_template.csv"


def write_submission(row_ids, predicted_trips, team_name: str = "NAME") -> Path:
    out = pd.DataFrame({"row_id": row_ids, "predicted_trips": predicted_trips})
    path = SUBMISSION / f"team_{team_name}_submission.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


# Prediction pipeline lives in `src/modeling.py` (`predict_bundle`) and notebook 04.
