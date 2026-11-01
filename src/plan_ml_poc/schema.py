"""Explicit feature roles and units shared by training and inference.

Angles use an arbitrary POC convention: positive coronal angles mean varus;
positive slope means posterior slope. These are not Kinomatic export conventions.
"""

import numpy as np
import pandas as pd

X_COLUMNS = ["age", "bmi", "baseline_function", "preop_hka_deg", "mldfa_deg", "mpta_deg"]
W_COLUMNS = ["surgeon_b", "hospital_b"]
PLAN_COLUMNS = ["femoral_varus_deg", "tibial_varus_deg", "tibial_slope_deg"]
OUTCOME = "synthetic_fjs_12mo"
REFERENCE = np.array([0.0, 0.0, 3.0])
# Teaching search bounds, NOT clinically approved ranges.
BOUNDS = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 7.0]])
X_CENTER = np.array([65.0, 29.0, 50.0, 0.0, 88.0, 88.0])
X_SCALE = np.array([10.0, 5.0, 15.0, 4.0, 3.0, 3.0])


def numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Select by name (never incidental column order); reject missing/nonfinite data."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Inputs must be finite; choose a missing-data strategy before fitting.")
    return values


def patient_features(frame: pd.DataFrame) -> np.ndarray:
    # Fixed domain scaling keeps fold-specific learned preprocessing out of this example.
    return (numeric(frame, X_COLUMNS) - X_CENTER) / X_SCALE


def validate_cases(frame: pd.DataFrame) -> None:
    numeric(frame, X_COLUMNS + W_COLUMNS + PLAN_COLUMNS + [OUTCOME])
    if "patient_id" not in frame or frame.patient_id.isna().any():
        raise ValueError("Every case needs a patient_id for grouped splits.")
    if not frame[W_COLUMNS].isin([0, 1]).all().all():
        raise ValueError("This POC expects binary surgeon/hospital indicators.")
    if not frame[OUTCOME].between(0, 100).all():
        raise ValueError("Synthetic FJS-like outcome must be on the 0–100 scale.")
