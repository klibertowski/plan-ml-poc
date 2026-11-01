"""Save a trusted local fitted bundle plus readable provenance metadata."""

import importlib.metadata
import json
import platform
from pathlib import Path

import joblib

from .schema import BOUNDS, PLAN_COLUMNS, W_COLUMNS, X_COLUMNS
from .training import ModelBundle


def save_bundle(bundle: ModelBundle, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / "models.joblib")
    metadata = {
        "synthetic_only": bundle.synthetic_only,
        "seed": bundle.seed,
        "training_rows": bundle.training_rows,
        "x_columns": X_COLUMNS,
        "w_columns": W_COLUMNS,
        "plan_columns": PLAN_COLUMNS,
        "teaching_bounds_deg": BOUNDS.tolist(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["econml", "xgboost", "scikit-learn", "numpy"]
        },
        "causal_model": "LinearDML, quadratic treatment features, linear X heterogeneity",
        "warning": "Synthetic mechanics only. No clinical conclusions.",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_bundle(directory: str | Path) -> ModelBundle:
    """Load only files you created/trust: joblib can execute code when loading."""
    return joblib.load(Path(directory) / "models.joblib")
