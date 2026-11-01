"""Synthetic-only causal diagnostics on untouched, fixed contrasts."""

import numpy as np
import pandas as pd

from .schema import REFERENCE, patient_features
from .synthetic import treatment_response
from .training import ModelBundle


def oracle_diagnostics(bundle: ModelBundle, test: pd.DataFrame) -> dict[str, float]:
    """Pointwise coverage on a prespecified contrast, never on optimized choices.

    This metric is available only for synthetic data. It cannot be computed
    from ordinary factual clinical outcomes. It is not simultaneous coverage.
    """
    reference = np.tile(REFERENCE, (len(test), 1))
    candidate = np.tile([1.0, 0.5, 4.0], (len(test), 1))
    x = patient_features(test)
    truth = treatment_response(test, candidate) - treatment_response(test, reference)
    effect = bundle.causal.effect(x, T0=reference, T1=candidate)
    lower, upper = bundle.causal.effect_interval(x, T0=reference, T1=candidate)
    return {
        "oracle_effect_mae": float(np.mean(np.abs(effect - truth))),
        "fixed_contrast_pointwise_coverage": float(np.mean((lower <= truth) & (truth <= upper))),
    }
