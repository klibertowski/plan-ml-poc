"""An invented world with known counterfactual means, never clinical evidence.

The response is intentionally quadratic in treatment and linear in effect
modifiers, matching our teaching estimator. This makes recovery easier than
in real data. Noise is bounded to avoid clipping the response surface.
"""

import numpy as np
import pandas as pd

from .schema import PLAN_COLUMNS, W_COLUMNS, patient_features


def treatment_response(cases: pd.DataFrame, plans: np.ndarray) -> np.ndarray:
    """The simulator's treatment contribution. Only diagnostics may call this."""
    x = patient_features(cases)
    f, t, slope = np.asarray(plans, dtype=float).T
    s = slope - 3.0
    return (
        (0.8 + 0.5 * x[:, 3]) * f
        + (0.5 - 0.4 * x[:, 5]) * t
        + (0.4 + 0.2 * x[:, 1]) * s
        - 0.7 * f**2
        - 0.6 * t**2
        - 0.4 * s**2
        + 0.25 * f * t
    )


def true_mean(cases: pd.DataFrame, plans: np.ndarray) -> np.ndarray:
    """Oracle available only because we authored the simulator."""
    x = patient_features(cases)
    w = cases[W_COLUMNS].to_numpy()
    baseline = 73 + 4 * x[:, 2] - 2 * x[:, 1] - 1.5 * x[:, 0] + 3 * w[:, 0] - 2 * w[:, 1]
    return baseline + treatment_response(cases, plans)


def make_cases(n: int = 3000, seed: int = 42, hidden_confounding: float = 0.0) -> pd.DataFrame:
    """Generate observational assignments influenced by measured X and W.

    hidden_confounding > 0 adds an unobserved common cause of T and Y, letting
    notebook 04 demonstrate how identification can fail despite runnable code.
    The hidden variable and oracle means are deliberately absent from the table.
    """
    if n < 30 or hidden_confounding < 0:
        raise ValueError("Use at least 30 cases and nonnegative hidden confounding.")
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.5, 1.5, size=(n, 6))
    cases = pd.DataFrame(
        {
            "patient_id": [f"synthetic-{i:05d}" for i in range(n)],
            "age": 65 + 10 * z[:, 0],
            "bmi": 29 + 5 * z[:, 1],
            "baseline_function": 50 + 15 * z[:, 2],
            "preop_hka_deg": 4 * z[:, 3],
            "mldfa_deg": 88 + 3 * z[:, 4],
            "mpta_deg": 88 + 3 * z[:, 5],
            "surgeon_b": rng.integers(0, 2, n),
            "hospital_b": rng.integers(0, 2, n),
        }
    )
    # Each action has independent residual variation; no hard clipping masses.
    hidden = rng.uniform(-1, 1, n)
    shifts = np.column_stack(
        [
            0.35 * z[:, 3] + 0.35 * cases.surgeon_b + hidden_confounding * hidden,
            -0.3 * z[:, 5] + 0.2 * cases.hospital_b,
            0.25 * z[:, 1] - 0.2 * cases.surgeon_b,
        ]
    )
    raw = rng.normal(size=(n, 3)) + shifts
    plans = np.column_stack(
        [3 * np.tanh(raw[:, 0]), 3 * np.tanh(raw[:, 1]), 3.5 + 3.5 * np.tanh(raw[:, 2])]
    )
    cases[PLAN_COLUMNS] = plans
    cases["synthetic_fjs_12mo"] = (
        true_mean(cases, plans) + rng.uniform(-3, 3, n) + 4 * hidden_confounding * hidden
    )
    return cases
