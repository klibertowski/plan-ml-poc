"""Patient + candidate -> constraints, support, outcome, effect, objective."""

import numpy as np
import pandas as pd

from .schema import BOUNDS, PLAN_COLUMNS, REFERENCE, W_COLUMNS, numeric, patient_features
from .training import ModelBundle, outcome_features


def feasible(plans: np.ndarray) -> np.ndarray:
    """Toy angle and coupled constraints; this is NOT an implant geometry engine.

    A production adapter must check meshes, cuts, fit, gaps, collisions, implant
    sizes, surgeon preferences and biomechanics before model evaluation.
    """
    plans = np.asarray(plans, dtype=float)
    finite = np.isfinite(plans).all(axis=1)
    inside = ((plans >= BOUNDS[:, 0]) & (plans <= BOUNDS[:, 1])).all(axis=1)
    coupled = np.abs(plans[:, 0] + plans[:, 1]) <= 4.0
    return finite & inside & coupled


class PlanEvaluator:
    def __init__(
        self,
        bundle: ModelBundle,
        patient: pd.DataFrame,
        reference: np.ndarray = REFERENCE,
        movement_penalty: float = 0.1,
    ):
        if len(patient) != 1 or movement_penalty < 0:
            raise ValueError("Supply one patient and a nonnegative movement penalty.")
        patient_features(patient)
        if not np.isin(numeric(patient, W_COLUMNS), [0, 1]).all():
            raise ValueError("Unknown surgeon/hospital context; POC expects binary indicators.")
        self.bundle = bundle
        self.patient = patient.copy()
        self.reference = np.asarray(reference, dtype=float).reshape(1, 3)
        if not feasible(self.reference)[0]:
            raise ValueError("Reference violates the teaching constraints.")
        if not bundle.support.check(patient, self.reference)[0][0]:
            raise ValueError("Unsupported patient/reference; abstaining from plan search.")
        self.movement_penalty = movement_penalty
        self.reference_prediction = float(
            bundle.outcome.predict(outcome_features(patient, self.reference))[0]
        )

    def evaluate(self, plans: np.ndarray) -> pd.DataFrame:
        """Reject first; never assign model benefit to an unsupported candidate."""
        plans = np.asarray(plans, dtype=float)
        if plans.ndim != 2 or plans.shape[1] != 3 or not len(plans):
            raise ValueError("Expected a nonempty (n_candidates, 3) array.")
        result = pd.DataFrame(plans, columns=PLAN_COLUMNS)
        result["feasible"] = feasible(plans)
        result["supported"] = False
        result["support_distance"] = np.nan
        result["status"] = "rejected_constraints"
        result["synthetic_only"] = True
        columns = [
            "outcome_prediction",
            "effect",
            "ci_lower",
            "ci_upper",
            "anchored_outcome",
            "movement_cost",
            "utility",
        ]
        result[columns] = np.nan
        valid = np.flatnonzero(result.feasible.to_numpy())
        if not len(valid):
            return result
        supported, distance = self.bundle.support.check(self.patient, plans[valid])
        result.loc[valid, "supported"] = supported
        result.loc[valid, "support_distance"] = distance
        result.loc[valid, "status"] = "rejected_support"
        accepted = valid[supported]
        if not len(accepted):
            return result
        cases = pd.concat([self.patient] * len(accepted), ignore_index=True)
        x = patient_features(cases)
        candidates = plans[accepted]
        reference = np.repeat(self.reference, len(accepted), axis=0)
        effect = self.bundle.causal.effect(x, T0=reference, T1=candidates)
        lower, upper = self.bundle.causal.effect_interval(
            x, T0=reference, T1=candidates, alpha=0.05
        )
        prediction = self.bundle.outcome.predict(outcome_features(cases, candidates))
        cost = self.movement_penalty * np.square(
            (candidates - reference) / (BOUNDS[:, 1] - BOUNDS[:, 0])
        ).sum(axis=1)
        # The reference anchor is constant across candidates for this patient.
        # Thus ranking uses causal LCB minus cost; XGB cannot add a second benefit.
        result.loc[accepted, columns] = np.column_stack(
            [
                prediction,
                effect,
                lower,
                upper,
                self.reference_prediction + effect,
                cost,
                self.reference_prediction + lower - cost,
            ]
        )
        result.loc[accepted, "status"] = "accepted"
        return result
