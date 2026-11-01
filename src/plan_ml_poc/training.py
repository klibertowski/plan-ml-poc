"""Offline fitting: hold out patients, cross-fit nuisances, freeze evaluators."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

from .schema import OUTCOME, PLAN_COLUMNS, W_COLUMNS, numeric, patient_features, validate_cases
from .support import SupportGuard


def split_cases(cases: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every procedure for a patient on one side of an 80/20 split."""
    validate_cases(cases)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, test = next(splitter.split(cases, groups=cases.patient_id))
    return cases.iloc[train].copy(), cases.iloc[test].copy()


def outcome_features(cases: pd.DataFrame, plans: np.ndarray) -> np.ndarray:
    return np.column_stack([patient_features(cases), numeric(cases, W_COLUMNS), plans])


@dataclass
class ModelBundle:
    outcome: XGBRegressor
    causal: LinearDML
    support: SupportGuard
    seed: int
    training_rows: int
    synthetic_only: bool = True


def fit_models(train: pd.DataFrame, seed: int = 42) -> ModelBundle:
    """XGB learns associations; DML learns a structured treatment contrast.

    LinearDML refers to the *final coefficient model*. Polynomial treatment
    features permit curved responses and pairwise angle interactions. At fixed
    X, tau(x, a0, a1) = theta(x) @ [phi(a1) - phi(a0)].
    W adjusts confounding but does not modify the final treatment effect here.
    """
    validate_cases(train)
    if train.patient_id.nunique() < 6:
        raise ValueError("Need at least six patient groups for this teaching fit.")
    x = patient_features(train)
    w = numeric(train, W_COLUMNS)
    t = numeric(train, PLAN_COLUMNS)
    y = numeric(train, [OUTCOME]).ravel()
    outcome = XGBRegressor(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.06,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=seed,
    )
    outcome.fit(outcome_features(train, t), y)
    # ExtraTrees handles the multi-output phi(T) nuisance regression directly.
    nuisance = dict(n_estimators=120, min_samples_leaf=12, n_jobs=1, random_state=seed)
    causal = LinearDML(
        model_y=ExtraTreesRegressor(**nuisance),
        model_t=ExtraTreesRegressor(**nuisance),
        treatment_featurizer=PolynomialFeatures(degree=2, include_bias=False),
        discrete_treatment=False,
        cv=GroupKFold(n_splits=3),
        random_state=seed,
    )
    causal.fit(y, t, X=x, W=w, groups=train.patient_id.to_numpy(), inference="statsmodels")
    return ModelBundle(outcome, causal, SupportGuard.fit(train), seed, len(train))


def predictive_metrics(bundle: ModelBundle, test: pd.DataFrame) -> dict[str, float]:
    """Factual outcome accuracy is NOT validation of causal identification."""
    y = numeric(test, [OUTCOME]).ravel()
    pred = bundle.outcome.predict(outcome_features(test, numeric(test, PLAN_COLUMNS)))
    return {
        "factual_mae": float(mean_absolute_error(y, pred)),
        "factual_rmse": float(root_mean_squared_error(y, pred)),
    }
