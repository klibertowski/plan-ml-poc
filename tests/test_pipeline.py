"""Behavioral checks for the mechanics, not evidence of clinical validity."""

import numpy as np
import pandas as pd
import pytest

from plan_ml_poc.demo import example_patient
from plan_ml_poc.diagnostics import oracle_diagnostics
from plan_ml_poc.evaluation import PlanEvaluator, feasible
from plan_ml_poc.persistence import load_bundle, save_bundle
from plan_ml_poc.schema import PLAN_COLUMNS, REFERENCE, patient_features
from plan_ml_poc.search import expected_improvement, search_plans
from plan_ml_poc.synthetic import make_cases
from plan_ml_poc.training import fit_models, split_cases


@pytest.fixture(scope="module")
def fitted():
    train, test = split_cases(make_cases(1600))
    return fit_models(train), train, test


def test_generator_repeatable_and_group_split():
    data = make_cases(100)
    pd.testing.assert_frame_equal(data, make_cases(100))
    repeated = pd.concat([data, data.iloc[:20]], ignore_index=True)
    train, test = split_cases(repeated)
    assert set(train.patient_id).isdisjoint(test.patient_id)
    assert len(train) + len(test) == len(repeated)


def test_missing_data_rejected():
    data = make_cases(100)
    data.loc[0, "bmi"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        split_cases(data)


def test_effect_identity_and_direction(fitted):
    bundle, _, test = fitted
    x = patient_features(test.iloc[:5])
    t0 = np.tile(REFERENCE, (len(x), 1))
    t1 = t0 + [0.5, 0.5, 0.5]
    zero = bundle.causal.effect(x, T0=t0, T1=t0)
    forward = bundle.causal.effect(x, T0=t0, T1=t1)
    reverse = bundle.causal.effect(x, T0=t1, T1=t0)
    np.testing.assert_allclose(zero, 0, atol=1e-10)
    np.testing.assert_allclose(forward, -reverse, atol=1e-10)
    assert oracle_diagnostics(bundle, test)["oracle_effect_mae"] < 2.5


def test_constraints_support_and_objective(fitted):
    bundle, _, _ = fitted
    evaluator = PlanEvaluator(bundle, example_patient())
    rows = evaluator.evaluate(np.array([REFERENCE, [3, 3, 3], [np.nan, 0, 3]]))
    assert rows.status.tolist() == ["accepted", "rejected_constraints", "rejected_constraints"]
    assert rows.loc[1:, "utility"].isna().all()
    assert rows.loc[0, "effect"] == pytest.approx(0)
    assert rows.loc[0, "utility"] == pytest.approx(evaluator.reference_prediction)
    extreme = example_patient()
    extreme["age"] = 150
    with pytest.raises(ValueError, match="Unsupported"):
        PlanEvaluator(bundle, extreme)
    assert not feasible(np.array([[3, 3, 3]]))[0]


def test_frozen_inference_and_search(fitted, tmp_path):
    bundle, _, _ = fitted
    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)
    patient = example_patient()
    evaluator = PlanEvaluator(loaded, patient)
    before = loaded.causal.coef_.copy()
    first = search_plans(evaluator, budget=12, seed=7, startup=4)
    second = search_plans(evaluator, budget=12, seed=7, startup=4)
    pd.testing.assert_frame_equal(first.history, second.history)
    np.testing.assert_array_equal(before, loaded.causal.coef_)
    assert 1 <= len(first.history) <= 12
    assert first.history.supported.all() and first.history.feasible.all()
    assert first.history.best_so_far.is_monotonic_increasing
    assert (first.history.phase == "expected_improvement").any()
    assert first.alternatives.utility.iloc[0] >= first.history.utility.iloc[0]
    assert list(first.alternatives[PLAN_COLUMNS].columns) == PLAN_COLUMNS
    original = PlanEvaluator(bundle, patient).evaluate(np.array([REFERENCE, [1, 0, 3]]))
    reloaded = evaluator.evaluate(np.array([REFERENCE, [1, 0, 3]]))
    pd.testing.assert_frame_equal(original, reloaded)


def test_budget_and_acquisition(fitted):
    bundle, _, _ = fitted
    result = search_plans(PlanEvaluator(bundle, example_patient()), budget=1)
    assert len(result.history) == 1 and result.stop_reason == "evaluation_budget"
    np.testing.assert_array_equal(expected_improvement(np.array([1.0]), np.array([0.0]), 1), 0)
    with pytest.raises(ValueError):
        search_plans(PlanEvaluator(bundle, example_patient()), budget=0)


def test_stopping_and_unsupported_candidates(fitted, monkeypatch):
    bundle, _, _ = fitted
    evaluator = PlanEvaluator(bundle, example_patient())
    plateau = search_plans(evaluator, budget=10, startup=2, patience=3, tolerance=100)
    assert plateau.stop_reason == "utility_plateau" and len(plateau.history) < 10

    # Restrict support to the reference to exercise honest abstention in the loop.
    def reference_only(patient, plans):
        supported = np.isclose(plans, REFERENCE).all(axis=1)
        return supported, np.where(supported, 0.0, 100.0)

    monkeypatch.setattr(bundle.support, "check", reference_only)
    rejected = evaluator.evaluate(np.array([[1, 0, 3]]))
    assert rejected.status.iloc[0] == "rejected_support"
    assert rejected[["effect", "utility", "outcome_prediction"]].isna().all().all()
    exhausted = search_plans(evaluator, budget=10)
    assert exhausted.stop_reason == "no_supported_candidate_in_pool"
    assert len(exhausted.history) == 1
