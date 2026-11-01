"""Small, inspectable Gaussian-process Bayesian optimizer.

Fit a GP to observed *utility*, calculate Expected Improvement on a fresh
candidate pool, evaluate the most promising supported point, and repeat.
The pool approximates acquisition maximization; this is not a global guarantee.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .evaluation import PlanEvaluator, feasible
from .schema import BOUNDS, PLAN_COLUMNS


@dataclass
class SearchResult:
    history: pd.DataFrame
    alternatives: pd.DataFrame
    stop_reason: str
    rejected_proposals: int


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float, exploration: float = 0.01
) -> np.ndarray:
    improvement = mean - best - exploration
    safe_std = np.maximum(std, 1e-12)
    z = improvement / safe_std
    ei = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
    return np.where(std > 1e-12, ei, 0.0)


def search_plans(
    evaluator: PlanEvaluator,
    budget: int = 35,
    startup: int = 7,
    pool_size: int = 500,
    patience: int = 12,
    tolerance: float = 0.02,
    seed: int = 42,
    method: str = "bayesian",
) -> SearchResult:
    """Return best evaluated supported plans, including the reference candidate.

    `budget` counts actual utility evaluations, including the reference. Cheap
    feasibility/support checks do not consume it, but rejected proposals are
    counted. A finite pool can be exhausted without proving global exhaustion.
    The causal and XGB models are never retrained inside this loop.
    """
    if budget < 1 or startup < 1 or pool_size < 1 or patience < 1 or tolerance < 0:
        raise ValueError("Use positive budgets/counts and nonnegative tolerance.")
    if method not in {"bayesian", "random"}:
        raise ValueError("method must be 'bayesian' or 'random'")
    rng = np.random.default_rng(seed)
    low, span = BOUNDS[:, 0], BOUNDS[:, 1] - BOUNDS[:, 0]
    initial = evaluator.evaluate(evaluator.reference).iloc[0].to_dict()
    initial.update(iteration=0, acquisition_ei=np.nan, search_std=np.nan, phase="reference")
    records = [initial]
    significant_best = float(initial["utility"])
    stale = 0
    rejected = 0
    stop_reason = "evaluation_budget"
    for iteration in range(1, budget):
        pool = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(pool_size, 3))
        allowed = feasible(pool)
        support = np.zeros(pool_size, dtype=bool)
        if allowed.any():
            support[allowed] = evaluator.bundle.support.check(evaluator.patient, pool[allowed])[0]
        keep = allowed & support
        rejected += int((~keep).sum())
        pool = pool[keep]
        observed = np.array([[row[c] for c in PLAN_COLUMNS] for row in records])
        # Avoid exact/nearly repeated evaluations, including the reference.
        if len(pool):
            separation = np.linalg.norm((pool[:, None] - observed[None, :]) / span, axis=2)
            pool = pool[separation.min(axis=1) > 1e-6]
        if not len(pool):
            stop_reason = "no_supported_candidate_in_pool"
            break
        acquisition, search_std = np.nan, np.nan
        phase = "random_startup"
        chosen = 0
        if method == "bayesian" and len(records) >= startup:
            # Fixed GP hyperparameters keep the example deterministic and concise.
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, constant_value_bounds="fixed")
                * Matern(length_scale=np.full(3, 0.35), length_scale_bounds="fixed", nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                optimizer=None,
            )
            utilities = np.array([row["utility"] for row in records])
            gp.fit((observed - low) / span, utilities)
            mean, std = gp.predict((pool - low) / span, return_std=True)
            ei = expected_improvement(mean, std, float(utilities.max()))
            chosen = int(np.argmax(ei))
            acquisition, search_std = float(ei[chosen]), float(std[chosen])
            phase = "expected_improvement"
        row = evaluator.evaluate(pool[chosen : chosen + 1]).iloc[0].to_dict()
        row.update(
            iteration=iteration, acquisition_ei=acquisition, search_std=search_std, phase=phase
        )
        records.append(row)
        if row["utility"] > significant_best + tolerance:
            significant_best = row["utility"]
            stale = 0
        else:
            stale += 1
        if len(records) >= startup and stale >= patience:
            stop_reason = "utility_plateau"
            break
    history = pd.DataFrame(records)
    history["best_so_far"] = history.utility.cummax()
    alternatives = history.sort_values("utility", ascending=False).head(5).copy()
    return SearchResult(history, alternatives, stop_reason, rejected)
