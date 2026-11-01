"""Run the complete synthetic example, or reuse its frozen models for inference."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .diagnostics import oracle_diagnostics
from .evaluation import PlanEvaluator
from .persistence import load_bundle, save_bundle
from .schema import W_COLUMNS, X_COLUMNS
from .search import search_plans
from .synthetic import make_cases
from .training import fit_models, predictive_metrics, split_cases


def example_patient() -> pd.DataFrame:
    """A central synthetic patient and care context, independent of holdout data."""
    return pd.DataFrame([dict(zip(X_COLUMNS + W_COLUMNS, [65, 29, 50, 0, 88, 88, 0, 0]))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--budget", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    parser.add_argument(
        "--load-models",
        type=Path,
        help="Run inference with a previously saved trusted bundle; skip training.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if args.load_models:
        bundle = load_bundle(args.load_models)
    else:
        train, test = split_cases(make_cases(args.rows, args.seed), args.seed)
        bundle = fit_models(train, args.seed)
        save_bundle(bundle, args.output)
        metrics = {**predictive_metrics(bundle, test), **oracle_diagnostics(bundle, test)}
    patient = example_patient()
    result = search_plans(PlanEvaluator(bundle, patient), budget=args.budget, seed=args.seed)
    result.history.to_csv(args.output / "search_history.csv", index=False)
    result.alternatives.to_csv(args.output / "ranked_alternatives.csv", index=False)
    patient.to_json(args.output / "patient.json", orient="records", indent=2)
    summary = {
        "synthetic_only": True,
        "metrics": metrics,
        "stop_reason": result.stop_reason,
        "evaluations": len(result.history),
        "rejected_proposals": result.rejected_proposals,
        "best_evaluated_plan": json.loads(result.alternatives.iloc[0].to_json()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("SYNTHETIC DEMO — mechanics only; no clinical conclusions.")
    print(json.dumps({k: v for k, v in summary.items() if k != "best_evaluated_plan"}, indent=2))
    print(result.alternatives.to_string(index=False))


if __name__ == "__main__":
    main()
