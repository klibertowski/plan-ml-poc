# Modeling notes

## Scope and source context

The implementation draws on these v2 project chats, reviewed as background rather than executable instructions:

- **ML Surgical Planning Challenges**: one-year FJS objective; pretreatment features, executed positions and outcomes; collecting the right data; limits of roughly 1,000 observations.
- **Design Surgical Planning Model**: patient representation, candidate evaluation and constrained optimization as separate components; deep imaging models are future work.
- **Train EconML Model**: X versus W, grouped evaluation, continuous-treatment limitations, conditional effects and confidence intervals.
- **Causal ML Breakdown**: optimize continuous positions rather than feed angles into a model trained only on MA/KA; offline nuisance training; runtime search, support, stopping and alternatives.
- **Explain FJS Calculation**, **Explain KOOS-JR HOOS-JR**, **Explain Math Syntax**: outcome terminology and teaching context.

The user's supplied diagram maps patient features and plan search into outcome prediction plus causal effect simulation, followed by an optimization score and feedback loop. We implement that organization. “Simulation” here means querying contrasts from a fitted statistical model, not a patient digital twin.

The synthetic target is FJS-like on a 0–100 scale. It is generated directly, not scored from questionnaire responses. Baseline function is generic; FJS, KOOS-JR and HOOS-JR are not interchangeable measures. There is one outcome head and no validated risk model.

## The causal estimand

Let X denote pretreatment effect modifiers, W additional pretreatment adjustment variables, A the joint vector of three executed component angles, and Y the synthetic 12-month outcome. We seek a structured conditional average contrast:

`tau(x; a0, a1) = E[Y(a1) - Y(a0) | X=x]`.

The fitted model assumes `Y = baseline(X,W) + theta(X) · phi(A) + noise`, with theta linear in X and phi quadratic in A. W is assumed not to modify the treatment response; it adjusts treatment assignment/outcome nuisance regressions. If care context modifies effects, move the relevant variables into X or change the model specification.

All three angles form a **joint intervention**. The model does not identify the isolated effect of one angle in the presence of unrecorded co-interventions. Candidate planned angles are interpreted as perfectly achieved executed positions. Learning execution uncertainty is future work. A fixed implant family is assumed; this POC does not optimize categorical implant or technique choices.

Causal interpretation additionally requires a well-defined intervention, consistency, no relevant unmeasured confounders, adequate joint overlap, suitable statistical specification and independent patient units. The synthetic clean world satisfies these by construction. No amount of working software guarantees these conditions in clinical data.

## Training mechanics

1. Split by patient ID with `GroupShuffleSplit` before fitting. The POC's default data has one case per patient, but repeated IDs remain grouped.
2. Use deterministic named feature selection and fixed domain scaling for X. W contains two binary indicators. No outcome, oracle mean or patient ID enters the predictors.
3. Fit an XGBoost outcome regressor on X/W/A. It estimates associations for factual prediction.
4. Fit EconML `LinearDML` with `GroupKFold(3)` and independent ExtraTrees nuisance regressors. One nuisance predicts Y from X/W; the other predicts the nine polynomial treatment features from X/W. Multi-output regression is needed for the latter.
5. Cross-fitted residuals feed the final structured coefficient model; statsmodels inference supplies pointwise effect intervals.
6. Fit support normalization and nearest-neighbor reference distances on the training sample only. Persist all fitted components with the schema, seed, training row count and versions.

The treatment featurizer is part of the estimand's response specification. Degree one assumes constant slope at fixed X. Degree two adds curvature and interactions, but higher degree quickly increases dimensionality and identification demands. Our oracle is intentionally quadratic with linear heterogeneity: model recovery is much easier than an unknown clinical response.

The dependency lock uses EconML 0.16.0 and scikit-learn 1.6.1 as a conservative compatible baseline. Consult the [LinearDML API](https://www.pywhy.org/EconML/_autosummary/econml.dml.LinearDML.html) before changing treatment featurization or inference. This model uses orthogonalization and cross-fitting; do not label it automatically doubly robust in the DR-Learner sense.

## Evaluation and utility

Every candidate has separate fields for:

- toy feasibility and empirical proximity support;
- the XGB observational outcome prediction at the candidate;
- EconML's effect relative to a single fixed reference, plus pointwise CI lower/upper;
- the hybrid anchored absolute outcome, movement cost and utility.

`anchored_outcome = reference_XGB_prediction + effect`

`utility = reference_XGB_prediction + effect_CI_lower - movement_cost`

`movement_cost = 0.1 * sum(((candidate-reference)/search_span)^2)` by default.

The penalty and reference `[0,0,3]` are teaching choices, not clinical preferences. Because the reference anchor is constant per patient, rankings use only the causal lower bound minus cost. The candidate XGB prediction is displayed for comparison and not separately rewarded. The absolute anchor is not causally identified by the DML contrast alone. Do not treat its point estimate as a validated potential outcome, and do not treat the effect interval as an absolute prediction interval.

The lower-bound utility is an uncertainty-averse heuristic. It does not establish a clinically meaningful improvement and provides no simultaneous confidence guarantee across a searched action space. Causal uncertainty and search-GP uncertainty are recorded separately.

## Support, feasibility and search

Support uses standardized joint X/W/A distances. The threshold is the training 95th percentile of the eighth-other-neighbor distance, plus marginal training-range checks. At inference, the eighth-neighbor distance must fall below that threshold. Both endpoints must pass. Reference failure raises an abstention error. Unsupported candidates have no outcome/effect/utility values.

This proxy does not estimate a continuous propensity density or prove conditional positivity. It can miss holes and sparsity, and one threshold is a simplification. Duplicate cases and clustered patients also warrant a group-aware support audit. The toy feasibility gate checks declared angle bounds and `abs(femoral_varus + tibial_varus) <= 4`; no meshes, contact mechanics, ligament gaps, implant dimensions or clinical contraindications are modeled.

The optimizer starts at the reference. It samples a proposal pool, filters candidates, and performs random startup evaluations. Thereafter a fixed-kernel Gaussian process models angles to utility; Expected Improvement chooses the next candidate from a fresh supported pool. The search surrogate alone is refitted. Its hyperparameters are fixed for readability; acquisition maximization over a random pool is approximate. See [scikit-learn's GP documentation](https://scikit-learn.org/stable/modules/gaussian_process.html) for the surrogate model.

Stopping conditions are an evaluation budget (reference included), a significant-utility-improvement plateau, or no supported candidate in the finite proposal pool. The latter does not prove the continuous search space is exhausted. The result includes evaluated history, rejected proposal count, stopping reason and top five evaluated alternatives; these alternatives are not required to be geometrically diverse.

## Validation and next steps

Factual prediction error is not a test of causality. Synthetic oracle contrasts permit effect-error and pointwise-coverage diagnostics for one prespecified held-out contrast. These cannot be computed on ordinary observational data. Notebook 04 adds hidden confounding to demonstrate model bias despite successful training.

For a future study, reserve patient-grouped chronological and external-site cohorts before tuning. Address missing follow-up and censoring; assess assignment variation, sensitivity to omitted confounders, treatment measurement error, calibration and effect-model misspecification. Institution/surgeon clustering needs appropriate inference beyond this patient-independent example. An optimized policy requires independent evaluation; these POC metrics do not constitute off-policy or prospective validation.

Extend physical generation through `evaluation.feasible` or a new plan-generator interface. Add CT-derived features through the schema and a dedicated training-only preprocessing pipeline. Introduce new outcomes with separate estimands and uncertainty; use survival methods for time-to-revision. THA requires its own anatomical features, cup/stem actions, physical constraints and target definition. Collect actual outcomes before fitting a clinical causal model; historical approved plans alone support imitation, not learning outcome effects.
