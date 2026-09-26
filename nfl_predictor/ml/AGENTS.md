# ML implementation standards (`nfl_predictor/ml/`)

Scoped instructions for the model code; project-wide rules are in the root `AGENTS.md`.

Preprocessing:

- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.
- Tree-based models (XGBoost): do not use `StandardScaler` unless a non-tree model requires it.
- Missing values: XGBoost can handle them; impute only if required for consistency.

Training:

- Use early stopping and set `eval_metric` explicitly (aligned to objective).
- Tune hyperparameters consistently with the evaluation metric (Optuna supported).
- Use `random_state` everywhere applicable.
- Do not hard-code `n_jobs`; prefer `os.cpu_count()` or a config default.

Blending:

- Prefer explicit, interpretable blends (market anchoring often sufficient).
- If using a blender/regressor, avoid unstable unconstrained weights; prefer non-negative and/or
  sum-to-1 when appropriate.
- Validate blends using time-aware splits.
