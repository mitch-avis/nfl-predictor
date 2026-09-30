# ML implementation standards (`src/nfl_predictor/ml/`)

Scoped instructions for the model code; project-wide rules are in the root `AGENTS.md`.

Preprocessing:

- Use `ColumnTransformer` for categorical one-hot + numeric passthrough/impute.
- Avoid densifying large sparse matrices unintentionally.
- Tree-based models (XGBoost): do not use `StandardScaler` unless a non-tree model requires it.
- Missing values: XGBoost can handle them; impute only if required for consistency.

Training:

- Optuna tuning trials early-stop on their validation fold with `eval_metric` set explicitly
  (aligned to the objective); the final fit and walk-forward folds run the full tree budget
  with no eval set.
- Tune hyperparameters consistently with the evaluation metric (Optuna supported).
- Use `random_state` everywhere applicable.
- Do not hard-code `n_jobs`; prefer `os.cpu_count()` or a config default.

Market and probability path:

- The market enters the model only as features and through market anchoring (residual targets
  against the spread and total); there is no blend layer and no market blend or clamp of the
  win probability.
- The win probability is the deterministic floor of the predicted margin
  (`Phi(margin / sigma)`, sigma from `floor_sigma.estimate`), the same in every run type. A
  different probability path is a default change for the user to decide, measured first against
  the floor with time-aware splits.
