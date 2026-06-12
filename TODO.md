# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo. This file contains **active** work only.
Completed milestones live in `ARCHIVE.md`.

- Completed work should be moved to `ARCHIVE.md` (with dates/notes).
- Agent workflow + guardrails live in `AGENTS.md`.

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
1. **Plan**: outline the smallest set of changes needed.
1. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
1. **Implement**: make changes incrementally (small diffs, one logical change at a time).
1. **Run checks (venv only)**:
   - `.venv/bin/ruff format --check .`
   - `.venv/bin/ruff check .`
   - `.venv/bin/pyright .`
   - `.venv/bin/ty check .`
   - `.venv/bin/python -m pytest`
   - `markdownlint .`

1. **Update docs** where behavior changes (README/AGENTS/CHANGELOG), and update TODO/ARCHIVE.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `uv`, `pytest`, `ruff`, `pyright`, or `ty` without the venv
  prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.
- Prefer `uv ...` for dependency management and environment sync.

### Current validated baseline (2026-06-12)

- `.venv/bin/ruff format --check .` passes.
- `.venv/bin/python -m pytest` passes (`238 passed`).
- `markdownlint .` passes.
- `uv lock --check` passes.
- `uv sync --check --active` passes.
- `.venv/bin/ruff check .` fails with 69 diagnostics.
- `.venv/bin/pyright .` fails broadly in pandas-heavy modules.
- `.venv/bin/ty check .` fails broadly and is currently advisory.
- Coverage is `78%`, below the preseason target of `90%` or higher.

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.
- [ ] Confirm historical data is cached (TeamRankings + nflreadpy), and re-scrapes are minimized.

---

## Milestone 44 - Preseason 2026 repo hardening and tooling alignment

Goal: bring the repository back to a fully coherent, season-ready baseline on Python 3.14 before
resuming larger feature work.

Tasks:

- [ ] Align project instructions and docs with the current toolchain:
  - keep `AGENTS.md` as the single instruction file and remove duplicate agent-instruction files.
  - update `AGENTS.md`, `README.md`, and helper docs to reflect Ruff-only formatting plus the
    current venv entrypoints for agents.
  - remove stale Black references and `.venv/bin/pip` assumptions that do not match the `uv`
    environment layout.
  - document the intended role of `update_requirements.sh`, `uv lock`, and `uv sync`.
  - add and maintain `CHANGELOG.md` using Common Changelog, with `0.1.0` on `main` as the
    historical baseline for future tagged releases.
  - decide whether to add nested README files for `ml` and `reporting`, or add that work to the
    active plan explicitly.
- [ ] Reconcile `pyproject.toml` metadata and tool configuration:
  - align `project.requires-python`, `tool.ruff.target-version`, and `tool.pyright.pythonVersion`.
  - remove stale copied comments that reference other repositories or outdated Python versions.
  - migrate development dependency declarations into `pyproject.toml` dependency groups and make
    `uv.lock` the lockfile source of truth.
  - decide whether the coverage floor remains advisory or becomes enforced again.
- [ ] Decide the `ty` rollout strategy:
  - keep `pyright` as the primary blocking type checker until parity is proven on this repo.
  - evaluate a minimal `[tool.ty]` configuration for environment, include paths, and test overrides.
  - document which checker is blocking and which checker is advisory during the transition.
- [ ] Close the current Ruff debt:
  - fix missing package docstrings and docstring-style violations.
  - address security findings such as `S110` and `S607` with either code changes or documented,
    justified exceptions.
  - resolve long Excel-formula lines plus simplify and NumPy diagnostics, or explicitly scope any
    exceptions that remain.
- [ ] Establish a passing type-check baseline under the refreshed dependency set:
  - fix or intentionally scope the current `pyright` failures in pandas-heavy modules.
  - fix or intentionally scope the current `ty` failures and re-run both checkers.
- [ ] Fix the highest-value operational correctness issues from the current audit:
  - validation script exit codes and logger usage.
  - stale season-specific defaults in checked-in orchestration config and scripts.
  - any command guidance that no longer matches the actual `.venv/bin` contents.
- [ ] Re-establish quality gates and automation:
  - raise the coverage gate toward `90%` or higher, with `100%` as the aspirational ceiling.
  - document the canonical local validation sequence.
  - evaluate a GitHub Actions validation workflow that provisions `.venv` with `uv` and runs Ruff,
    Pyright, Ty, pytest, and markdownlint.
  - decide whether to add CI now or keep the gate local during the preseason hardening pass.
  - decide whether to add a tag-driven GitHub release workflow that publishes `CHANGELOG.md`
    entries once version tags are standardized.

Acceptance:

- [ ] Ruff format, Ruff check, Pyright, Ty, pytest, and markdownlint all pass, or any remaining
      exceptions are documented and intentionally accepted.
- [ ] Editable install and primary CLI help smoke checks pass on Python 3.14.
- [ ] Docs and agent instructions match the actual toolchain and workflow.
- [ ] `CHANGELOG.md` is current and the intended CI/release automation path is documented.

---

## Deferred roadmap (resume after Milestone 44 baseline hardening)

## Milestone 39 - Off-season configuration sweep + lock default settings (GPU-first)

Goal: run an **objective, repeatable sweep** of candidate modeling configurations under the
canonical evaluation protocol, then **write out the selected best configuration** as the default for
weekly runs.

Rationale:

- Weekly results are noisy; the off-season is the right time to decide what’s “best”.
- Walk-forward evaluation already supports the exact loop: “train up through week N, predict week
  N+1” for every week.

Tasks:

- [ ] Define a **sweep config schema** (JSON or YAML) that can express:
  - model kind(s): `margin_total`, `blended_margin_total`, (optional) `score`
  - calibration: `none|elo|platt|isotonic|auto` - market mode: `features|anchor|hybrid` - market
    prob source: `raw|novig` - blending: method `prob|logit`, weight grid, clamp grid
  - uncertainty: `win_prob_uncertainty` on/off
  - tuning: on/off, timeout, objective
  - XGBoost params (including GPU preference)
- [ ] Implement a long-running sweep entrypoint:
  - Option A: new `scripts/config_sweep.py`
  - Option B: add a `--mode sweep` to `scripts/weekly_run.py`
- [ ] The sweep must:
  - run walk-forward evaluation per config
  - output a **single comparable summary table** (CSV + JSON) with:
    - Brier / log loss / reliability
    - pool metrics (tie-breakers)
    - margin/total MAE (tie-breakers)
    - market-relative metrics when market is present
  - support **resume** (skip configs already evaluated for the same dataset hash + config hash)
- [ ] Add “GPU whenever possible” support:
  - allow `xgb_device=auto` (prefer `cuda` if available, else CPU)
  - ensure the same device settings are used in both the walk-forward evaluation and the final
    trained model
- [ ] Clarify the **ScoreModel** decision:
  - audit whether `model-kind score` is used anywhere in scripts/docs
  - if kept: document it explicitly as experimental / likely inferior to margin/total
  - if removed: deprecate cleanly (clear error + doc update)

Acceptance:

- [ ] One command + one config file can run a sweep and produce:
  - `sweep_summary.csv` sorted by the selection hierarchy
  - `best_config.json` (or `.yaml`)
- [ ] `scripts/weekly_run.py` can accept `--defaults-path best_config.json` and run end-to-end
      without manual CLI overrides.
- [ ] The sweep report includes deltas vs a baseline (no market, no blending), so improvements are
      obvious.

---

## Milestone 40 - Market integration monitoring + ablations

Goal: keep market integration honest by measuring its value and failure modes over time.

Tasks:

- [ ] Add **market impact reporting** to comparison outputs:
  - Always include a baseline row with market blending/clamping OFF.
  - Report deltas vs baseline for primary metrics (Brier/log loss) and pool metrics.
- [ ] Add a stability view:
  - performance by season and by week bucket (early-season, mid, late, postseason)
  - highlight weeks where market blending hurts calibration or increases MAE
- [ ] Add a single “recommended defaults” section to the report:
  - mode (features vs anchor vs hybrid)
  - prob source (raw vs novig)
  - blend method (prob vs logit)
  - selected weight + clamp settings

Acceptance:

- [ ] Market integration can be defended (or disabled) based on walk-forward evidence.

---

## Milestone 41 - One-command weekly orchestration (regular season + postseason)

Goal: a single command to run 1–2x per week that:

1. refreshes data
2. selects defaults (from the sweep, when available)
3. trains the chosen model configuration
4. emits all weekly outputs in a consistent place

Outputs to include (as available):

- weekly predictions (`*_predictions.csv`)
- confidence pool picks (unique 1..N ranks)
- power rankings for the week
- betting report + optional Excel template

Tasks:

- [ ] Data refresh control:
  - Add a `--data-min-season` / `--data-max-season` pass-through (or a generic
    `--data-collection-args`) so orchestration can rebuild from **1999+** when desired.
- [ ] Postseason considerations:
  - Decide and document how postseason games enter evaluation and training (include flag +
    weighting).
  - If the prediction input week is postseason, default power rankings “through week” to the last
    regular-season week (`constants.get_regular_season_weeks(season)`) unless explicitly overridden.
- [ ] Wire sweep-selected defaults into `scripts/weekly_run.py` once Milestone 39 lands.
- [ ] Confirm the weekly output package and resume behavior still work with the selected defaults.

Acceptance:

- [ ] One command produces a complete weekly output package from scratch.
- [ ] Re-running does not redo expensive work unless inputs or config changed.

---

## Milestone 42 - Ensembles and alternative models (accuracy experiments)

Goal: test whether alternative model families or simple ensembles produce **measurable** and
**repeatable** gains under the canonical evaluation protocol.

Notes:

- Start with **no new heavy dependencies** (e.g., XGBoost-based classifier) so the experiment is
  cheap.
- Only keep what wins under walk-forward (and doesn’t add fragile complexity).

Tasks:

- [ ] Direct win-prob classifier (no new deps):
  - Train an `XGBClassifier` (or logistic regression baseline) to predict win/loss directly.
  - Calibrate its probabilities using the same calibration scheme (Platt/isotonic/auto).
  - Compare against the margin->prob approach under walk-forward.
- [ ] Simple probability ensemble:
  - Blend the calibrated margin-based probability with the calibrated classifier probability.
  - Prefer blending in **log-odds (logit) space**.
  - Evaluate blend weights under walk-forward.
- [ ] Optional: alternative GBMs behind extras (only if needed):
  - LightGBM and/or CatBoost as optional dependencies (extras) with matching feature prep.
  - Evaluate them under the same protocol and (optionally) ensemble.
- [ ] Optional: season-phase specialization:
  - Train separate early-season vs late-season models (or learn a gating function) and ensemble
    them.
  - Validate that this beats a single model and doesn’t overfit.

Acceptance:

- [ ] At least one alternative/ensemble approach is evaluated and reported under walk-forward.
- [ ] If any approach wins, it can be enabled via config and runs end-to-end in weekly
      orchestration.

---

## Milestone 43 - Power rankings realism + recency weighting

Goal: make weekly power rankings reflect **current-season strength** (and recent performance),
instead of long-run franchise averages.

Rationale:

- Current rankings mix all historical seasons equally, which can swamp a breakout season.
- Users expect “power rankings” to track the current season and recent form.

### Tasks

#### 43.1 - Add explicit recency/season window controls

- [ ] Add config options to `scripts/power_rankings.py` (and `scripts/weekly_run.py`) such as:
  - `--ratings-window-seasons N` (use last N seasons including current)
  - `--ratings-half-life-seasons` or `--ratings-half-life-weeks` (optional exponential weighting)
  - keep `--ratings-min-season` for explicit overrides
- [ ] Define a **sane default** (e.g., last 2–3 seasons) when no overrides are provided.

Acceptance:

- [ ] Defaults use a recent-season window without requiring extra flags.

---

#### 43.2 - Weighted Bradley–Terry fit

- [ ] Extend `fit_bradley_terry_ratings` to accept sample weights.
- [ ] Compute weights from season/week age (heavier weight for recent games).
- [ ] Ensure weighting does **not** change fold semantics or introduce leakage.

Acceptance:

- [ ] Weights change ratings in the expected direction on synthetic tests.

---

#### 43.3 - Outcome probability mapping improvements

- [ ] Add an option to map completed-game outcomes to probabilities using margin-based logic (e.g.,
      `margin_to_home_win_prob`) instead of fixed 0.97/0.03.
- [ ] Keep current binary mapping as an option for comparability.

Acceptance:

- [ ] A configurable mapping exists and is documented.

---

#### 43.4 - Tests

- [ ] Add unit tests for:
  - recency weighting shifts ratings toward recent performance
  - season-window logic excludes older seasons
  - new probability mapping behaves as expected

Acceptance:

- [ ] `pytest` passes with coverage for the new options.

---

#### 43.5 - Docs + usage guidance

- [ ] Update README power rankings notes to describe the new defaults and flags.
- [ ] Clarify in `scripts/power_rankings.py --help` how to reproduce “franchise” vs “current-season”
      rankings.

Acceptance:

- [ ] Docs explain how to get realistic weekly rankings.

---

Acceptance (Milestone 43 complete)

- [ ] Power rankings for a late-season week align with current-season results and recent form.
- [ ] Users can still opt into long-run franchise ratings via explicit flags.
