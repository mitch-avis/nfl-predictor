# TODO - Active Work for nfl-predictor

This file is the **authoritative worklist** for the repo.
This file contains **active** work only. Completed milestones live in `ARCHIVE.md`.

- Completed work should be moved to `ARCHIVE.md` (with dates/notes).
- Agent workflow + guardrails live in `AGENTS.md`.

---

## Execution loop (required)

For each task:

1. **Understand scope**: read the relevant modules/tests/docs.
2. **Plan**: outline the smallest set of changes needed.
3. **Test-Driven Development**: add/adjust tests for all planned changes. Aim to increase coverage.
4. **Implement**: make changes incrementally (small diffs, one logical change at a time).
5. **Run checks (venv only)**:
   - `.venv/bin/ruff check .`
   - `.venv/bin/black .`
   - `.venv/bin/python -m pytest`
6. **Update docs** where behavior changes (README/AGENTS), and update TODO/ARCHIVE.

### Always use the repo venv

Agents and humans should not rely on the shell activation state.

- Do **not** run `python`, `pip`, `uv`, `pytest`, `black`, or `ruff` without the venv prefix.
- Use `.venv/bin/python ...` or the tool-specific binary under `.venv/bin/`.

---

## Guardrails checklist (must stay true)

- [ ] Confirm all new features apply to all matchups (no end-of-season-only logic).
- [ ] Confirm no leakage: all splits, features, and labels are time-aware.
- [ ] Confirm ETL is Polars-first and pulls NFLverse via `nflreadpy`.
- [ ] Confirm artifacts remain reproducible (run folder, metadata, metrics).
- [ ] Confirm new code includes unit tests and does not reduce coverage.
- [ ] Confirm historical data is cached (TeamRankings + nflreadpy), and re-scrapes are minimized.

---

## Milestone 38 - Off-season configuration sweep + lock default settings (GPU-first)

Goal: run an **objective, repeatable sweep** of candidate modeling configurations under the
canonical evaluation protocol, then **write out the selected best configuration** as the default
for weekly runs.

Rationale:

- Weekly results are noisy; the off-season is the right time to decide what’s “best”.
- Walk-forward evaluation already supports the exact loop:
  “train up through week N, predict week N+1” for every week.

Tasks:

- [ ] Define a **sweep config schema** (JSON or YAML) that can express:
  - model kind(s): `margin_total`, `blended_margin_total`, (optional) `score`
  - calibration: `none|elo|platt|isotonic|auto`
  - market mode: `features|anchor|hybrid`
  - market prob source: `raw|novig`
  - blending: method `prob|logit`, weight grid, clamp grid
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
  - ensure the same device settings are used in both the walk-forward evaluation and the final trained
    model
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
- [ ] The sweep report includes deltas vs a baseline (no market, no blending), so improvements are obvious.

---

## Milestone 39 - Market integration monitoring + ablations

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

## Milestone 40 - One-command weekly orchestration (regular season + postseason)

Goal: a single command to run 1–2x per week that:

1) refreshes data
2) selects defaults (from the sweep, when available)
3) trains the chosen model configuration
4) emits all weekly outputs in a consistent place

Outputs to include (as available):

- weekly predictions (`*_predictions.csv`)
- confidence pool picks (unique 1..N ranks)
- power rankings for the week
- betting report + optional Excel template

Tasks:

- [ ] Ensure the orchestration script is the canonical interface (`scripts/weekly_run.py`).
- [ ] Data refresh control:
  - Add a `--data-min-season` / `--data-max-season` pass-through (or a generic `--data-collection-args`)
    so orchestration can rebuild from **1999+** when desired.
- [ ] Postseason considerations:
  - Decide and document how postseason games enter evaluation and training (include flag + weighting).
  - If the prediction input week is postseason, default power rankings “through week” to the last
    regular-season week (`constants.get_regular_season_weeks(season)`) unless explicitly overridden.
- [ ] Fix `scripts/power_rankings.py` correctness + ergonomics:
  - Filter current records to `game_type == "REG"` in `_load_current_records()` so postseason games
    cannot pollute regular-season standings.
  - Fail fast if required model feature columns are missing (do not silently drop required features).
  - Ensure the ScoreModel win-prob path either applies a calibrator (if present) or documents that it
    is intentionally uncalibrated.
  - Add/extend unit tests that cover these edge cases.
- [ ] README “one authoritative weekly pipeline” paragraph:
  - Document the intended weekly command(s) to:
    - refresh data
    - select defaults (from the sweep, when available)
    - train + predict + generate betting outputs
  - Document output locations and naming conventions.

Acceptance:

- [ ] One command produces a complete weekly output package from scratch.
- [ ] Re-running does not redo expensive work unless inputs or config changed.

---

## Milestone 41 - Ensembles and alternative models (accuracy experiments)

Goal: test whether alternative model families or simple ensembles produce **measurable** and
**repeatable** gains under the canonical evaluation protocol.

Notes:

- Start with **no new heavy dependencies** (e.g., XGBoost-based classifier) so the experiment is cheap.
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
  - Train separate early-season vs late-season models (or learn a gating function) and ensemble them.
  - Validate that this beats a single model and doesn’t overfit.

Acceptance:

- [ ] At least one alternative/ensemble approach is evaluated and reported under walk-forward.
- [ ] If any approach wins, it can be enabled via config and runs end-to-end in weekly orchestration.
