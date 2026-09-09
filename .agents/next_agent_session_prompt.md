# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is Milestone 46 from
`.agents/TODO.md`: publish a leakage-safe, pre-week schedule-adjusted offense/defense/special-teams
strength per team as ETL features, plus EPA-based schedule strength for games played and remaining.

Milestone 45 landed the play-by-play foundation this milestone consumes. Read its archive entry
first: the per-snap EPA family is built, ablatable and leakage-safe, but **it did not improve the
primary metric**. Brier and log loss were marginally worse with it on; margin MAE improved in 3 of 3
seasons. The working hypothesis for why is that raw per-snap EPA is unadjusted for opponent, so it
adds little over the existing Elo and TeamRankings predictive ratings. **This milestone is the direct
test of that hypothesis: the same inputs, opponent-adjusted.** If the ridge snapshot also fails to
beat the group-off baseline, say so plainly and recommend reordering the roadmap rather than
continuing to add unadjusted families.

You may spawn subagents, but only in the phases below, with the stated file ownership, and you
remain responsible for integration, the full validation gate, and the honesty of every number you
report. Subagent claims are not results until you have re-run their tests yourself.

## 0. Read first, in this order

1. `AGENTS.md` (non-negotiables, command forms, readiness behaviors that must not regress)
2. `.agents/TODO.md` (Milestone 46 tasks 46.1-46.7, plus the open follow-ups inherited from 45)
3. `.agents/ARCHIVE.md`, the Milestone 45 entry (what exists, what it measured, what broke)
4. `.agents/feature_crosswalk.md` sections 4.4, 4.5, and 6
5. `README.md`, `CHANGELOG.md`

Reference implementations (read-only; never modify or import; say which repo you are reading):

- `../nfl-sos-ratings/nfl_sos_ratings/simultaneous_adjustment.py` (`solve_team_stat_ridge`, `solve_srs`)
- `../nfl-sos-ratings/nfl_sos_ratings/validation/snapshots.py` (weekly snapshot construction)

Owning modules in this repo:

- `nfl_predictor/utils/polars/pbp.py`: `aggregate_pbp_team_game_stats` produces the team-game rows
  (counts and sums only) that the ridge solve consumes. `PBP_TEAM_GAME_COLUMNS` is its contract.
- `nfl_predictor/utils/polars/teamrankings.py`: `aggregate_team_stats_to_week` (the `week < target`
  filter), `_compute_pbp_derived_metrics` (rate derivation), `recompute_derived_metrics`.
- `nfl_predictor/data_collection.py`: `collect_all_data` loads and joins play-by-play via
  `_join_pbp_team_game_stats`; `process_week` builds the per-week feature rows.
- `nfl_predictor/constants.py`: `PBP_COUNT_COLUMNS`, `PBP_STATS`, `FEATURE_GROUP_COLUMN_MARKERS`,
  `EXCLUDE_FROM_OPPONENT_STATS`.
- `nfl_predictor/ml/walk_forward.py`: `resolve_feature_group_columns` and
  `WalkForwardConfig.disabled_feature_groups` (the ablation switch, already authoritative inside
  `run_walk_forward_backtest`).

## 1. Facts to trust unless your verification disproves them

- Baseline 2026-09-09, version `0.3.0`: all gates green, `471 passed`, coverage `90.51%`.
- `data/completed_games_ml.csv` covers `1999-2025` (`7260` rows, `465` columns);
  `data/predict/week_01_games_to_predict.csv` has `16` rows for 2026 Week 1.
- Walk-forward (`2023-2025`, `--eval-last-n-seasons 3`, `720` games), play-by-play group **off**:
  Brier `0.2314`, log loss `0.7440`, pick accuracy `0.6708`, margin MAE `9.9977`, total MAE
  `10.1164`, ECE `0.1269`. Group **on**: `0.2317`, `0.7455`, `0.6778`, `9.9178`, `10.1295`, `0.1244`.
  **Compare against these, not against the older `0.2312`/`0.7352` reference, which is not
  reproducible** (the untouched pre-change dataset yields `0.2300`/`0.7501` under the same config).
- A full walk-forward run at `--eval-last-n-seasons 3` takes roughly 25-30 minutes. Budget for it and
  run arms in the background.
- A full ETL rebuild takes about 320s warm (play-by-play cache is populated for `1999-2025`).
- nflreadpy has no play-by-play for the current season before kickoff and signals that with
  `ValueError`, not `ConnectionError`. Both are handled; do not regress it.
- `markdownlint` on this machine is `markdownlint-cli2` (`/usr/local/bin/markdownlint-cli2`).

## 2. Non-negotiables

- TDD: characterization or failing tests first, then production code, small diffs.
- No leakage: any schedule-adjusted value for week `N` must be solved from games strictly before
  week `N` of that season, with a documented prior-season fallback for early weeks; playoff rows use
  the full regular season. A test must prove perturbing week `N+1` leaves week `N` unchanged.
- All matchups: every new column exists on every row; nulls when a source is missing.
- Polars-first ETL; NumPy inside solvers; pandas only in ML modules.
- Rates are ratios of sums computed after season-to-date aggregation, never mean-of-game-rates.
- Name allowed/defensive metrics explicitly and add them to `EXCLUDE_FROM_OPPONENT_STATS`.
- Cite the formula in each docstring; test each metric against a hand-built fixture; docstrings and
  type hints on everything; no new `noqa`/`type: ignore`/`pragma: no cover` without a real reason;
  no milestone numbers in code or docstrings.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python`/`pytest`/`ruff`.
- XGBoost margin/total is the only model family; no tuning campaigns; no Week 1 production pass.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Back up `data/*.csv` before any ETL run. `python -m nfl_predictor.data_collection` overwrites the
  full dataset unconditionally, so a partial-window run is destructive.
- Commit only if the user asks. If asked: one file per commit, imperative message, and end the
  message with the attribution line the harness provides.

## 3. Subagent rules (apply to every spawn)

- Every subagent receives an explicit file-ownership list and may not edit any other file. If it
  believes another file must change, it reports that instead of editing.
- All agents work in the main working tree and share the repo `.venv`. Do not use git worktrees.
- Subagents run `ruff format` and `ruff check` only on the files they own, and run only their own
  test files. You run the repo-wide gate at each checkpoint.
- Each subagent's final report must contain: files changed, tests added (names), the last 15 lines
  of its own pytest output, formulas or thresholds it chose, and open questions.
- Use `model: "sonnet"` for mechanical work; the default Opus for design-bearing work.
- Run subagents in the background and wait for completion notifications. Never predict a pending
  agent's result. Use `SendMessage` to continue an agent that needs a follow-up.

## Phase 0 - Baseline (you)

1. Run `.venv/bin/python -m pytest -q` and record the actual pass count and coverage.
2. Confirm the data files and the play-by-play cache under `data/cache/nflreadpy/` exist.
3. Back up `data/*.csv` to your scratchpad before anything touches the ETL.

## Phase 1 - Parallel implementation (two background subagents, strict ownership)

Agent A, "ridge-solver" (Opus). Owns new `nfl_predictor/utils/polars/adjusted_strength.py` and new
`tests/test_adjusted_strength.py`. Task: `solve_team_ridge(team_games, response_col, *,
ridge_lambda)` returning centered offense and defense coefficients per team plus a home-field term,
via a NumPy normal-equation solve (port the design of
`../nfl-sos-ratings/simultaneous_adjustment.py::solve_team_stat_ridge`). Add `solve_srs` for the
point-margin companion. Tests: a synthetic round-robin recovers known strengths within tolerance;
home-field term has the right sign and magnitude; coefficients are centered per side; empty and
single-game inputs return typed empties rather than raising; the solve is deterministic.

Agent B, "schedule-strength" (Opus). Owns new `nfl_predictor/utils/polars/schedule_strength.py` and
new `tests/test_schedule_strength.py`. Task: pure helpers that, given a pre-week rating per team and
a schedule, compute `sos_played_adj` (mean opponent rating over games already played) and
`sos_remaining_adj` (mean opponent rating over games not yet played), for every `(season, week,
team)`. These must take ratings as an argument rather than computing them, so they stay independent
of Agent A. Tests against a hand-built four-team schedule with known means; bye weeks; week 1 (no
games played) yields null for played and a full-schedule mean for remaining.

Integration checkpoint 1 (you): run the full gate. Do not start Phase 2 until green.

## Phase 2 - Sequential wiring (you, or one Opus subagent; no parallelism)

2.1 Weekly snapshot builder: for each `(season, week)`, solve on prior-week regular-season rows
using pass and rush EPA per snap as responses. Emit `adj_off_pass_epa_snap`, `adj_off_rush_epa_snap`,
`adj_def_pass_epa_snap`, `adj_def_rush_epa_snap`, `adj_hfa`, an SRS companion, and `st_rating` from
the special-teams EPA margin. Fixed ridge lambda for v1; make it a named constant.
2.2 Early-week prior: previous-season final snapshot regressed by `WEEK1_REGRESSION_FACTOR`, blended
with the in-season solve by `games / (games + K)` with a documented `K`. Playoff weeks use the full
regular season.
2.3 Composite `adj_strength_composite` with documented default weights over within-season
standardized components. Model features stay in raw adjusted units.
2.4 Schedule strength from 2.3 via Agent B's helpers.
2.5 Merge in `process_week` as `away_`/`home_`/`_diff`; add `constants.ADJUSTED_STRENGTH_STATS`,
exclusion entries, and a `"strength"` entry in `FEATURE_GROUP_COLUMN_MARKERS`; extend
`build_final_column_order` and `get_stats_for_diff`.
2.6 Leakage test: perturb week `N+1` play-by-play and assert week `N` snapshot columns are unchanged.
Add the playoff-branch and Week-1-fallback perturbation cases that Milestone 45 left uncovered.

Integration checkpoint 2: full gate green, coverage at or above `90%`.

## Phase 3 - Independent review (two background subagents, read-only)

Reviewer 1 (Opus): leakage, solver correctness against the sos reference, numerical conditioning
(singular matrices, teams with no games), schema invariance, readiness behaviors.
Reviewer 2 (Sonnet): docstrings, type hints, test quality, dead code, lint/type cleanliness.
Address findings yourself, then re-run the gate.

## Phase 4 - Expensive validation (you, background, monitored)

4.1 Back up `data/*.csv`, then `.venv/bin/python -m nfl_predictor.data_collection --timing`.
4.2 `.venv/bin/python scripts/leakage_audit.py --data-path data/completed_games_ml.csv --out-json
    <scratchpad>/leakage.json` must pass.
4.3 Three walk-forward arms, in the background, each about 25-30 minutes:
    - both groups on: `--eval-last-n-seasons 3`
    - strength off: `--eval-last-n-seasons 3 --disable-feature-groups strength`
    - both off: `--eval-last-n-seasons 3 --disable-feature-groups pbp,strength`
4.4 Report all three verbatim from `metrics_report.json`, next to the Milestone 45 numbers in
    section 1, with per-season breakdowns so a one-season fluke is visible.
4.5 Sanity checks: a late-season snapshot must rank teams consistently with current-season point
    differential and adjusted EPA, not prior seasons. Print the top and bottom five for a known
    season and check them against reality.

## Phase 5 - Docs and handoff (you)

- `README.md`, `CHANGELOG.md` (`0.4.0`), `.agents/ARCHIVE.md`, `.agents/TODO.md`, and a regenerated
  `.agents/next_agent_session_prompt.md` for the following session (Milestone 43, the power-rankings
  redesign, which consumes the snapshot this milestone builds).
- `markdownlint-cli2` on every touched Markdown file; then the full gate one last time.

## Final report to the user (structure)

1. Outcome first: what landed, gate status, and whether Phase 4 ran to completion.
2. Walk-forward table: all three arms plus the Milestone 45 numbers, with per-season detail.
3. Whether the opponent-adjustment hypothesis held. If the ridge snapshot also fails to beat the
   group-off baseline, say so plainly and recommend reordering the roadmap.
4. Feature summary: columns added, null rates, sanity-check results.
5. What was left out or deferred, and why.
6. The recommended first step for the next session, with the exact command to resume anything
   unfinished.
