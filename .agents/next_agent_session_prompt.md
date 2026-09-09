# Next Agent Session Prompt

Use the following prompt to start the next agent session on this repository. It is written for an
orchestrating Claude Opus 5 session that may spawn subagents.

```text
You are the orchestrating agent (Claude Opus 5) for an implementation session in the
`nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is Milestone 45
from `.agents/TODO.md`: bring play-by-play (PBP) data into the Polars ETL with per-season caching,
publish per-snap team EPA, success, explosive, and special-teams feature families for every
matchup, and measure them under walk-forward. If the full gate is green with time remaining, start
Milestone 46.1 (the ridge solver module only).

You may spawn subagents, but only in the phases below, with the stated file ownership, and you
remain responsible for integration, the full validation gate, and the honesty of every number you
report. Subagent claims are not results until you have re-run their tests yourself.

## 0. Read first, in this order

1. `AGENTS.md` (non-negotiables, command forms, readiness behaviors that must not regress)
2. `.agents/TODO.md` (Milestone 45 tasks 45.1-45.7; Milestone 46 is next)
3. `.agents/feature_crosswalk.md` sections 1, 2, 4.1-4.3, 4.7, and 8 (the spec for this session)
4. `README.md`, `CHANGELOG.md`

Reference implementations (read-only; never modify or import; say which repo you are reading):
- `../nfl-sos-ratings/nfl_sos_ratings/pbp_expressions.py`
- `../nfl-sos-ratings/nfl_sos_ratings/team_stats.py` (`compute_team_game_stats_from_pbp`,
  `compute_team_snap_counts_from_pbp`)
- `../nfl-sos-ratings/nfl_sos_ratings/team_stats_expanded.py` (`_aggregate_play_stats`)
- `../nfl-sos-ratings/nfl_sos_ratings/validation/snapshots.py`
  (`build_special_teams_game_frame_from_pbp`)

Owning modules in this repo (line numbers approximate as of 2026-09-09):
- `nfl_predictor/utils/polars/loaders.py`: `load_team_stats` (~334-417) is the caching pattern to
  copy; `load_pbp` (~657) and `aggregate_pbp_stats` (~679) exist but are unused and uncached.
- `nfl_predictor/data_collection.py`: `collect_all_data` (~289-483) joins team-game rows before
  `add_per_game_opponent_stats`; `process_week` (~579) aggregates with
  `aggregate_team_stats_to_week`.
- `nfl_predictor/utils/polars/teamrankings.py`: `aggregate_team_stats_to_week` (~346) means every
  numeric column; `_compute_derived_metrics` (~411) is where rates belong.
- `nfl_predictor/constants.py`: `NFLREADPY_STATS`, `EXCLUDE_FROM_OPPONENT_STATS`.
- `nfl_predictor/utils/polars/finalize.py`: `build_final_column_order` (invariant schema).
- `scripts/walk_forward_backtest.py` `_trend_feature_columns` (~33) and `--disable-trend-features`
  are the precedent for an ablation switch; `nfl_predictor/ml/walk_forward.py` holds
  `WalkForwardConfig`.
- Tests live flat under `tests/`; `tests/test_polars_loaders.py` (~424) already exercises
  `aggregate_pbp_stats` with a mocked frame.

## 1. Facts to trust unless your verification disproves them

- Baseline 2026-09-09: all gates green, `412 passed`, coverage `90.03%`; tracked-file diff is
  limited to `AGENTS.md` and `README.md` doc updates from the planning session.
- `data/completed_games_ml.csv` covers `1999-2025` (`7260` rows, `384` columns);
  `data/predict/week_01_games_to_predict.csv` has `16` rows for 2026 Week 1.
- Reference walk-forward (`2023-2025`, default config): Brier `0.2312`, log loss `0.7352`, pick
  accuracy `0.6833`, margin MAE `9.8954`, total MAE `10.1021`, ECE `0.1308`.
- nflreadpy `0.1.5` caches only in memory; the repo persists its own Parquet cache under
  `data/cache/nflreadpy/`. Current-season PBP may not exist before kickoff; `load_team_stats`
  already catches `ConnectionError` and falls back to cache or continues. Copy that pattern.
- `markdownlint` on this machine is `markdownlint-cli2` (`/usr/local/bin/markdownlint-cli2`).

## 2. Non-negotiables

- TDD: characterization or failing tests first, then production code, small diffs.
- No leakage: week-N features use only games strictly before week N of that season; playoff rows
  use the full regular season; a test must prove a future week's plays cannot change an earlier
  week's features.
- All matchups: every new column exists on every row; nulls when a source is missing (invariant
  schema even for a season with no PBP).
- Polars-first ETL; NumPy inside solvers; pandas only in ML modules.
- Rates are ratio of sums computed after season-to-date aggregation, never mean-of-game-rates.
- Name allowed/defensive metrics explicitly (`*_allowed_*`) and add them to
  `EXCLUDE_FROM_OPPONENT_STATS`.
- Cite the formula in each docstring; test each metric against a hand-built fixture; docstrings
  and type hints on everything; no new `noqa`/`type: ignore`/`pragma: no cover` without a real
  reason; no milestone numbers in code or docstrings.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python`/`pytest`/`ruff`.
- XGBoost margin/total is the only model family; no tuning campaigns; no Week 1 production pass.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`; leave `../nfeloqb/.bash_history` alone.
- Commit only if the user asks. If asked: one file per commit, imperative message, and end the
  message with the attribution line the harness provides.

## 3. Subagent rules (apply to every spawn)

- Every subagent receives an explicit file-ownership list and may not edit any other file. If it
  believes another file must change, it reports that instead of editing.
- All agents work in the main working tree and share the repo `.venv`. Do not use git worktrees:
  `.venv/` and `data/` live only in the main tree and would be missing there.
- Subagents run `ruff format` and `ruff check` only on the files they own, and run only their own
  test files. You run the repo-wide gate at each checkpoint.
- Each subagent's final report must contain: files changed, tests added (names), the last 15
  lines of its own pytest output, formulas or thresholds it chose, and open questions. Relay what
  matters to the user; the user does not see subagent reports.
- Use `model: "sonnet"` for mechanical work (test scaffolding, docstrings, doc edits, the ablation
  switch). Use the default Opus for design-bearing work (loader, aggregation, wiring, review).
- Run subagents in the background and wait for their completion notifications. Never predict or
  fabricate a pending agent's result. Use `SendMessage` to continue an agent that needs a
  follow-up so it keeps its context.
- Prefer two or three well-scoped agents over many small ones; context re-derivation is the
  expensive part.

## Phase 0 - Baseline and PBP schema survey (you, plus one background subagent)

1. Run `.venv/bin/python -m pytest -q` and record the actual pass count and coverage.
2. Confirm `data/completed_games_ml.csv`, `data/all_data.csv`, `data/qb_elos.csv`, and
   `data/cache/nflreadpy/` exist.
3. Spawn one background subagent (general-purpose, Sonnet), "pbp-schema-survey", owning only
   `.agents/pbp_schema_survey.md` and the scratchpad. Task: for seasons 1999, 2003, 2006, 2015,
   and 2024, load `nflreadpy.load_pbp(seasons=[season])` one season at a time, and for every
   column in `feature_crosswalk.md` section 8 report presence, dtype, null rate, and whether the
   special-teams flag is named `special` or `special_teams_play`; also report row count, distinct
   `season_type` values, and load time per season. It must not write under `data/`. Output: a
   Markdown table in `.agents/pbp_schema_survey.md`.
4. While it runs, read the owning modules listed in section 0 yourself. Do not edit anything the
   Phase 1 agents will own.
5. When the survey lands, freeze the column contract yourself (you own `constants.py` in this
   phase): add `PBP_COLUMNS` (the guarded selection list), `PBP_SPECIAL_TEAMS_FLAG_CANDIDATES`,
   and `FEATURE_GROUP_COLUMN_MARKERS: dict[str, tuple[str, ...]] = {"pbp": ()}` (filled in
   Phase 2), with a small test in `tests/test_constants.py`. Run the gate. Phase 1 agents read
   these constants and do not edit `constants.py`.

## Phase 1 - Parallel implementation (three background subagents, strict ownership)

Spawn all three after Phase 0 is green. Wait for all three before integrating.

Agent A, "pbp-loader" (Opus). Owns `nfl_predictor/utils/polars/loaders.py` (only `load_pbp` and
new private helpers) and new tests appended to `tests/test_polars_loaders.py` (do not modify
existing tests). Task: rewrite `load_pbp(seasons, *, cache_dir=None, force_refresh=False,
current_season=None, regular_season_only=True)` mirroring `load_team_stats`: per-season Parquet
cache `pbp_<season>_<reg|all>.parquet`, historical seasons from cache, current season refreshed,
`force_refresh` honored, `ConnectionError` on the current season falls back to cache or returns an
empty typed frame with a warning, historical failures raise; select `constants.PBP_COLUMNS` with
existence guards; normalize `posteam`, `defteam`, `home_team`, `away_team` via
`normalize_team_column`; regular-season filter when `season_type` exists. Leave
`aggregate_pbp_stats` untouched. Tests (mock `loaders.nfl.load_pbp` with `monkeypatch`): cache
hit, cache miss and write, force refresh, current-season failure with cache, current-season
failure without cache, missing optional columns tolerated, alias normalization, REG filter.

Agent B, "pbp-aggregation" (Opus). Owns new `nfl_predictor/utils/polars/pbp.py` and new
`tests/test_polars_pbp.py`. Task: `aggregate_pbp_team_game_stats(pbp_df) -> pl.DataFrame` with one
row per `(season, week, team_abbr, opponent_abbr)` and count/sum columns only (rates come later):
`offensive_snaps`, `defensive_snaps` (scrimmage snap = `qb_dropback + rush + qb_kneel + qb_spike
> 0`), `dropbacks`, `carries`, `pass_epa_sum` (EPA on dropback plays), `rush_epa_sum` (EPA on
rush plays), `pass_epa_allowed_sum`, `rush_epa_allowed_sum`, `pass_success_count`,
`rush_success_count`, and their allowed mirrors, `explosive_pass_count` (dropback play with
`yards_gained >= 20`), `explosive_rush_count` (rush with `yards_gained >= 12`),
`stuffed_rush_count` (rush with `yards_gained <= 0`), allowed mirrors of those three,
`early_down_plays`, `early_down_passes` (downs 1-2), `st_epa_for`, `st_epa_against`,
`st_plays` (special-teams flag from `constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES`), and the
third/fourth-down, red-zone, and two-point counts currently produced by
`loaders.aggregate_pbp_stats` (re-implement here; do not edit `loaders.py`). Thresholds are
module-level named constants. Also export `PBP_TEAM_GAME_COLUMNS`. Tests: a hand-built fixture of
roughly 16 plays across two games with known sums, every column asserted against hand computation;
the allowed columns equal the same-game opponent's offensive columns (mirror invariant); empty
input returns a typed empty frame; missing optional columns yield nulls, not crashes; non-REG rows
are excluded when `season_type` exists; kneels and spikes count as snaps but not dropbacks or
carries.

Agent C, "ablation-switch" (Sonnet). Owns `nfl_predictor/ml/walk_forward.py`,
`scripts/walk_forward_backtest.py`, `scripts/wf_compare.py`, and new tests in
`tests/test_walk_forward.py` and `tests/test_walk_forward_backtest_script.py`. Task: add
`disabled_feature_groups: tuple[str, ...] = ()` to `WalkForwardConfig` (serialized in `to_dict`),
a `--disable-feature-groups pbp,...` CLI option on both scripts, resolution of group names to
columns through `constants.FEATURE_GROUP_COLUMN_MARKERS` (a column belongs to a group when any
marker is a substring of its name), dropping those columns from train and eval frames the same
way `--disable-trend-features` does, and recording the disabled groups in the metrics report
metadata. Unknown group names raise a clear error. It reads `constants.py` but does not edit it.

Integration checkpoint 1 (you): run `.venv/bin/ruff format .`, `.venv/bin/ruff check .`,
`.venv/bin/pyright .`, `.venv/bin/ty check .`, `.venv/bin/python -m pytest`. Fix small
integration issues yourself; send larger ones back to the owning agent with `SendMessage`. Do not
start Phase 2 until green.

## Phase 2 - Sequential wiring (you, or one Opus subagent; no parallelism)

These are hotspot files; exactly one editor at a time. Ownership: `data_collection.py`,
`teamrankings.py`, `finalize.py`, `constants.py`, `loaders.py` (removal only), and their tests.

2.1 `constants.py`: add `PBP_STATS` (final published names for counts and rates, for example
`off_pass_epa_per_snap`, `off_rush_epa_per_snap`, `def_pass_epa_allowed_per_snap`,
`def_rush_epa_allowed_per_snap`, `epa_per_dropback`, `epa_per_carry`, `epa_margin_per_play`,
`pass_success_rate`, `rush_success_rate`, `success_rate_allowed`, `explosive_pass_rate`,
`explosive_rush_rate`, `stuffed_rush_rate`, `early_down_pass_rate`, `st_epa_margin_per_play`,
plus the allowed variants and the raw count columns you decide to publish); add every allowed name
to `EXCLUDE_FROM_OPPONENT_STATS`; set `FEATURE_GROUP_COLUMN_MARKERS["pbp"]` so the ablation
switch catches every new column and nothing old (test this explicitly, including `passing_epa`,
which is an existing column and must stay outside the group). Also add `rushing_epa` to
`NFLREADPY_STATS` as agreed in the crosswalk.
2.2 `collect_all_data`: load PBP for `stats_seasons` (which includes the prior season), aggregate
with Agent B's function, left-join onto `team_stats_df` on `(season, week, team_abbr)` before
`add_scoring_data_to_team_stats` so downstream steps see the columns; when PBP is empty, add the
columns as nulls so the schema is invariant. Log the per-season null rate at debug level.
2.3 `_compute_derived_metrics`: derive every rate from the season-to-date mean counts (mean of
per-game sums divided by mean of per-game snaps equals ratio of sums; say so in the docstring),
with zero-denominator guards returning null.
2.4 `finalize.py`: publish `PBP_STATS` in the ordered schema via `get_stat_columns` or a new
`get_pbp_columns`; update the schema, no-duplicates, and missing-data-policy tests.
2.5 Week 1: verify `calculate_league_means` and `regress_to_mean` cover the new columns with a
test that the regressed fallback is between the team value and the league mean.
2.6 Leakage test: synthetic two-week PBP fixture through `process_week` for week 2; perturb week
2 and week 3 plays; assert the week-2 feature rows are unchanged.
2.7 Remove `loaders.aggregate_pbp_stats` and retarget its existing test to the new module (grep
for other callers first).

Integration checkpoint 2: full gate green, coverage at or above `90%`.

## Phase 3 - Independent review (two background subagents, read-only)

Reviewer 1 (Opus): review `git diff` for leakage (any use of the current or future week), formula
correctness against the sos reference, schema invariance, and readiness-behavior regressions.
Reviewer 2 (Sonnet): docstrings, type hints, test quality, dead code, ruff/pyright cleanliness.
Both report findings with file and line; neither edits. Address findings yourself, then re-run the
gate.

## Phase 4 - Expensive validation (you, background Bash, monitored; run each once)

4.1 `.venv/bin/python -m nfl_predictor.data_collection --timing` for the default `1999-2026`
window. The first run downloads about 27 seasons of PBP; record the timing. A historical-season
failure is a bug to fix and rerun, not something to skip. Confirm the outputs still cover
`1999-2025` and the Week 1 2026 file still has 16 rows.
4.2 `.venv/bin/python scripts/leakage_audit.py` must pass.
4.3 Walk-forward, group on and off (check `--help` for exact flag names first; add
`--xgb-device cuda` only if `nvidia-smi` works):
    `.venv/bin/python scripts/walk_forward_backtest.py --eval-last-n-seasons 3 --xgb-tree-method
    hist --run-dir models/wf_m45_pbp_on`
    `.venv/bin/python scripts/walk_forward_backtest.py --eval-last-n-seasons 3 --xgb-tree-method
    hist --disable-feature-groups pbp --run-dir models/wf_m45_pbp_off`
    Run them in the background, one after the other, and wait for the notifications.
4.4 Report both runs' Brier, log loss, pick accuracy, margin MAE, total MAE, and ECE verbatim
from each `metrics_report.json`, next to the reference numbers in section 1.
4.5 Sanity checks on the rebuilt dataset: per-season null rates for the new columns; league means
that make football sense (passing EPA per dropback near zero to slightly positive in recent
seasons, rushing EPA per carry slightly negative, success rates roughly 40-50%); the allowed
columns for a team equal its opponents' offensive columns in aggregate.

## Phase 5 - Docs and handoff (you; a Sonnet subagent may draft README/CHANGELOG text)

- `README.md`: data sources (PBP + cache), implemented feature areas, walk-forward notes.
- `CHANGELOG.md`: new `0.3.0` entry (Changed, Added, Removed, Fixed), commit links only for
  commits that exist.
- `.agents/ARCHIVE.md`: Milestone 45 entry with the walk-forward table (on, off, reference), the
  ETL timing, and the null-rate summary.
- `.agents/TODO.md`: check off 45.x with notes; leave 46 as next; adjust 46 if Phase 4 changed
  what should be tried first.
- `.agents/next_agent_session_prompt.md`: regenerate for the Milestone 46 session in this same
  phased structure (Phase 1 candidates: the ridge solver module, the weekly snapshot builder, and
  the schedule-strength helper are disjoint new files).
- `markdownlint-cli2` on every touched Markdown file; then the full gate one last time.

## Stretch - Milestone 46.1 only (one Opus subagent, new files only)

If the gate is green and time remains: `nfl_predictor/utils/polars/adjusted_strength.py` with
`solve_team_ridge(team_games, response_col, *, ridge_lambda)` returning centered offense and
defense coefficients plus a home-field term (port the design of
`../nfl-sos-ratings/nfl_sos_ratings/simultaneous_adjustment.py::solve_team_stat_ridge`), with
tests: synthetic round-robin recovers known strengths, home-field sign, centering, empty input.
Do not wire it into the ETL.

## Final report to the user (structure)

1. Outcome first: what landed, gate status, and whether Phase 4 ran to completion.
2. Walk-forward table: group on, group off, reference.
3. Feature summary: column count added, per-season null rates, sanity-check results.
4. What was left out or deferred, and why.
5. The recommended first step for the Milestone 46 session.
If any phase did not finish, say so plainly and list the exact next command to resume.
```
