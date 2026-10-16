# Task: publish `qb_elos.csv`'s quarterback adjustment as the `qb_adj` feature group

The contract given to the implementer on 2026-10-15 (branch `feat/qb-adj-features`, version
`0.60.0`), kept here for the reviewer. Roadmap step 7, first item (`.agents/TODO.md`).

Decided with the user (2026-10-15): adopt `qb1_adj`/`qb2_adj` from `data/qb_elos.csv` as
features. Production keeps them off until the two-seed measurement in
`models/qb_elos_measure/HYPOTHESIS.md` decides, so this change must not alter what the weekly run,
`nfl-predictor train` or `nfl-predictor backtest` produce by default.

## Background

- `data/qb_elos.csv` (from `../nfeloqb`, `team1` is home) has `qb1_adj`/`qb2_adj`: the starter's
  QB value minus the team's offensive QB value, in Elo points (x3.3 already applied). Played rows
  come from nfeloqb's QB model, next-week rows from its projection path. It is built from
  box-score QB values only, with no betting data, so it is market-free (unlike the `elo` group,
  whose Elo pulls preseason ratings toward win totals).
- Before this task, `src/nfl_predictor/utils/polars/loaders.py` (`_QB_ELO_COLUMN_TYPES`,
  `load_elo_ratings`, `load_raw_elo_data`, `get_latest_elo_by_team`) read `elo1_pre`,
  `qb1_value_pre`, `qbelo1_pre` and their pairs, but not `qb1_adj`.
- Name collision to watch: this repo's own Elo (`utils/polars/elo_quarterbacks.py`,
  `utils/polars/elo_ratings.py`, attached through `_attach_own_elo` in `data_collection.py`)
  uses internal frames with columns named `home_qb_adj`/`away_qb_adj`. Those must never reach or
  overwrite the published columns.

## Contract (tests first; each case a failing test before the code)

1. Loader: `load_elo_ratings` reads `qb1_adj` as `home_qb_adj` and `qb2_adj` as `away_qb_adj`,
   Float64 (`team1` is home); a blank value reads as null. A `qb_elos.csv` copy without those
   columns still loads, handled the way the code already handles a missing `qb*_value_pre`.
2. Published dataset columns: `away_qb_adj`, `home_qb_adj` and `qb_adj_diff`, the diff built by
   the same mechanism and sign convention as `qb_value_pre_diff` and `elo_pre_diff`. The per-team
   stat is declared in `constants.py` as a new list next to `ELO_COLUMNS`; no hard-coded column
   lists elsewhere.
3. Upcoming weeks: the exact-week merge (`week_rows._merge_elo`) carries the file's next-week
   values; the fallback when the week has no rows (`get_latest_elo_by_team` and `_merge_elo`)
   carries each team's latest `qb_adj`; `game_utils.fill_future_qb_data` fills a null `qb_adj`
   from the latest starter row, as it fills `qb_value_pre`.
4. Collision: a test builds a game row with both the `qb_elos.csv` Elo and the own Elo attached
   and asserts `home_qb_adj`/`away_qb_adj` equal the `qb_elos.csv` values (no `_right` columns,
   no overwrite by the own Elo's internal adjustment).
5. Feature group: a new group `qb_adj` in `FEATURE_GROUP_COLUMN_MARKERS` whose markers are the
   new list, disjoint from every other group (a test covers the three new columns). Not in
   `MARKET_INFORMED_FEATURE_GROUPS`, with a one-line comment saying why. Added to
   `PRODUCTION_DISABLED_FEATURE_GROUPS`, so the weekly run, `train` and `backtest` leave it out by
   default; the comment there and the `disable_feature_groups` help text in
   `weekly_run/config.py` and `config/weekly_run.yaml` updated if they list the groups. A test
   pins that the default includes `qb_adj` and that `--disable-feature-groups
   strength,own_elo,own_elo_wt` keeps the three columns in the feature spec.
6. No leakage: each value is pre-game for the predicted week. If `nfl-predictor leakage-audit`
   keeps a list of allowed columns or families, the new ones are added with a test; if it is
   pattern-based and already covers them, the report names the file and line.
7. Out of scope: no four-week trend for `qb_adj`, no API or web display, no change to the own Elo,
   no tuning.

## Must not

- Rebuild, delete or write anything under `data/` or `models/`; run walk-forwards or the weekly
  run; touch `../nfeloqb`, `../nfl-sos-ratings` or the user's web server.
- Edit `.agents/TODO.md` or `.agents/next_agent_session_prompt.md`.
- Inline lint suppressions; tools other than `.venv/bin/*`.

## Version, docs, gate

- `0.60.0` (minor: schema change): `pyproject.toml`, a Common Changelog entry dated 2026-10-15,
  `uv lock`, then `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` in the worktree.
- Living docs that list feature groups or the ETL's Elo columns (README, `.agents/modeling_spec.md`,
  `.agents/feature_crosswalk.md` if it covers the nfeloqb QB fields), current state only.
- `scripts/gate.sh` with `VIRTUAL_ENV` set to the worktree's `.venv`, exit `0` on the final tree.
- Conventional Commits, scope `etl`; tests with or before the code they cover; the version bump
  and changelog as their own commit.
