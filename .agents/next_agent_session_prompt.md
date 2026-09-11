# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is **Milestone 51: power
rankings on the adjusted composite** (51.1 through 51.5 in `.agents/TODO.md`). If it lands with time
to spare, start the **Milestone 52** diagnosis (52.1, the flat total head).

Milestone numbers changed on 2026-09-10: the worklist was renumbered into execution order, and the
old-to-new map is at the top of `.agents/ARCHIVE.md`. Milestone 51 was "43 phase 2" and Milestone 52
was "50". Older commits, changelog entries and `feature_crosswalk.md` use the old numbers.

## The one thing to settle in your first response

51.1 has a design fork, written up in full under 51.1 in `.agents/TODO.md`. A team on bye in week
`through_week + 1` has no game row that week, and its next row leaks that week's results into a
historical rerun. The options:

- **(a)** use the team's latest row at or before `through_week + 1`: leak-free, no new data path,
  one game stale for bye teams;
- **(b)** solve the snapshot inside the reporting script: exact, but it means factoring the
  per-team-game frame out of `collect_all_data` and giving a reporting script network access and
  a second route to the same numbers;
- **(c)** have the ETL write the per-team weekly snapshot it already computes (bye teams included)
  to a new file such as `data/strength_snapshots.csv`: exact, offline, consistent with the model's
  features by construction; costs one ETL rebuild of about 10 minutes and a new artifact to
  document.

The previous session explained these to the user and **recommended (c)**; the user was leaning
toward (b) or (c). If the user's opening message names a choice, take it and proceed. If it does
not, restate the three options in a few lines with the (c) recommendation, ask them to choose, and
read code while you wait. Everything else below is decided.

## 0. Read first, in this order

1. `AGENTS.md`: non-negotiables, command forms, the current benchmark (measured 2026-09-10 on the
   blend build), and the walk-forward operating notes (one run at a time, checkpoints, OpenMP wait
   policy).
2. `.agents/TODO.md`: Milestone 51 in full, then Milestone 52 and the open follow-ups.
3. `.agents/ARCHIVE.md`: the renumbering map, "Milestone 43 phase 1" (what the Bradley-Terry
   defaults already do), and the Milestone 49 entry (how the last session measured, and what it
   got wrong about week 1).
4. `nfl_predictor/data_collection.py`: `process_week` around the `strength_features` substep,
   `build_strength_features`, `_schedule_teams`, `_merge_strength_features`, and where the CSV
   outputs are written (`write_csv`). With option (c), this is where the snapshot file comes from.
5. `nfl_predictor/utils/polars/strength_snapshot.py` (`build_strength_snapshot`, `_with_composite`,
   `COMPOSITE_WEIGHTS`) so you know what `adj_strength_composite` is before you rank on it.
6. `scripts/power_rankings.py` (`_build_games_for_ratings`, `main`),
   `nfl_predictor/reporting/power_rankings.py` (`fit_bradley_terry_ratings`,
   `scale_ratings_1_to_10`, `ratings_to_power_0_to_10`, `build_power_rankings_and_standings`),
   `scripts/weekly_run.py` around the power-rankings stage, and
   `scripts/golden_command.py::_build_pregame_power_rankings`.
7. Tests: `tests/test_power_rankings.py`, `tests/test_power_rankings_script.py`,
   `tests/test_golden_command_power_rankings.py`, and `tests/test_stat_prior_blend.py` for how the
   last session tested `process_week` and `process_season`.

## 1. Facts to trust unless your verification disproves them

- Version `0.5.0`, committed and tagged locally on 2026-09-10 (the tag is not pushed). All gates
  green: `590 passed`, coverage `91.04%`. The working tree should be clean apart from gitignored
  data; check `git status` first.
- `data/completed_games_ml.csv` is the stat-prior-blend build (`7261` rows including the 2026
  opener, fingerprint `e46f1be9...`); `data/all_data_ml.csv` carries the future 2026 rows with
  `away_`/`home_adj_strength_composite`. Back up `data/*.csv` before any rebuild (option (c) needs
  one); keep the backup alongside the existing `*.pre_m49.csv` files.
- `process_week` builds the strength snapshot for every team on the season's schedule
  (`teams=_schedule_teams(...)`), so bye teams are solved every week and only dropped when the
  snapshot is joined onto game rows. Verify this before relying on it.
- Rank on `adj_strength_composite`, never on the raw `adj_*` columns: the frozen ridge penalty makes
  the raw columns drift in scale across a season, while the composite is standardized within each
  snapshot. A **higher** `adj_def_*` is a **better** defense.
- The 2024 pre-week-18 anchor: BAL, DET, PHI, BUF, GB on the composite; DET, BAL, BUF, GB, PHI on
  the current Bradley-Terry default. Use it as a sanity check, not as a test oracle.
- The composite is a within-snapshot z-score, not a win probability. Mapping it onto the existing
  1-10 and 0-10 scales needs a documented transform; do not reuse `sigmoid(rating_raw)` blindly.
- `--method bradley_terry` must keep today's default output reachable, and `--legacy-franchise-fit`
  must keep reproducing the old franchise fit exactly (a test pins it).
- Adding a snapshot file must not change the training rows. After a rebuild, confirm
  `completed_games_ml.csv` matches the previous build except for the known last-ULP drift in the
  schedule-strength columns (see the Milestone 46 follow-ups); if anything else moves, stop and
  find out why. No walk-forward is needed for Milestone 51.
- If you run a walk-forward anyway (Milestone 52), read the `AGENTS.md` operating notes first: one
  run at a time, check `uptime` and choose the OpenMP wait policy from it, and a stopped run resumes
  by re-running the identical command.
- `.agents/skills/` is the user's separate clone of agent skills. It is gitignored and excluded from
  ruff and markdownlint; never edit it.

## 2. Decided (do not relitigate; record deviations)

- 51.1 defaults to the composite with the components published next to the rank; Bradley-Terry
  stays as `--method bradley_terry`.
- 51.4: the phase-1 flags (`--ratings-window-seasons`, `--ratings-prior-season-weight`,
  `--ratings-target`, `--ratings-include-future`, `--legacy-franchise-fit`) and the new `--method`
  get wired through `scripts/weekly_run.py`.
- 51.2: projected standings stay record plus model win probabilities; label or retire
  `golden_command._build_pregame_power_rankings` so one ranking artifact is canonical.
- 51.5 extends the README and `--help` text written in phase 1; it does not redo it.
- New work that is not part of an existing milestone takes number 58 onward; never renumber
  existing milestones (see the numbering rules at the top of `.agents/TODO.md`).

## 3. Non-negotiables

- TDD: failing or characterization tests first, then production code, small diffs.
- No leakage: a week-`N` ranking uses only games through week `N - 1` in that season.
- Polars-first in ETL; pandas is acceptable in the reporting module as it already is.
- Docstrings with formulas; type hints; no milestone numbers in code, comments, or tests; no new
  `noqa` / `type: ignore` / `pragma: no cover` without a real reason.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python` / `pytest` / `ruff`.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Commit only if the user asks. If asked: one logical change per commit, Conventional Commits
  subject (`type(scope): imperative summary`), a body explaining what and why, and the attribution
  line the harness provides. Do not push commits or tags unless asked; pushing a version tag
  publishes a GitHub release.

## Phase 1 - 51.1 and 51.3 (tests first)

1. Tests: the composite ranking for a synthetic week ranks the obviously strongest team first; a
   bye team is handled by the chosen option and never reads a later week's data; the 1-10 and 0-10
   mappings are monotone and bounded; `--method bradley_terry` reproduces today's output; a
   synthetic breakout team ranks first late in the season. With option (c), also test the snapshot
   writer: one row per scheduled team per week, bye teams included, schema stable when a season
   has no games yet.
2. Implement the chosen option, the scale mapping, and the published components.
3. Check the 2024 pre-week-18 top five against the anchor above and the 2026 Week 1 output for
   sanity (every team present, no nulls).

## Phase 2 - 51.2, 51.4 and 51.5, then the gate

Wire the flags through `weekly_run.py`, settle the canonical artifact, update `README.md` (and the
data-files list if option (c) adds a file), `AGENTS.md` (the `scripts/power_rankings.py` entry),
`CHANGELOG.md` `[Unreleased]`, and move Milestone 51 to `ARCHIVE.md`. Then the full gate (commands
in `AGENTS.md` and `.agents/TODO.md`; the local markdownlint command excludes `#.agents/skills`).

## Phase 3 - Only if time remains: 52.1

Diagnose why predicted totals sit between `43.9` and `44.1` for every 2026 Week 1 game: total-head
feature importance, early-stopping round, train versus holdout MAE, and whether pruning drops
total-relevant columns. Diagnosis only; record findings under Milestone 52 in `.agents/TODO.md`.

## Final report to the user (structure)

1. Outcome first: which 51.1 option landed, the new default ranking, and gate status.
2. The 2024 pre-week-18 top ten under the composite and under `--method bradley_terry`.
3. What `scripts/weekly_run.py` now produces for power rankings, and which flags it exposes.
4. What was left out or deferred, and why.
5. Recommendation for the next session: Milestone 52, then 53, then the Milestone 49 follow-ups
   (the `games_played` effective-sample column first).
