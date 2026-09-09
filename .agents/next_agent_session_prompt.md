# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is **Milestone 43** from
`.agents/TODO.md`: make the power rankings measure current-season strength, consuming the
schedule-adjusted snapshot that Milestone 46 just built.

Milestone 46 landed the weekly ridge snapshot this milestone consumes. Read its archive entry
first. Unlike the play-by-play milestone before it, **it improved the primary metric**: Brier
`0.2277` with the strength group on versus `0.2312` with it off and `0.2320` with both feature
groups off. The columns you are about to rank teams with are therefore known to carry real signal,
and the gain-based importance rank puts `adj_strength_composite_diff` 6th and `adj_srs_diff` 7th of
533 model features, behind only the three market columns and the two Elo diffs.

This milestone is smaller and lower-risk than the last two: it is a reporting change over columns
that already exist in the dataset, not new feature engineering. Do not turn it into one.

## 0. Read first, in this order

1. `AGENTS.md` (non-negotiables, command forms, readiness behaviors that must not regress)
2. `.agents/TODO.md`, Milestone 43 tasks 43.1-43.5, and the five follow-ups inherited from 46
3. `.agents/ARCHIVE.md`, the Milestone 46 entry (what exists, what it measured, what broke)
4. `.agents/feature_crosswalk.md` sections 3.1, 4.4, 4.5 and **6** (section 6 is the actual design
   recommendation for this milestone)
5. `README.md`, `CHANGELOG.md`

Owning modules:

- `scripts/power_rankings.py`: today fits Bradley-Terry over every season since 1999 with equal
  weights, fixed `0.97 / 0.03` targets, and future games filled with model probabilities.
- `nfl_predictor/reporting/power_rankings.py`
- `scripts/weekly_run.py`, `scripts/golden_command.py`
  (`_build_pregame_power_rankings` is the duplicate ranking artifact to label or retire)
- `nfl_predictor/utils/polars/strength_snapshot.py` (read-only for you: the snapshot builder)

## 1. Facts to trust unless your verification disproves them

- Baseline 2026-09-09, version `0.4.0`: all gates green, `548 passed`, coverage `90.8%`.
- `data/completed_games_ml.csv` is `7260` rows x `498` columns covering `1999-2025`;
  `data/predict/week_01_games_to_predict.csv` has `16` rows for 2026 Week 1.
- The strength columns you need are already in the dataset per team as `away_`/`home_`/`_diff`:
  `adj_off_pass_epa_snap`, `adj_off_rush_epa_snap`, `adj_def_pass_epa_snap`,
  `adj_def_rush_epa_snap`, `adj_srs`, `st_rating`, `adj_strength_composite`,
  `strength_games_played`, `sos_played_adj`, `sos_remaining_adj`, `sos_played_raw`
  (`constants.ADJUSTED_STRENGTH_STATS`).
- **A higher `adj_def_*` value is a BETTER defense.** The solve models a team-game as
  `offense[team] - defense[opponent]`, so the defense coefficient is what suppresses the opponent.
  Getting this sign backwards in a ranking would be the single easiest way to ship a wrong artifact.
- `adj_strength_composite` is standardized **within each snapshot**, so it is comparable across
  weeks and seasons. The raw `adj_*` columns are **not**: the frozen ridge penalty shrinks harder
  when fewer games have been played (about 30% of true magnitude at 4 games, about 50% by 17), so
  their scale drifts across a season. **Rank on the composite, not on the raw components.**
- Sanity anchor, 2024 pre-week-18: top five by composite BAL, DET, PHI, BUF, GB; bottom five TEN,
  NYG, JAX, NE, CAR. Spearman against current-season point differential `0.966`. If your ranking
  disagrees with this materially, your ranking is wrong, not the snapshot.
- Walk-forward at `--eval-last-n-seasons 3` takes about 35 minutes. A full ETL rebuild takes about
  480s warm. **You should need neither**: this milestone changes reporting, not the dataset. If you
  think you need an ETL rebuild, stop and re-read the scope.
- `markdownlint` on this machine is `markdownlint-cli2`
  (`markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings"`); CI runs `markdownlint .`.

## 2. Design decisions already made (do not relitigate; record deviations)

- Rank by the pre-week `adj_strength_composite` for `(season, through_week + 1)` rows, mapped onto
  the existing 1-10 and 0-10 scales, with the components published alongside the rank so the
  ranking is explainable and identical to the model's inputs (`feature_crosswalk.md` section 6).
- Keep Bradley-Terry reachable as `--method bradley_terry`, and keep `--legacy-franchise-fit` for
  the old all-seasons equal-weight behavior. Do not delete the franchise view; it answers a
  different question.
- Projected standings stay as they are: current record plus model win probabilities for the
  remaining schedule. That table is the right home for forward-looking information.
- The 43.1 Bradley-Terry defaults (window, prior-season weight, margin-based targets, excluding
  future model-probability rows) are still worth doing, because they improve the fallback method.
  Do 43.1 first: it is small, self-contained, and independent of the snapshot work.

## 3. Non-negotiables

- TDD: characterization or failing tests first, then production code, small diffs.
- No leakage: a week `N` ranking may only use the snapshot for week `N`, which is solved from games
  strictly before week `N`. Do not rank on end-of-season values and backfill them to earlier weeks.
- Polars-first ETL; NumPy inside solvers; pandas only in ML modules.
- Docstrings and type hints on everything; cite formulas; no milestone numbers in code or
  docstrings; no new `noqa`/`type: ignore`/`pragma: no cover` without a real reason.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python`/`pytest`/`ruff`.
- XGBoost margin/total is the only model family; no tuning campaigns.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Back up `data/*.csv` before any ETL run. `.venv/bin/python -m nfl_predictor.data_collection`
  overwrites the full dataset unconditionally, so a partial-window run is destructive.
- Walk-forward artifacts stay under `models/`. Every number you report must be readable from a
  `metrics_report.json` on disk.
- Commit only if the user asks. If asked: one file per commit, imperative message, and end the
  message with the attribution line the harness provides.

## 4. Follow-ups inherited from the schedule-adjusted strength milestone

Five items are open. **Only item A is in scope for this session** (it is a one-line schema change
you will be in the right file for anyway). The rest are recorded so a later session does not have
to rediscover them. Do not expand scope to chase B through E unless the user asks.

### A. `strength_games_played_diff` is a dead column (in scope)

Gain-based importance of exactly `0.0`. The two teams in a game have almost always played the same
number of games, so the diff is a constant zero outside bye weeks. The per-team
`away_`/`home_strength_games_played` columns do rank (12th and 17th of 533) and should stay.

**Recommendation:** drop the `_diff` companion only, via `get_stats_for_diff`. This changes the
schema from `498` to `497` columns, so it needs an ETL rebuild and a walk-forward re-run to
confirm no regression - which is why it is worth doing *now*, bundled with any other schema
change, rather than alone later.

### B. The raw adjusted components drift in scale across a season

The frozen ridge penalty (`STRENGTH_RIDGE_LAMBDA = 10.0`) shrinks coefficients harder when fewer
games have been played. Ordering is unaffected, but the same numeric value means a stronger team in
December than in September, which a tree splitting on an absolute threshold cannot reconcile. The
raw components rank at a median of 183 of 533 while the standardized composite ranks 6th.

**Recommendation:** measure a games-aware penalty (or publishing only the composite plus `adj_srs`)
against the recorded arms. This is a feature-engineering experiment, not a reporting change; give it
its own milestone rather than folding it into this one.

### C. `sos_played_raw` is null for weeks 1 and 2

11.7% of the dataset. In week 2, a faced opponent's only prior game is the one against the subject,
and the head-to-head exclusion removes it, so nothing remains to profile. This is the method being
correct, not a defect.

**Recommendation:** a documented fallback (the prior-season profile, or the adjusted lens) would
make the column usable in exactly the two weeks where schedule strength is least knowable. Ablate
the fallback against the recorded arms before keeping it.

### D. Schedule-strength columns are not bit-reproducible

Polars parallel `group_by` summation order moves the last 1-2 ULP on `sos_*` columns across
identical rebuilds; the ridge and SRS columns are exactly stable. Pre-existing - the same is true of
`aggregate_team_stats_to_week` - but it means the dataset fingerprint in the model artifact contract
changes across identical runs.

**Recommendation:** a line in the artifact-contract docs, not a code change.

### E. `uv sync --check --active` reports the environment is outdated

Predates this milestone and is unrelated to it: it was already failing when `pyproject.toml` and
`uv.lock` were untouched. Needs a plain `uv sync` to clear. `uv lock --check` passes.

## Phase 0 - Baseline (you)

1. Run `.venv/bin/python -m pytest -q` and record the actual pass count and coverage.
2. Confirm the strength columns are present in `data/completed_games_ml.csv` (expect `33`).
3. Run the current `scripts/power_rankings.py` for a late-season week and **save the output**, so
   you can show a before/after and prove the change did what you claim.

## Phase 1 - Bradley-Terry defaults (43.1)

Self-contained and independent of the snapshot. Add the window, prior-season weight, margin-based
targets, and future-row exclusion, with `--legacy-franchise-fit` reproducing today's output exactly.
Test that recency weighting shifts ratings toward recent results and that the legacy flag is a true
no-op against the saved baseline.

## Phase 2 - Rank on the adjusted composite (43.2)

Default the ranking to the pre-week composite, publish components alongside, keep
`--method bradley_terry`. The acceptance test that matters: **a synthetic breakout team ranks first
late in the season**, and the 2024 anchor above reproduces.

## Phase 3 - One canonical artifact (43.3), tests (43.4), docs (43.5)

Label or retire `golden_command._build_pregame_power_rankings`. Then the full gate, and only if you
made a schema change (item A), an ETL rebuild plus a walk-forward re-run against the four recorded
arms in `models/wf_strength_2023_2025_*/`.

## Final report to the user (structure)

1. Outcome first: what landed and gate status.
2. Before/after rankings for a known week, with the 2024 anchor as the correctness check.
3. Whether the ranking now reflects current-season strength, with evidence rather than assertion.
4. If you changed the schema: the walk-forward comparison against the four recorded arms.
5. What was left out or deferred, and why.
6. The recommended first step for the next session (Milestone 47, QB per-dropback families), with
   the exact command to resume anything unfinished.
