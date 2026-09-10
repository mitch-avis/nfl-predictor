# Next Agent Session Prompt

You are the orchestrating agent (Claude Opus 5) for an implementation session in the `nfl-predictor`
workspace (`/home/mitch/workspace/nfl-predictor`). Your deliverable is **Milestone 43, phase 2**
(tasks 43.2 through 43.5) from `.agents/TODO.md`: default the power rankings to the schedule-adjusted
composite the ETL already publishes, leave one canonical ranking artifact, and document the result.

Phase 1 (43.1) landed on 2026-09-09 and is committed. Read the Milestone 43 section of
`.agents/TODO.md` for exactly what it did and what it deliberately left undone.

## A recommendation the user should see before you start

The user directed this milestone next, so deliver it. But say plainly in your first response that
**the highest-expected-value work in the backlog is the early-season shrinkage defect**, not this
milestone. Phase 1 already fixed the egregious ranking bug (a 4-13 team ranked first). Phase 2 is a
refinement of a reporting artifact that no longer feeds anything else. The shrinkage defect changes
real predictions on two weeks of every season. Both are written up in `.agents/TODO.md`; let the
user decide, and do not silently reorder.

## 0. Read first, in this order

1. `AGENTS.md` (non-negotiables, command forms, readiness behaviors, the validated baseline)
2. `.agents/TODO.md` in full. Three defects sit above Milestone 43 in that file and all three were
   found by measurement rather than by reading code:
   - early-season shrinkage (week 2 is the weakest week of the season)
   - the total/over-under model carries almost no signal
   - the remaining follow-ups inherited from the strength milestone
3. `.agents/ARCHIVE.md`, the Milestone 46 entry (the snapshot you are about to rank on)
4. `.agents/feature_crosswalk.md` section 6 (the actual design recommendation for this milestone)
5. `README.md`, `CHANGELOG.md`

Owning modules:

- `scripts/power_rankings.py` and `nfl_predictor/reporting/power_rankings.py` (phase 1 changed both)
- `scripts/weekly_run.py`, `scripts/golden_command.py`
- `nfl_predictor/utils/polars/strength_snapshot.py` (read-only for you)

## 1. Facts to trust unless your verification disproves them

- Baseline 2026-09-09, version `0.4.0`: all gates green, `558 passed`, coverage `90.83%`.
  `uv sync --check --active` fails; that predates this work and needs a plain `uv sync`.
- `data/completed_games_ml.csv` is `7260` rows x `498` columns covering `1999-2025`.
  `data/predict/week_01_games_to_predict.csv` holds `16` rows for 2026 Week 1.
- The strength columns are already published per team as `away_`/`home_`/`_diff`
  (`constants.ADJUSTED_STRENGTH_STATS`). You are consuming them, not building them.
- **A higher `adj_def_*` value is a BETTER defense.** The solve models a team-game as
  `offense[team] - defense[opponent]`, so the defense coefficient is what suppresses the opponent.
  Inverting this would ship a confidently wrong ranking.
- **Rank on `adj_strength_composite`, never on the raw `adj_*` columns.** The composite is
  standardized within each snapshot. The raw components are not: the frozen ridge penalty shrinks
  harder when fewer games have been played (about 30% of true magnitude at 4 games, about 50% by
  17), so their scale drifts across a season.
- Sanity anchor, 2024 pre-week-18. The adjusted composite ranks BAL, DET, PHI, BUF, GB at the top
  and CAR, NE, JAX, NYG, TEN at the bottom, with Spearman `0.966` against current-season point
  differential. Phase 1's Bradley-Terry default independently produces DET, BAL, BUF, GB, PHI.
  Those two orderings agreeing is your correctness signal; if your phase 2 output disagrees with
  both, your ranking is wrong.
- Phase 1 evidence, useful as a regression anchor: `--legacy-franchise-fit` for 2024 week 18 returns
  NE, PIT, GB, BAL, IND (the old all-seasons franchise fit) and is pinned by a test.
- A full ETL rebuild takes about `480s` warm. A walk-forward at `--eval-last-n-seasons 3` takes
  about 35 minutes from week 3, about 40 from week 1. **You should need neither.** This milestone
  changes reporting only. If you think you need an ETL rebuild, re-read the scope first.
- `markdownlint` here is `markdownlint-cli2`
  (`markdownlint-cli2 "**/*.md" "#.venv" "#nfl-sos-ratings"`); CI runs `markdownlint .`.

## 2. Design decisions already made (do not relitigate; record deviations)

- Rank by the pre-week `adj_strength_composite` for `(season, through_week + 1)` rows, mapped onto
  the existing 1-10 and 0-10 scales, publishing the components next to the rank so the ranking is
  explainable and identical to the model's inputs (`feature_crosswalk.md` section 6).
- Keep Bradley-Terry reachable as `--method bradley_terry`, and keep `--legacy-franchise-fit`. Do
  not delete the franchise view; it answers a different question and a test pins it.
- Projected standings stay as current record plus model win probabilities for the remaining
  schedule. That table is the right home for forward-looking information.
- Where the snapshot has no value (week 1 of the first season, or a team with no rows), fall back to
  the Bradley-Terry rating rather than emitting a null rank, and say so in the output.

## 3. Non-negotiables

- TDD: characterization or failing tests first, then production code, small diffs.
- No leakage: a week `N` ranking may only use the snapshot for week `N`, which is solved from games
  strictly before week `N`. Do not rank on end-of-season values and backfill them to earlier weeks.
- Polars-first ETL; NumPy inside solvers; pandas only in ML and reporting modules.
- Docstrings and type hints on everything; cite formulas; no milestone numbers in code or
  docstrings; no new `noqa`/`type: ignore`/`pragma: no cover` without a real reason.
- All Python tooling via `.venv/bin/...`; `uv` from PATH; never bare `python`/`pytest`/`ruff`.
- XGBoost margin/total is the only model family; no tuning campaigns.
- Do not modify `../nfeloqb` or `../nfl-sos-ratings`.
- Back up `data/*.csv` before any ETL run; the ETL overwrites unconditionally.
- Walk-forward artifacts stay under `models/`. Every number you report must be readable from a
  `metrics_report.json` on disk.
- Commit only if the user asks. If asked: one logical change per commit, imperative message, ending
  with the attribution line the harness provides.

## 4. Open follow-ups

Everything below is recorded in `.agents/TODO.md` with fuller detail. **Only item A is in scope for
this session.** Do not chase the others unless the user asks.

### A. `weekly_run.py` does not expose the phase 1 flags (in scope)

`scripts/weekly_run.py` calls `power_rankings._build_games_for_ratings` without the new arguments,
so it silently inherits the new defaults (two-season window, prior weight `0.25`, margin targets,
no future rows) but offers no way to override them and has no `--legacy-franchise-fit`. Wire the
flags through while you are in that file for 43.3.

### B. Early-season shrinkage (highest value in the backlog, out of scope here)

Week 2 is the weakest week of the season (`0.5208` pick accuracy against `0.6958` for weeks 3-18)
because season-to-date features there are unshrunk one-game means: `games_played` is `17` in week 1
(the regressed prior season) but `1` in week 2. Week 1, by contrast, is the **best-calibrated** week
in the season. Affects every season-to-date family, not just strength. Needs a full rebuild plus a
`--wf-start-week 1` walk-forward to validate, reporting weeks 1, 2 and 3-18 separately.

### C. The total/over-under model carries almost no signal (out of scope here)

Predicted totals for all 16 Week 1 games land in `43.9`-`44.1` against market totals of `40.5`-`47.5`;
holdout `total_mae` is `10.9974`. The total columns of the betting workbook are consequently not
actionable. Do not present total-based betting recommendations as usable until this is investigated.

### D. Smaller items

`strength_games_played_diff` has gain-based importance of exactly `0.0` (drop the `_diff` companion,
keep the per-team columns); `sos_played_raw` is null for weeks 1-2 by construction; schedule-strength
columns are not bit-reproducible across identical rebuilds (Polars parallel `group_by` ordering,
last 1-2 ULP); `uv sync --check --active` needs a plain `uv sync`.

## Phase 0 - Baseline (you)

1. `.venv/bin/python -m pytest -q` and record the actual pass count and coverage.
2. Run the current ranking for 2024 week 18 under all three modes (new default,
   `--legacy-franchise-fit`, and once 43.2 exists, `--method bradley_terry`) and **save the output**,
   so every later claim is a before/after you can show rather than assert.

## Phase 1 - Rank on the adjusted composite (43.2)

Default the ranking to the pre-week composite; publish the components alongside; keep
`--method bradley_terry`. The acceptance tests that matter: a synthetic breakout team ranks first
late in the season, and the 2024 anchor above reproduces.

## Phase 2 - One canonical artifact and docs (43.3, 43.4, 43.5)

Label or retire `golden_command._build_pregame_power_rankings`. Wire the phase 1 flags through
`weekly_run.py` (item A). Update `README.md` and `--help` to explain "current-season" versus
"franchise" rankings. Then the full gate.

## Final report to the user (structure)

1. Outcome first: what landed and gate status.
2. Before/after rankings for a known week, with the 2024 anchor as the correctness check.
3. Whether the ranking now reflects current-season strength, with evidence rather than assertion.
4. What was left out or deferred, and why.
5. Your recommendation for the next session, which should almost certainly be the early-season
   shrinkage defect (item B) unless the user has said otherwise.
