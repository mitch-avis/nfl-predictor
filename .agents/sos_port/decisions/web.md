# Step 6 decisions: the web work that does not wait on the ratings port

Judgment calls made on 2026-10-10 while building Milestone 58's pool helpers (58.1), the
margin-only feature importance switch (58.6) and the live betting design (58.3), on the
implementer branch that fast-forwarded to `feat/step6-sos-port` at `3ab0bd1`. Each entry gives
what differs, the options, the decision, the reasoning, any data, how to reverse it and its review
state. Commit hashes are on that branch.

## 1. The margin-only ranking covers every feature

- **What differs:** the Model page's payload carried, for each of the 40 features ranked over both
  heads, its margin head value. A switch that re-sorted those 40 rows would rank only the
  features the combined ranking already kept.
- **Options:** (a) re-sort the combined top 40 in the browser by the margin value; (b) rank every
  feature by the margin head on the server and return it as `feature_importance.margin_rows`;
  (c) a `head=margin` query parameter that reloads the Model page's payload.
- **Decision:** (b). The default view stays both heads added (the user's 2026-09-27 decision);
  the switch is hidden for runs that recorded no per-head numbers. The margin ranking uses the
  run's own measure (mean |SHAP|, else total gain, else the summed average gain) and the margin
  head's own split count.
- **Reasoning:** (a) drops margin features that the total head pushes out of the combined list;
  (c) refetches the whole Model payload on every flip. (b) adds 40 rows to the payload and the
  switch flips instantly.
- **Data:** on `models/weekly_20261008_234130_eee65bf5`, 5 of the margin head's top 25 features
  are outside the combined top 40 (`qb_value_pre_diff`, `home_opponent_red_zone_td_pct`,
  `pass_yards_diff`, `defensive_snaps_diff`, `away_2pt_conversions`). Reproduce from the main
  checkout:

  ```bash
  .venv/bin/python -c "from pathlib import Path; from nfl_predictor.api.readers.model import feature_importance as fi; d = fi(Path('models/weekly_20261008_234130_eee65bf5/feature_importance.json')); top = {r['feature'] for r in d['rows']}; print([r['feature'] for r in d['margin_rows'][:25] if r['feature'] not in top])"
  ```

- **How to reverse it:** revert `f30ac07`.
- **Review:** not yet reviewed.

## 2. The active run keeps its own week when later weeks are predicted into it

- **What differs:** the existing app (Phase 3). The `predict` job writes later weeks into the
  active run's directory, and the run index took the newest file as the run's week. Generating a
  later week therefore moved the active run to that week: its own week left the week selector,
  asking for it returned 404 "no predictions" (offering to generate it again, which would
  overwrite the weekly run's file), and the picks tab showed the own week's picks file under the
  later week. The survivor planner needs later weeks predicted, so it would have triggered this
  on every use.
- **Options:** (a) leave the index alone and have only the pool routes scan every week file;
  (b) fix the index: the run's own week is the file its predictions stage wrote
  (`predictions_state.json`, matched by file name since the stored paths go stale when a run
  directory moves), else the newest file as before, and every week file of the active run is
  listed and resolvable; (c) make the `predict` job write later weeks to `data/predict/` instead.
- **Decision:** (b), plus a rule for the picks file: it is served only when it is for the
  requested week and no older than that week's predictions; otherwise the picks are derived from
  the predictions (so a lines refresh followed by a re-predict no longer leaves stale picks).
- **Reasoning:** (a) leaves the Predictions page broken and the pool page reading different files
  from it; (c) changes where jobs write, which other pages and the weekly workflow rely on. (b) is
  local to the run index and the route resolution and changes no file on disk.
- **Data:** reproduced by `tests/api/test_run_weeks.py` (a week-5 weekly run holding a week-6
  file); checked on a scratch copy of `models/weekly_20261008_234130_eee65bf5` with weeks 6-8
  predicted into it.
- **How to reverse it:** revert `db81f1c` and `b66cbe2`.
- **Review:** not yet reviewed.

## 3. The week selector changes the week in one URL update (a correction)

- **What differs:** the existing Predictions page set the run, season, week and source URL
  parameters with four setters in a row. React Router builds each update from the last render's
  parameters, so only the last setter took effect: choosing another run's week or a week not
  predicted yet changed nothing.
- **Options:** fix it in place in the page, or move the hook to `hooks/useWeekParams.ts` (the
  Pool page needs it too) and fix it there.
- **Decision:** move it and rewrite the four parameters in one navigation; a failing test came
  first (`useWeekParams.test.tsx`).
- **Reasoning:** a bug with one plausible fix that changes no model output (`AGENTS.md` rule 2,
  "correcting is not narrowing").
- **How to reverse it:** revert `494c35e`.
- **Review:** not yet reviewed.

## 4. What the confidence sheet reads and how it scores

- **What differs:** the plan names a Confidence tab without saying what it reads.
- **Options:** read the weekly run's `*_confidence_picks.csv`, or the week's predictions file;
  for results, only the predictions file's score columns, or the ETL's `all_data.csv` too; for
  the lock warning, the predictions file's modification time or the predictions stage marker's
  `created_at`.
- **Decision:** the predictions file the Predictions page shows for the week (it carries the same
  ranks and is the newest after a re-predict); final scores from the predictions file first, then
  `all_data.csv` by `game_id` (a predictions file is written before its games and never updated);
  the lock is the week's first kickoff in US Eastern time, and the warning compares it with the
  file's modification time. Ranks that are not the unique `1..N` are reported and never repaired;
  a game's points are its rank, a tie scores for neither side, and points still possible are the
  points won plus the ranks of games not yet final.
- **Reasoning:** the pool helpers must not change a probability, pick or rank, so the sheet only
  reads and scores. The marker's `created_at` would miss a legitimate warning: a re-predict after
  the lock rewrites the file but not the marker. A copied run directory gets new modification
  times and can raise a false warning; the warning only advises.
- **How to reverse it:** `src/nfl_predictor/api/pool/sheet.py` and `results.py` (commit
  `e08ec30`).
- **Review:** not yet reviewed.

## 5. The tiebreaker card's games, entries and extreme scores

- **What differs:** the plan says "MNF and SNF games from `game_datetime` weekday and latest
  kickoff" and "highest and lowest projected team score".
- **Options:** for the games, fix them to the Sunday and Monday night games, or list every game
  and preselect those two; for the entries, the mean prediction or the quantile median, rounded
  per team or keeping the total; for the extremes, the best and worst projected team scores, or
  the distribution of the week's actual top and bottom score.
- **Decision:** every game listed, the latest Sunday and Monday kickoffs preselected (the latest
  other kickoffs fill a week without them), because Yahoo designates its two tiebreak games each
  week (help page SLN6629: total points, then home and road points, of the first game and then the
  second, then the highest and the lowest team score). Entries come from the unrounded mean
  predictions: the total rounds half up and the home side gets half of that total plus half the
  margin, rounded, so the entry keeps the rounded total, the first tiebreaker. The extremes are the
  median of the maximum and minimum of normal team scores (mean from the margin and total, spread
  from the quantile bands, margin and total errors independent, teams independent), floored at
  zero, with the projected top and bottom teams named beside them.
- **Reasoning:** the projections never reach the week's actual extremes: one of 30 teams nearly
  always has a big day, so the best projection (about 29 points) is a poor answer to "the score of
  the team that scores the most points".
- **Data:** over the 197 regular-season weeks of 2015-2025 in `data/all_data.csv`, the week's top
  team score had median 44 (p10 38, p90 52) and the bottom median 3 (p10 0, p90 9). On the copied
  week-5 predictions the card gives a top of 42 (p10 36.5, p90 48.5) and a bottom of 3 (0, 8.6):
  the bottom matches history, and the top runs about two points low with a narrower band, the
  normal tail being thinner than real, right-skewed scores. Recommended follow-up: calibrate the
  top-score estimate on walk-forward predictions before relying on it. Reproduce the history
  with:

  ```python
  import polars as pl

  games = pl.read_csv(
      "data/all_data.csv",
      columns=["season", "week", "game_type", "away_score", "home_score"],
      infer_schema_length=10000,
  ).filter(
      (pl.col("season") >= 2015)
      & (pl.col("game_type") == "REG")
      & pl.col("home_score").is_not_null()
  )
  teams = pl.concat(
      [
          games.select("season", "week", pl.col("home_score").alias("score")),
          games.select("season", "week", pl.col("away_score").alias("score")),
      ]
  )
  weekly = teams.group_by("season", "week").agg(
      pl.col("score").max().alias("top"), pl.col("score").min().alias("bottom")
  )
  for column in ("top", "bottom"):
      score = pl.col(column)
      quantiles = weekly.select(
          score.quantile(0.1).alias("p10"),
          score.median().alias("median"),
          score.quantile(0.9).alias("p90"),
      )
      print(weekly.height, column, quantiles.row(0))
  ```

- **How to reverse it:** `src/nfl_predictor/api/pool/tiebreakers.py` (commit `e08ec30`).
- **Review:** not yet reviewed.

## 6. The survivor planner's objective and lookahead

- **What differs:** the plan asks for a beam search "with a future-value lookahead penalty so
  strong teams are saved" and ranked plans; it leaves the objective, the future value and the
  ranking open.
- **Options:** for the objective, the chance of surviving the planned weeks alone or every week
  with predictions; for the future value, a strength proxy (Elo or the power rankings), the
  teams' own win probabilities in the planned weeks, or the later predicted weeks; for the
  ranking, the beam's top plans (often the same first pick) or the best plan per first pick.
- **Decision:** maximize the chance of surviving every regular-season week from the start week
  that has predictions, multiplying the published win probabilities as independent games, a tie
  losing like a loss. The beam runs once per possible pick this week over the planned weeks
  (`horizon_weeks`, default 4, at most 10; `beam_width` default 64, at most 256), keeping the best
  state per set of teams used. A team's future value is what the best assignment of the later
  predicted weeks loses without it (solved exactly by `scipy.optimize.linear_sum_assignment`),
  charged when the beam uses the team, and each finished plan is completed by that exact
  assignment so its later picks show. Plans come back one per first pick, ranked by the full
  chance. Games already final are not offered; weeks without predictions are listed, not
  guessed. The teams already used are kept in the browser per season, not on the server.
- **Reasoning:** the user acts on this week's pick, so one plan per first pick answers the real
  question. A strength proxy would tie the planner to the Elo or power-ranking columns that step 6
  replaces, and the planned weeks' own probabilities double count (with a single planned week the
  penalty cancels the pick's own chance exactly). The later weeks' predictions are the plan's
  stated input (Phase 3 files), and the exact assignment makes the horizon a display choice: on a
  random full season the best plan's chance was the same at 4, 6, 8 and 10 planned weeks. Browser
  storage needs no schema change; the cost is that the used teams do not follow the user across
  devices.
- **Data:** `tests/api/test_pool_survivor.py` checks every first pick's plan against exhaustive
  search on three random five-week seasons. Timing on a random 32-team, 14-week season: about 0.2
  s at the defaults and 2.3 s at the caps (`/tmp` benchmark, not kept); 0.5 s for weeks 5-8 on the
  copied weekly run.
- **How to reverse it:** `src/nfl_predictor/api/pool/survivor.py` (commit `e08ec30`); the caps are
  `MAX_PLANNED_WEEKS` and `MAX_BEAM_WIDTH` in `src/nfl_predictor/api/schemas/pool.py`.
- **Review:** not yet reviewed.

## 7. The picks download follows the sheet's week

- **What differs:** the plan's `GET /predictions/picks?format=csv` took the week by run, season and
  week only, while the sheet also takes `source=unattached`.
- **Options:** leave the download blind to unattached files, or let the picks route take `source`.
- **Decision:** the picks route takes `source` like the predictions and pool routes.
- **Reasoning:** the Pool page's download should hand over the file its sheet shows.
- **How to reverse it:** revert `3229b58`.
- **Review:** not yet reviewed.

## 8. The live betting design (58.3)

- **What differs:** the plan's one-line sketch: an odds-provider adapter and a "Live" blend with
  `w_time = min(1, minutes_remaining / 60)` and a sigma shrink.
- **Options:** for the in-game probability, Stern's random walk from the pregame prediction (score
  and clock only), a play-by-play model like nflfastR's `wp` (needs live play-by-play), or the
  live market's own price; for providers, free unofficial feeds or paid official ones.
- **Decision:** the random walk, made precise as `Phi((lead + r * mu) / (sigma * sqrt(r)))` with
  `r` the share of regulation left, `mu` the pregame margin and `sigma` the week's floor sigma, so
  it equals the published probability at kickoff; the market shown beside it, never blended; no
  action ladder for live markets; both providers off by default; a historical validation on
  play-by-play before anything is shown; three questions for the user before any build (the
  score source, the odds source, keeping odds snapshots under `data/`). Written into
  `.agents/web_ui_plan.md`, Phase 6.
- **Reasoning:** it needs only data a free scoreboard gives, matches the plan's sketch, keeps the
  repository's rule that the market enters only as features and anchoring, and adds no external
  service until the user picks one.
- **How to reverse it:** edit the Phase 6 section; nothing is built.
- **Review:** not yet reviewed.
