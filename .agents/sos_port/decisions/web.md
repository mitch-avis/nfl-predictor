# Step 6 decisions: the web work that does not wait on the ratings port

Judgment calls made on 2026-10-10 while building Milestone 58's pool helpers (58.1), the
margin-only feature importance switch (58.6) and the live betting design (58.3), on the
implementer branch that fast-forwarded to `feat/step6-sos-port` at `3ab0bd1`. Each entry gives
what differs, the options, the decision, the reasoning, any data, how to reverse it and its review
state. Commit hashes are on that branch. Entries 9 to 15 record the fixes the independent review
of the same day asked for; where a later entry changes an earlier one, the earlier entry says so.

Reproduction commands that read `models/` run from a checkout whose `models/` holds the named run
(run directories are not in git), with this branch's code installed: `main` before the merge has
none of the pool modules or `margin_rows`.

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
  `pass_yards_diff`, `defensive_snaps_diff`, `away_2pt_conversions`). Reproduce with this branch's
  code (see the note at the top) from a checkout that holds the run:

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
  Entry 10 adds the exception for predictions rewritten after the lock, and entry 14 the order
  between a pinned run's ahead-of-time file and a newer run's own week.
- **Reasoning:** (a) leaves the Predictions page broken and the pool page reading different files
  from it; (c) changes where jobs write, which other pages and the weekly workflow rely on. (b) is
  local to the run index and the route resolution and changes no file on disk.
- **Data:** reproduced by `tests/api/test_run_weeks.py` (a week-5 weekly run holding a week-6
  file).
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
- **Data:** over the 192 regular-season weeks of 2015-2025 in `data/all_data.csv` (built
  2026-10-09), the week's top team score had median 44 (p10 38, p90 52) and the bottom median 3
  (p10 0, p90 9). For week 5 of 2026, `models/weekly_20261008_234130_eee65bf5`, the card gives a
  top of 42 (p10 36.5, p90 48.5) and a bottom of 3 (p10 0, p90 8.6): the bottom matches history,
  and the top runs about two points low with a narrower band, the normal tail being thinner than
  real, right-skewed scores. Recommended follow-up: calibrate the top-score estimate on
  walk-forward predictions before relying on it. Reproduce the card's numbers with this branch's
  code:

  ```bash
  .venv/bin/python -c "import polars as pl; from nfl_predictor.api.pool.tiebreakers import build_tiebreakers; card = build_tiebreakers(pl.read_csv('models/weekly_20261008_234130_eee65bf5/season_2026_week_05_predictions.csv', infer_schema_length=10000)); print(card.highest, card.lowest)"
  ```

  and the history with:

  ```python
  import polars as pl

  games = pl.read_csv(
      "data/all_data.csv",
      columns=["season", "week", "game_type", "away_score", "home_score"],
      infer_schema_length=10000,
  ).filter(
      pl.col("season").is_between(2015, 2025)
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

## 6. The survivor planner's objective and lookahead (search method superseded by entry 9)

- **What differs:** the plan asked for a beam search "with a future-value lookahead penalty so
  strong teams are saved" and ranked plans; it left the objective, the future value and the
  ranking open.
- **Options:** for the objective, the chance of surviving the planned weeks alone or every week
  with predictions; for the future value, a strength proxy (Elo or the power rankings), the
  teams' own win probabilities in the planned weeks, or the later predicted weeks; for the
  ranking, the search's top plans (often the same first pick) or the best plan per first pick.
- **Decision:** maximize the chance of surviving every regular-season week from the start week
  that has predictions, multiplying the published win probabilities as independent games, a tie
  losing like a loss. Plans come back one per first pick, ranked by that chance, and a team's
  future value is what the best assignment of the later predicted weeks loses without it. Games
  already final are not offered; weeks without predictions are listed, not guessed. The teams
  already used are kept in the browser per season, not on the server. The first build searched
  the planned weeks with a beam (`e08ec30`); entry 9 replaced it with an exact solution.
- **Reasoning:** the user acts on this week's pick, so one plan per first pick answers the real
  question. A strength proxy would tie the planner to the Elo or power-ranking columns that step 6
  replaces, and the planned weeks' own probabilities double count (with a single planned week the
  penalty cancels the pick's own chance exactly). The later weeks' predictions are the plan's
  stated input (Phase 3 files). Browser storage needs no schema change; the cost is that the used
  teams do not follow the user across devices.
- **Data:** none kept for the beam design; the timing and horizon figures first given here came
  from a benchmark that was not kept and are withdrawn. Entry 9 has the kept comparison.
- **How to reverse it:** see entry 9.
- **Review:** reviewed 2026-10-10 by the independent review; its findings are entry 9.

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

## 9. The survivor planner is solved exactly (a substitute method, on the user's approval list)

- **What differs:** the task text (`.agents/web_ui_plan.md`, Phase 4) asks for a beam search over
  (used-set, log survival) states with a future-value lookahead penalty. Entry 6's first build
  did that, searching the planned weeks with a beam of partial plans and leaving the weeks after
  them to an exact assignment; the independent review found its plans below the exact optimum.
  Replacing the task text's method is a substitute under `AGENTS.md` rule 2, so it goes on the
  user's approval list, and task 58.1 stays open until the user accepts it.
- **Options:** keep the beam with a wider default; or solve exactly: maximizing the product of
  independent win probabilities with each team used once is an assignment of weeks to teams at
  cost `-log p`, so fix each possible pick this week and assign every later week with
  predictions with `scipy.optimize.linear_sum_assignment`.
- **Decision:** solve exactly (decided by the delegating session after the review). Planned and
  later weeks differ only in display; `beam_width` and the caps on it and on `horizon_weeks` are
  gone, and the Survivor tab's search width control with them.
- **Reasoning:** the exact solution costs milliseconds, so the beam's approximation buys nothing,
  and a wrong first pick is the one mistake a survivor entry cannot take back.
- **Data:** the review simulated 30 seasons of 32 teams over 14 weeks with persistent ratings
  drawn from `Normal(0, 6)`, home win probability `Phi((r_home - r_away + 1.5) / 13.5)`, four
  byes a week in weeks 3 to 12 and seeds `default_rng(1000 + s)`: with 10 planned weeks and a
  beam of 64, 18 of 30 seasons got a best plan below the exact optimum, up to 8.6% below; with a
  beam of 16, the recommended first pick was wrong in 3 of 30. The kept script
  `.agents/sos_port/survivor_beam_vs_exact.py` simulates the same description with its own
  generator (so the counts differ) and a frozen copy of the beam: at 64, 9 of 30 below the
  optimum (largest shortfall 13.8%) with the right first pick in all 30; at 16, 17 of 30 below
  (largest 20.3%) and the first pick wrong in 4 of 30 (seasons 0, 5, 18, 25); the shipped exact
  planner matches the optimum in all 30 (largest gap 7e-16) at about 2 ms a season. Run it with
  `.venv/bin/python .agents/sos_port/survivor_beam_vs_exact.py`. `tests/api/test_pool_survivor.py`
  checks full simulated seasons against one global assignment, including seasons 0, 6, 8, 13 and
  25 that the beam got wrong.
- **How to reverse it:** revert `aa73764` and `fa2095a`.
- **Review:** the review's numbers are its own; the script's numbers are not yet reviewed.

## 10. A sheet after the lock comes from the picks entered before it

- **What differs:** entry 4 read the sheet from the week's predictions file. A lines refresh after
  Thursday's kickoff chains a new prediction that rewrites that file with new ranks, so the sheet
  scored picks nobody entered.
- **Options:** keep the predictions and only warn (entry 4); or, when the predictions were
  written after the lock and the run's picks file before it, build the sheet and its points from
  that picks file.
- **Decision:** the second (decided by the delegating session after the review). Kickoffs and
  final scores still come from the predictions; each game whose pick or points changed shows the
  newer ones as a note, and the sheet names the picks file and counts the changes. The picks route
  (the Predictions page's picks tab and the CSV download) serves the same pre-lock file. Without a
  picks file from before the lock, the sheet shows the newer ranks with the warning, as before.
- **Reasoning:** confidence picks are single-shot before the week's first game, so the picks
  entered then are the ones the pool scores.
- **Data:** `tests/api/test_pool_routes.py` (a week re-predicted after the fixture week's lock,
  with the picks file dated before and after it) and `tests/api/test_pool_sheet.py`.
- **How to reverse it:** revert `f5d3087` and `cc15481`.
- **Review:** not yet reviewed.

## 11. The survivor panel keeps each season's used teams

- **What differs:** `3b27071` keyed the panel by season in the Pool page, so the guard against
  writing one season's list over another's lived in the caller.
- **Options:** keep the key in the page, or move it into the panel.
- **Decision:** the panel keys its stateful part by season itself, and a vitest seeds the 2026
  list, renders the panel with no season and then 2026, toggles a team and checks that the stored
  list kept its earlier teams.
- **Reasoning:** the guard then holds wherever the panel is used.
- **How to reverse it:** revert `5719161`.
- **Review:** not yet reviewed.

## 12. A survivor request lists at most 32 used teams

- **What differs:** `used_teams` had no bound.
- **Options:** bound the list, and debounce the team toggles if a request were expensive.
- **Decision:** `max_length=32`, the number of teams in the league (a longer list is a 422); no
  debounce, since the exact planner costs a few milliseconds per request.
- **How to reverse it:** revert `94a4d27`.
- **Review:** not yet reviewed.

## 13. The frontend's plan labels are removed (a correction)

- **What differs:** `web/src/app/nav.ts` carried an unused `phase` field, a comment about it and
  `CURRENT_PHASE`; the Overview cards could say "Coming in phase N"; and the catch-all route's
  page said a page "lands in phase ? of the web UI plan". `AGENTS.md` forbids plan labels in code.
- **Decision:** remove them; the Overview cards drop their availability footer, and the catch-all
  route shows a not-found page ("This page doesn't exist.", the path, a link to the overview).
- **Reasoning:** a fact with one plausible fix that changes no model output (`AGENTS.md` rule 2).
- **How to reverse it:** revert `1ba8867`.
- **Review:** not yet reviewed.

## 14. A newer run's own week outranks a file predicted ahead into the active run

- **What differs:** after entry 2, a week predicted ahead into a pinned older run kept winning
  once a newer weekly run had predicted that week as its own: the newer run's week was hidden
  from the selector and outranked when asked for. The survivor planner reads such ahead-of-time
  files.
- **Options:** fix the order, or log it as a follow-up.
- **Decision:** fix it: a week resolves to the active run's own week, else the newest other run
  whose predictions stage wrote that week, else the active run's ahead-of-time file, else another
  run's newest file; the selector lists the same file for each week. A run whose own week is only
  its newest file (no stage marker) does not outrank the active run's file.
- **Reasoning:** a weekly run made for the week has its fresh data and lines; a file predicted
  weeks ahead does not. The rule is the review's ("the newest run that wrote a week by its own
  predictions stage wins").
- **Data:** `tests/api/test_run_weeks.py` (a pinned week-5 run holding week 6 beside a newer
  week-6 run, with and without its stage marker).
- **How to reverse it:** revert `74a879b`.
- **Review:** not yet reviewed.

## 15. Live odds poll on a credit budget

- **What differs:** entry 8's design polled scores and odds on one 60-second interval; that would
  spend The Odds API's free 500 credits a month in about one Sunday (about 630 requests).
- **Options:** a separate fixed odds interval, or a monthly credit budget spread over the game
  windows left in the month.
- **Decision:** the budget: `NFLP_ODDS_CREDITS_PER_MONTH` (default 500) over the remaining game
  windows, re-read from the `x-requests-remaining` header after each call, never faster than
  `NFLP_LIVE_ODDS_MIN_SECONDS` (default 120), stopping at zero. About one moneyline poll every 9
  minutes on the free plan through Thursday, Sunday and Monday games. Scores keep their own
  60-second interval.
- **Reasoning:** a budget adapts to a paid plan or a short month without retuning an interval.
- **How to reverse it:** edit the Phase 6 section of `.agents/web_ui_plan.md` (`76a756d`).
- **Review:** not yet reviewed.
