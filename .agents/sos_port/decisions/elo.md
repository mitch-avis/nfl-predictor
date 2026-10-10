# This repository's own Elo: decision log

Decisions taken while building the own Elo, the step-6 phase-4 candidate that would replace the
columns read from `data/qb_elos.csv` (`.agents/TODO.md`, "Roadmap Status", step 6, item 4). The
code is `src/nfl_predictor/utils/polars/elo_ratings.py` (the ratings and the feature builder),
`elo_game_context.py` (byes, travel, rolling home field), `elo_quarterbacks.py` (the VALUE
quarterback adjustment) and `elo_win_totals.py` (the win-total source).

Each entry gives what differs from FiveThirtyEight's (538's) published method or from the plan,
the options, the decision and its reasoning, the data behind it with the command that reproduces
it, the effect on feature values, how to reverse it, and who reviewed it. Sources: 538's "How Our
NFL Predictions Work" (saved as `~/scratch/elo_research/538_method.html`), its
`fivethirtyeight/nfl-elo-game` repository at `fbec1af` (`forecast.py`, `data/nfl_games.csv`), and
538's published game file as copied into `data/qb_elos.csv` (538's own numbers through the 2022
season, `../nfeloqb`'s from 2023).

## 1. The rule for keeping components

Written and committed before any variant was scored.

**What differs**: the plan lists four component families after the framework (538's 2019
additions, the quarterback adjustment, the win-total pull) and says the own Elo replaces the
`qb_elos.csv` columns only if it measures at least as well. Which components the recommended
setting keeps is decided here, without walk-forward runs, by scoring each variant's own pre-game
win probability.

**Options**: keep every 538 component because 538 kept it; keep a component only when it
improves this repository's games by more than chance; pick the best-scoring combination over all
subsets (selection bias over 64 combinations, and no stated order).

**Decision**: forward selection in the build order, one component at a time, on one fixed window,
with a stated interval test:

- Window: every regular-season game of 2007-2025 with a final score, from this repository's
  schedule frame (the cached nflverse schedules). The outcome is a home win; a tie counts as not
  one, as in the walk-forward.
- Score: each variant's own pre-game home win probability. Brier score governs; log loss guards.
- Paired difference: candidate minus current, per game, averaged; its 95% interval is the 2.5th
  and 97.5th percentiles of 5,000 resamples of the games with replacement (`numpy` generator seed
  `42`), the walk-forward's bootstrap.
- Seeding first: on the framework, 538's 1999 preseason ratings against every team at 1505 with
  a burn-in. Every team at 1505 is the simpler variant (no outside numbers); the seeds are kept
  only if they pass the test below.
- Then, starting from the framework with the chosen seeding, in this order: rest and bye, travel
  (with its 55-point base), playoff boost, rolling home field, quarterback adjustment, win-total
  pull. Each candidate is the components kept so far plus the next one.
- A candidate is kept if and only if its Brier difference interval lies entirely below zero and
  its log-loss difference interval does not lie entirely above zero. Otherwise it is dropped, the
  simpler variant stays, and the next component is tested against the components kept so far. An
  interval that touches zero is not below it.
- The playoff boost changes playoff games directly and regular-season games only through the
  ratings it carries into the next season. It is judged by the same rule on the same
  regular-season window; its effect on playoff games is reported beside it and governs nothing.
- The win-total pull is tested last. The market-free recommendation is the kept set before it,
  and the anchored recommendation adds it only if it passes.
- The comparisons with the `qb_elos.csv` probabilities (538's classic `elo_prob1` and its
  quarterback model's `qbelo_prob1` on the same games), the per-season and per-week breakdowns
  and the checks against 538's published numbers are reported beside the steps and decide
  nothing here: whether the own Elo replaces the `qb_elos.csv` columns is phase 4's walk-forward
  question.

**Reasoning**: the build order puts the components 538 found most robust first and the market one
last, so each candidate answers one question with the earlier answers fixed. A component that
cannot show an improvement larger than resampling noise on 4,943 games is not kept on 538's word
alone (`AGENTS.md` rule 9: when a measurement ties, the simpler setting is the recommendation).
Log loss guards against a component that sharpens probabilities into overconfidence.

**Data**: none; this entry precedes every number.

**Feature-value impact**: none by itself. The kept set becomes the recommended setting of
`build_own_elo_features`, which no ETL step calls yet.

**How to reverse it**: change the rule here before rerunning `.agents/sos_port/elo_eval.py`; the
script prints every step and every interval, so another rule can be applied to the same numbers.

**Review**: not yet reviewed.
