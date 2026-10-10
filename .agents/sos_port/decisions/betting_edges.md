# Decisions in decomposing the betting report's moneyline edges (roadmap step 6, phase 5)

The betting report (`src/nfl_predictor/reporting/betting_report.py`, written by the weekly run)
compares the model's home win probability with the market's no-vig moneyline probability and turns
the difference into a 1-10 confidence score and an action label (`PASS` to `STRONG`). The open
follow-up "Betting report edges" (`.agents/TODO.md`) observed that the largest of these edges
mostly reflect the gap between each game's spread and its moneyline, not anything the model adds.
Entry 1 is the rule, committed before any game was scored; entry 2 is its outcome. The report
stays diagnostic and makes no profitability claim.

## 1. The decomposition and the rule, written before scoring

All quantities are on the home side of one game, with `sigma` that week's floor sigma:

- `p_model = Phi(predicted_margin / sigma)`: the model's win probability (the deterministic floor).
- `p_ml`: the no-vig moneyline probability, from the report's own `_moneyline_to_implied_prob` and
  `_novig_pair`.
- `p_spread = Phi(-home_spread / sigma)`: the same curve at the market's margin.
- `edge = p_model - p_ml`: today's edge.
- `gap = p_spread - p_ml`: the curve-versus-moneyline gap, which carries no model information.
- `beyond = p_model - p_spread`: what the model adds beyond the spread, so `edge = beyond + gap`
  exactly.

The rule:

- **The columns are added in every case.** The report gains `market_home_prob_spread`
  (`p_spread`), `edge_beyond_spread_home_prob` (`beyond`) and `spread_moneyline_gap_home_prob`
  (`gap`) whatever the measurement shows, and every existing column and label stays unchanged.
- **The measurement decides only the recommendation about the labels.** If `|gap| > |beyond|` in
  more than half of the games in the top decile of `|edge|`, the recommendation is to base the
  confidence score and the action label on `beyond` instead of `edge`; that switch is held for the
  user. Otherwise the recommendation is to keep them on `edge`.
- **The top decile** is the `ceil(n / 10)` scored games with the largest `|edge|`.
- **The data**: the adopted-settings reference, seed 42 (`models/step4_reference/adopted_seed42`),
  read from its fold checkpoints (`models/wf_checkpoints/df20b775abf7f2657d13/`, 328 folds), on
  the 2007-2025 regular-season games that have both moneylines and a spread. Each fold's
  `floor_sigma` metric is that week's sigma. Before scoring, `Phi(predicted_margin / floor_sigma)`
  must reproduce the fold's deterministic probability to `1e-12`, or the measurement stops.
- **Reported beside the rule, deciding nothing**: the mean `|edge|`, `|gap|` and `|beyond|`; in
  the top decile, the share with `sign(gap) == sign(edge)`; the correlation of `edge` with `gap`
  and with `beyond`; and the Brier score of `p_ml` and of `p_spread` on the same games against
  the fold's `actual_home_win` (a tie scores 0, as in every walk-forward Brier here), with the
  paired difference bootstrapped over games (5,000 resamples, numpy seed 0, 95% percentile
  interval).
