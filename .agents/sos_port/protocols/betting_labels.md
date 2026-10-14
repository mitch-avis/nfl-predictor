# Protocol: the betting report's labels on the whole edge or on the model's part

Written 2026-10-14 by the delegating session, before any game was scored for it. The user decided
on 2026-10-10 to run this test (`../../step6_plan.md`, "Betting labels") and confirmed it on
2026-10-11 (`../DECISIONS.md`, `0.57.2`): a pre-registered test of today's labels (the whole edge)
against labels on the model's part beyond the spread, governed by the games with real moneylines;
a tie keeps today's labels, and any switch goes back to the user.

## Question

The betting report turns each game's moneyline edge into a 1-10 confidence score and an action
label (`_edge_to_confidence_1_to_10`, `_edge_to_action` in
`src/nfl_predictor/reporting/betting_report.py`), with the value side from the edge's sign. Since
`0.57.2` the report splits that edge, `edge = beyond + gap` (`../decisions/betting_edges.md`,
entry 1): `beyond = p_model - p_spread` is what the model adds beyond the spread, and
`gap = p_spread - p_ml` is the gap between the normal curve at the market's spread and the no-vig
moneyline, which carries nothing from the model. Should the score and the label read `beyond`
instead of `edge`?

Each edge is a claim about the home side's probability against the moneyline: `edge` claims
`p_model`, and `beyond` claims `q_beyond = p_ml + beyond = p_model - gap`. Both label maps are
monotone in the edge's size, with the side from its sign, so the scheme whose claimed probability
predicts outcomes better is the one whose labels point at real deviations from the moneyline. The
two differ by exactly the gap: the test asks whether the gap is information or the normal curve's
misfit to the moneyline.

## Known before writing this

The edge measurement (`../BETTING_EDGES.md`, on the adopted-settings reference, seed 42) reported
beside its rule, deciding nothing, the Brier score of `p_spread` minus that of `p_ml`: a tie in
every group, `+0.00075` (`-0.00086` to `+0.00237`) on the 815 real-moneyline games. That compares
the two market baselines without the model; this test compares the two claims with it, on
production's reference after adoption, and has not been run on any data.

## Data

- **Runs**: the reviewed reference of the configuration production adopts (ported ratings,
  anchored; `models/step6/REVIEW.md`, decision 1): `models/step6/ratings_anchored_seed42` and
  `models/step6/ratings_anchored_seed7` in the main checkout, read from their fold checkpoints
  `models/wf_checkpoints/d97fab54e7261f18e349/` (seed 42) and
  `models/wf_checkpoints/2f7442e89fd6ff3981f4/` (seed 7), 328 folds each. Each fold's
  `floor_sigma` metric is that week's sigma; before scoring, `Phi(predicted_margin / floor_sigma)`
  must reproduce the fold's deterministic probability to `1e-12`, or the test stops.
- **Market**: the build's `models/step6/inputs/candidate_market_lines_metadata.json`, read as
  `betting_edges.py` reads its metadata: `p_ml` from the report's own `_moneyline_to_implied_prob`
  and `_novig_pair`, and the real-or-derived split from the metadata.
- **Governing games**: the 2007-2025 regular-season games with a real moneyline pair (prices, not
  derived from the spread through the fitted map) and a spread. The count is reported; it is
  expected near 815, the count on the earlier build, and any difference is explained from the
  metadata before scoring.
- **Outcome**: the fold's `actual_home_win`; a tie scores 0, as in entry 1 and every walk-forward
  Brier here.

## Instrument

- Per game and seed: `d = (q_beyond - y)^2 - (p_model - y)^2`, with `q_beyond` unclipped (a value
  outside `[0, 1]` is penalized by the square and counted in the report).
- Per game, `d` averaged over the two seeds (AGENTS.md rule 13), then the mean over games with a
  bootstrap over games: 5,000 resamples, numpy seed 0, 95% percentile interval.

## Decision rule

- **The interval lies wholly below zero** (`q_beyond` is the more accurate claim): the
  recommendation is to base the confidence score and the action label on `beyond`. The switch is
  held for the user; nothing in the report changes in this test.
- **Anything else** (the interval contains zero, or lies wholly above it): the labels stay on the
  whole edge.

## Reported beside the rule, deciding nothing

- Each seed alone, with its interval.
- The same difference on every scored game (real and derived moneylines) and on the derived ones
  alone.
- Log loss of both claims with each probability clipped to `[1e-6, 1 - 1e-6]`, paired and
  bootstrapped the same way.
- The difference within the top decile (`ceil(n / 10)` games) of `|edge|` and, separately, of
  `|beyond|`, on the governing games.
- How many of the governing games change action label, confidence score or value side when the
  labels read `beyond`, by seed.
- The difference by season, as a table.

## Implementation and review

A script beside `betting_edges.py` writes the report (`BETTING_LABELS.md` and a JSON record with
the SHA-256 of its code and every input file) and decision entry 3 in
`../decisions/betting_edges.md`; tests on synthetic folds cover the claims, the instrument, the
reproduction check and the rule's branches. An independent reviewer recomputes the governing
number from the fold checkpoints with its own script before any number enters a doc (AGENTS.md
rule 3).
