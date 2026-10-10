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

## 2. The outcome: the columns are added and the labels stay on the whole edge

- **What differs**: three things, none of them the rule.
  - The weekly run's sigma comes from the predictions, not from `_production_floor_sigma`. That
    function (`src/nfl_predictor/weekly_run/pipeline.py`) returns the final fit's error pool and
    the week, not a sigma; the sigma is estimated at the final fit and written to every
    prediction row as `floor_sigma` (`src/nfl_predictor/ml/ml_model_predict.py`), and it is the
    one behind `home_win_prob`. Stage 4 passes that column's single value
    (`pipeline._report_floor_sigma`), and none when the column is missing or holds several.
  - The tests follow the `tests/test_sos_port_*.py` pattern of a parallel, unmerged branch. Its
    loader (`tests/sos_port_scripts.py`) was never on the base branch; it is copied unchanged
    from that branch's commit `92b6342` and is to be reconciled with it at merge.
  - The weekly-run snapshot (`tests/fixtures/weekly_run_characterization/betting_report.csv`)
    gains the three columns; every other column is unchanged, which was checked by comparing
    the old and new snapshots without them.
- **Options**: (a) keep the confidence score and the action label on the whole edge and show the
  split beside it; (b) base both on `beyond`, the model's part, so a game's label ignores the
  spread-moneyline gap; (c) report two sets of labels.
- **Decision**: the rule's first branch fails (`|gap| > |beyond|` in 14.3% of the top decile,
  against more than half), so the recommendation to the user is (a): keep the labels on `edge`.
  The three columns are in the report either way, and no label changed.
- **Reasoning**: on 19 seasons of walk-forward games the largest edges are mostly the model's
  own disagreement with the spread, not the gap: `edge` correlates `0.78` with `beyond` and
  `0.44` with `gap`, and the mean `|beyond|` (`0.0339`) is larger than the mean `|gap|`
  (`0.0263`). The gap usually leans the same way as the edge (82.2% of the top decile), so it
  enlarges large edges without making them. The floor's curve at the spread and the no-vig
  moneyline score alike: their Brier difference has an interval that holds zero. The fold rows
  carry the pick-time lines the walk-forward anchors to (for a completed game, nfelo's opening
  line, `AGENTS.md` rule 11), which is the revisit the follow-up asked for. The Week 3
  observation that started it was one week; on this sample it does not hold in general. (b) is
  the branch the rule reserved for a gap that outweighs the model's part, which did not happen,
  and (c) doubles the labels; (a) stands, as the rule says. Most scored games have moneylines
  derived from the spread (see **Data**), but on the games with real moneyline pairs the gap
  still outweighs the model's part in under half of the top decile. How many labels (b) would
  move is in the review line below.
- **Data**: `.agents/sos_port/BETTING_EDGES.md`, written by `.agents/sos_port/betting_edges.py`
  from the adopted-settings reference, seed 42 (`models/step4_reference/adopted_seed42`, fold
  checkpoints `models/wf_checkpoints/df20b775abf7f2657d13/`, 328 folds). 4,943 game rows, all
  2007-2025 regular-season games with both moneylines and a spread, so all are scored; the floor
  check's largest difference is `0`. Mean `|edge|` `0.0384`, `|gap|` `0.0263`, `|beyond|`
  `0.0339`; top decile 495 games, `|gap| > |beyond|` in `0.1434`, `sign(gap) == sign(edge)` in
  `0.8222`; correlation of `edge` with `gap` `0.4363` and with `beyond` `0.7814`; Brier of
  `p_ml` `0.21286` and of `p_spread` `0.21337`, difference `+0.00051` (95% interval `-0.00026`
  to `+0.00126`). Every fold's fingerprint begins with the checkpoint directory's name.
  - 4,128 of the 4,943 games, in seasons 2007-2021 and 2023, have no real moneyline: their
    moneylines come from the opening spread through the fitted spread-to-moneyline map
    (`src/nfl_predictor/utils/polars/pick_time_lines.py`; `moneylines_from_map` in
    `models/step4_reference/inputs/adopted_market_lines_metadata.json`; it holds per-season
    counts, not per-game flags, and for 2023 it counts 283 of 285 games, playoffs included, so
    2023 is grouped whole with the derived seasons). Season 2022 uses stored closing lines. On
    those 4,128 games the gap is a function of the spread and the sigma, not a disagreement
    between a game's spread and its moneyline.
  - On the 815 games with real moneyline pairs (2022, 2024 and 2025), `|gap| > |beyond|` in
    `0.1829` of their top decile (82 games); on 2025 alone (272 games), in `0.3214` of its top
    decile (28 games). Both match the review's `0.183` and `0.321`. These groups decide
    nothing; their other numbers are in `BETTING_EDGES.md`.
  - Reproduce from the repository root (`--market-lines-metadata` is the metadata file above):
    `VIRTUAL_ENV="$PWD/.venv" .venv/bin/python .agents/sos_port/betting_edges.py
    --checkpoint-dir /home/mitch/workspace/nfl-predictor/models/wf_checkpoints/df20b775abf7f2657d13
    --market-lines-metadata
    /home/mitch/workspace/nfl-predictor/models/step4_reference/inputs/adopted_market_lines_metadata.json`.
- **How to reverse it**: the columns go with `_spread_home_prob` and the three assignments in
  `build_betting_report` (`src/nfl_predictor/reporting/betting_report.py`), the `floor_sigma`
  keyword in `pipeline._run_stage4`, and a snapshot rewrite (`NFLP_UPDATE_SNAPSHOTS=1`). A switch
  of the labels to `beyond` would map `edge_beyond_spread_home_prob` through
  `_edge_to_confidence_1_to_10` and `_edge_to_action`; it is the user's decision.
- **Label context from the review**: the reviewer's own scripts, copied unchanged into
  `.agents/sos_port/review/betting_edges/`, reproduce these numbers from the fold checkpoints
  (rerun from the copies on 2026-10-10, with the same results).
  - Basing the labels on the model's part alone would change 3,312 of the 4,943 action labels
    and 344 of the 495 in the top decile, and would flip the value side on 1,239 games.
    `recompute.py` prints them on its "labels (unrounded floor)" lines ("action differs on
    3312 of 4943", "top-decile games whose action changes: 344 of 495", "games where beyond
    points to the other side than edge: 1239").
  - At the floor's sigma, the normal curve gives favorites of 3 or more points a lower
    probability than the no-vig moneyline does: the gap is about `-0.045` at 7-7.5 points on
    the map-derived seasons and `-0.042` on the real ones (`tilt.py`, bucket `(6.5, 7.5]`:
    `-0.0451` and `-0.0418`). A model sitting exactly on a 7-point spread therefore shows an
    underdog edge of about `0.045`.
  - `groups_r2.py` recomputes every group of `BETTING_EDGES.md`, including the `0.1829` and
    `0.3214` above, and checks the season split against the metadata.
  - Commands, from the repository root:

    ```bash
    .venv/bin/python -I .agents/sos_port/review/betting_edges/recompute.py \
        /home/mitch/workspace/nfl-predictor/models/wf_checkpoints/df20b775abf7f2657d13
    .venv/bin/python -I .agents/sos_port/review/betting_edges/tilt.py \
        /home/mitch/workspace/nfl-predictor/models/wf_checkpoints/df20b775abf7f2657d13
    .venv/bin/python -I .agents/sos_port/review/betting_edges/groups_r2.py \
        /home/mitch/workspace/nfl-predictor/models/wf_checkpoints/df20b775abf7f2657d13 \
        /home/mitch/workspace/nfl-predictor/models/step4_reference/inputs/adopted_market_lines_metadata.json
    ```

- **Review**: Two independent rounds by a reviewer subagent that did not produce the work,
  2026-10-10. Round one: merge after fixes. Round two: merge after one doc fix (these script
  copies). Every number reproduced from the fold checkpoints with the reviewer's own scripts
  (commands above).
