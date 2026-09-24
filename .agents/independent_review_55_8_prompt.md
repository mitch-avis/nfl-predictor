# Independent review session: task 55.8 season-weighting arms

You are an independent reviewer for the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). Read `AGENTS.md` first, in particular the delegation
guardrails (rules 1-14; rules 3, 9, 12 and 13 govern this review). Then read the task 55.8 entry in
`.agents/TODO.md`. You did not produce any of the runs below; the session that launched five of
them also wrote a review, and your job is to be the second key that rule 3 requires.

## Scope and limits

- Read-only on everything except the one file you write:
  `models/wf_m55_8_review/INDEPENDENT_REVIEW.md` (plus your own script beside it, e.g.
  `models/wf_m55_8_review/independent_rescore.py`, and its output).
- Do **not** edit `AGENTS.md`, `.agents/`, `CHANGELOG.md`, `README.md`, `config/`, or any code.
  Do not commit, merge, or push. Do not start any walk-forward or ETL run.
- Do **not** open `models/wf_m55_8_review/REVIEW.md`, `review_55_8.py`, `review_55_8.json`,
  `review_55_8_output.txt`, `probability_paths.py` or `probability_paths_output.txt` until your
  own rescore is finished and written down. Then compare and list every disagreement.
- Write your own rescoring code. Do not reuse the producer's scripts or
  `models/wf_m55_7_2020_2025_trees200/compare_to_benchmark.py`.

## The nine arms

All: input `data/completed_games_ml.m54_flip_through_2025.csv`, seasons 2020-2025 from week 1,
`--calibration auto`, `--wf-calibration-weeks 4`, `--market-anchor --market-transform`, 200 trees.
Each run directory has `metadata.json` (its `config.checkpoint.dir` names the fold checkpoints),
`metrics_report.json`, `run.log` and `launch.sh`.

| arm | run directory | seed | `--recency-half-life-seasons` |
| --- | --- | --- | --- |
| unw_s42 | `models/wf_m55_8_2020_2025_unweighted/` | 42 | none |
| hl4_s42 | `models/wf_m55_8_2020_2025_half_life4/` | 42 | 4 |
| hl8_s42 | `models/wf_m55_8_2020_2025_half_life8/` | 42 | 8 |
| hl16_s42 | `models/wf_m55_8_2020_2025_half_life16/` | 42 | 16 |
| hl32_s42 | `models/wf_m55_8_2020_2025_half_life32/` | 42 | 32 |
| unw_s7 | `models/wf_m55_8_2020_2025_unweighted_seed7/` | 7 | none |
| hl4_s7 | `models/wf_m55_8_2020_2025_half_life4_seed7/` | 7 | 4 |
| hl16_s7 | `models/wf_m55_8_2020_2025_half_life16_seed7/` | 7 | 16 |
| hl32_s7 | `models/wf_m55_8_2020_2025_half_life32_seed7/` | 7 | 32 |

The decision rules were written before the runs, in:

- `models/wf_m55_8_2020_2025_unweighted/HYPOTHESIS.md` (the seed-42 ladder),
- `models/wf_m55_8_2020_2025_unweighted_seed7/HYPOTHESIS.md` (half-life 4 vs unweighted, two
  seeds),
- `models/wf_m55_8_2020_2025_half_life16_seed7/HYPOTHESIS.md` (half-lives 16 and 32 vs
  unweighted, two seeds).

Apply them exactly as written (rule 12); if a result falls between branches, say so.

## What to do

1. **Provenance, per arm.** Dataset hash (compare with `sha256sum` of the input file), git commit,
   seed, half-life, folds computed/restored, and that every fold's checkpoint (`fold_*.joblib`:
   keys `predictions` and `metrics`) shows `calibration_method` `none`, `best_iteration` 199 and
   `early_stopped` false for the margin and total heads. Confirm that the `game_id` sets are
   identical across arms and that only the seed and half-life differ in the configs. Confirm that
   nothing under `nfl_predictor/` changed between the commits the arms record
   (`git diff --stat <a>..<b> -- nfl_predictor`).
2. **Metrics from the checkpoints**, per arm, for four windows (week 1, week 2, weeks 3-18, all
   weeks), from the concatenated `predictions` frames:
   - deterministic Brier and log loss on `deterministic_home_win_prob` against `actual_home_win`;
   - pick accuracy, with ties incorrect for both sides;
   - margin and total MAE on `predicted_margin` / `predicted_total`;
   - confidence-pool points: rank `|p - 0.5|` within each week, 1..N;
   - market Brier on `market_home_win_prob`.
   Cross-check deterministic Brier and margin MAE against each `metrics_report.json`.
3. **Paired comparisons**, with a stated bootstrap method (resampling games; for pool points,
   resampling weeks):
   - every arm against the same-seed unweighted arm;
   - each setting against itself across seeds (the seed-noise floor);
   - the two-seed contrasts each rule names (half-life 4, 16 and 32, each minus unweighted,
     combined per game over the two seeds).
4. **Rule 9 read-out:** every interval that excludes zero, in every window, in either direction.
5. **The decision** under each written rule, and whether you agree with the recommendation that
   production should be unweighted.
6. Only then, compare your numbers and conclusions with `models/wf_m55_8_review/REVIEW.md` and
   list every difference, including wording that overstates or understates the evidence.

## Output

Write `models/wf_m55_8_review/INDEPENDENT_REVIEW.md`:

- who you are and a statement that you produced none of the runs;
- the reproduction command for your script;
- the tables above;
- the rule outcomes;
- the disagreement list.

Then report to the user in chat: the rule outcomes, whether you confirm the producer's numbers
(to what precision), every disagreement, and anything you think the user should know before
deciding the production season weighting. Stop there: the decision, the docs and the config
change belong to the user and the next implementation session.
