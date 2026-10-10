# Decisions in building and scoring the candidate power rankings (roadmap step 6, phase 5)

Every judgment call made while turning the pre-registered design (the main checkout's
`models/step6/rankings/HYPOTHESIS.md`, SHA-256 `44e3a33c...4f40`, unchanged) into code: the four
candidate ratings (`src/nfl_predictor/reporting/rating_sources.py` and
`power_rankings.composite_points_vs_average`), their score
(`src/nfl_predictor/reporting/ranking_evaluation.py`, run by `.agents/sos_port/rankings_eval.py`)
and the ranking methods behind a setting (`nfl-predictor rankings --method`, the weekly run's
`power_rankings_method`). Each was decided before any candidate was scored. The default ranking
stays the composite; switching it is the user's decision after the measurement.

No result of the measurement is quoted here: the numbers live in the generated
`RANKINGS_EVAL.md` and wait for an independent review (`AGENTS.md` rule 3).

## 1. The model rating's ridge penalty

- **What differs**: the hypothesis defines candidate 3 as "a ridge fit of team strengths (plus a
  home term) to the trained model's pre-game predicted margins for the season's games before the
  week" and names no penalty.
- **Options**: (a) a near-unpenalized fit, `1e-6` on the team strengths (the value the ported
  prior uses for its own near-unpenalized fits), the home term unpenalized; (b) a penalty
  cross-validated on the previous season's predictions, as the ported team rating takes its
  penalties; (c) `1.0`, the default of this module's Bradley-Terry fit; (d) an empirical-Bayes
  penalty from the residual and strength variances of the previous season.
- **Decision**: (a), `MODEL_RATING_RIDGE_LAMBDA = 1e-6`.
- **Reasoning**: the targets are already the model's regressed predictions, so a real penalty
  would shrink them a second time. The penalty's only job is to pin a solution where the games so
  far do not identify every strength: in week 2 each team has one game, and the smallest
  strengths that fit split each game's predicted margin, less the home term, evenly between the
  two teams, which is also the posterior mean of two equal-variance zero-mean strengths given
  their difference. (b) cross-validates on a full season, which says nothing about week 2's
  under-identified fit, and needs an earlier season the first walk-forward season (2007) does not
  have; (c) and (d) add a constant the design does not name. (a) also lets production compute a
  week's rating from the current season's predictions alone.
- **Data**: `tests/test_rating_sources.py` pins the fit: additive margins are recovered exactly
  (`test_fit_model_ratings_additive_margins_recovers_the_strengths`), a neutral-site game carries
  no home term, a team without a game rates zero, and one game a team splits each margin evenly
  around the home term
  (`test_fit_model_ratings_one_game_each_splits_the_margin_around_the_home_term`).
- **How to reverse it**: `MODEL_RATING_RIDGE_LAMBDA` in `rating_sources.py` (it changes every
  model rating and the blend, so the measurement would be rerun).

## 2. The model rating's week 1

- **What differs**: the hypothesis says week 1 "uses the previous season's last such fit,
  regressed by the same factor the ported prior uses". The ported prior
  (`src/nfl_predictor/ratings/team_prior.py`) carries last season's per-play offense and defense
  effects into the new season with a slope each, pooled over the earlier season pairs, and carries
  special teams with none. The model rating is one number per team, so "the same factor" needs one
  value.
- **Options**: (a) the factor the ported prior applies to a whole team rating that season: the
  least-squares slope, through the origin over the teams of both seasons, of each team's week-1
  ported rating (the prior at full weight) on its published rating for the whole previous season;
  (b) the mean of the prior's offense and defense slopes; (c) the same estimator applied to the
  model rating's own history (each season's last fit regressed on the season before's); (d) this
  repository's `WEEK1_REGRESSION_FACTOR`.
- **Decision**: (a), `rating_sources.carryover_factor`. "The previous season's last such fit" is
  the fit on every predicted regular-season game of the previous season. Week 1 then rates each
  team at that fit times the factor, centered over the season's teams (a team new to the league
  starts at zero).
- **Reasoning**: the sentence names the ported prior's factor, not its method, which rules out
  (c), and (d) is another family's constant. Of the readings of the ported prior's factor, (a) is
  exactly what that prior does to a whole team rating, special teams' zero carryover included,
  and it is computable from a build's published files (the game rows' week-1 values and
  `ratings/{season-1}_ratings.parquet`), so production can compute it on the morning of week 1.
  (b) needs the two slopes, which no output records, and weights offense and defense equally
  whatever their spreads.
- **Data**: `power_rankings_carryover.py` beside this file prints the factor for 2008-2026 beside
  the prior's two slopes (`power_rankings_carryover_output.txt`, from the candidate build and the
  nflreadpy cache snapshot). The factor runs from 0.557 to 0.718 (mean 0.621) and lies between the
  prior's defense slope (0.39-0.49) and offense slope (0.69-0.81) in every season, as a summary of
  the two should. Week 1 is not scored (the score starts in week 2), so this reading changes only
  the ranking shown before a season's first game, never the measurement.
- **How to reverse it**: `carryover_factor` and `week_one_model_ratings` in `rating_sources.py`.

## 3. A team without a game before a week from week 2 on, in the model rating

- **What differs**: a ridge fit to the season's games before the week has no data on a team that
  has not played yet; the ported rating gives such a team its preseason prior.
- **Options**: (a) the ridge's own answer, zero (an average team); (b) the week-1 rating of
  decision 2.
- **Decision**: (a).
- **Reasoning**: it is the literal reading ("a ridge fit ... to the season's games before the
  week"; week 1 alone is said to use the previous season), and it keeps the scored weeks free of
  the week-1 factor's inputs.
- **Data**: in the build's 2007-2025 regular seasons only two teams first play after week 1,
  Miami and Tampa Bay in 2017 (their week-1 game moved to week 11 for Hurricane Irma), so two
  scored games of 4,400 (2017 week 2, Miami at the Chargers and Chicago at Tampa Bay) read a zero
  model rating for one side. Query: the first week of each team's regular-season games in
  `completed_games_ml.csv`.
- **How to reverse it**: `model_ratings_by_week` in `rating_sources.py`.

## 4. Where the ported rating is read from

- **What differs**: the hypothesis names "`team_rating` from the ratings package's pre-week entry
  point". A build publishes it twice: the game rows' `ratings` feature group (the entry point's own
  values, for the two teams of each game) and the published weekly history
  `ratings/{season}_ratings_by_week.parquet`, whose week `w` rates the teams that have played on
  the games through `w`.
- **Options**: (a) the game rows, with the weekly history of the week before filling the teams
  on a bye; (b) the weekly history alone; (c) refitting with `RatingsHistory.pre_week` from the
  build's play-by-play cache.
- **Decision**: (a), checked against (b) on every team-week both rate.
- **Reasoning**: the game rows are exactly the entry point's values, including week 1's preseason
  prior and a team yet to play, which the history lacks; the history adds the teams on a bye,
  which the rank agreement needs (byes are not scored). (c) repeats the ETL's work and its
  minutes.
- **Data**: on the candidate build `~/scratch/step6/data`, 9,278 team-weeks are rated by both
  sources and none differs by more than `1e-9`; with the history rebuilt from the nflreadpy cache
  snapshot instead (`rankings_eval.py --ratings-from-snapshot`) the same 9,278 agree and every
  score is unchanged. The ratings parity audit had already found the history bit-identical to the
  entry point for the teams that have played (`.agents/sos_port/RATINGS_PARITY.md`).
- **How to reverse it**: `combine_ported_ratings` and `load_ported` in `rankings_eval.py`.

## 5. The composite on every digit

- **What differs**: `rank_teams_on_composite` publishes the composite's points rounded to two
  decimals.
- **Options**: (a) score the unrounded points; (b) score the published rounded values.
- **Decision**: (a): `composite_points_vs_average` returns them unrounded and the ranking rounds
  that for display, so the published table is unchanged.
- **Reasoning**: the rounding is for display; the formula the hypothesis names is the same.
- **Data**: the existing composite tests pass unedited after the extraction, and one of them fails
  when the extracted function is broken on purpose (commit `c0e7913`).
- **How to reverse it**: score `rank_teams_on_composite`'s column instead.

## 6. The league home term

- **What differs**: "the league home term fitted on earlier seasons' scored games only" names no
  loss and no window.
- **Options**: (a) least squares (the mean residual of the earlier non-neutral games), every
  earlier season from 2007; (b) least absolute deviation (their median); (c) a rolling window.
- **Decision**: (a), one home term per candidate and season.
- **Reasoning**: the `nfl-sos-ratings` validation fits its margin model by least squares
  (`git show 9081bce:nfl_sos_ratings/validation/walk_forward.py`, `_fit_margin_projection`), the
  mean is the usual home-field estimate, and (b) would fit the score's own loss. The hypothesis
  says 2007 warms the term up, which an expanding window from 2007 does; (c) needs a length the
  design does not give.
- **Data**: `tests/test_ranking_evaluation.py`
  (`test_fit_home_term_is_the_mean_residual_of_non_neutral_games`,
  `test_home_terms_by_season_reads_only_earlier_seasons`).
- **How to reverse it**: `fit_home_term` in `ranking_evaluation.py`.

## 7. The blend's weight and home term

- **What differs**: the weight is "chosen each season on the earlier seasons' scored weeks only";
  how it is scored there, how a tie breaks and which home term the blend then uses are not said.
- **Options**: for each weight on the grid, (a) fit its own home term on the earlier seasons and
  measure its error there with it, the blend then using the chosen weight's home term; (b) score
  each earlier season with that season's walk-forward home term, which leaves 2008 (chosen on 2007
  alone) without one.
- **Decision**: (a); the lowest error wins, errors within `1e-12` tie and a tie goes to the larger
  weight, the one nearer the ported rating alone.
- **Reasoning**: (a) is defined from 2008 on and treats every weight alike; the tie rule follows
  the hypothesis's "a tie goes to the reference", and the tolerance keeps floating-point noise from
  breaking a true tie (the same ratings blended at two weights can differ in the last bit).
- **Data**: `tests/test_ranking_evaluation.py` (`test_choose_blend_weights_*`,
  `test_blend_differences_apply_each_season_its_own_weight`).
- **How to reverse it**: `choose_blend_weights` in `ranking_evaluation.py`.

## 8. The two bootstraps

- **What differs**: the hypothesis fixes 10,000 resamples, seed 42, 95%, over games, with a
  season-cluster bootstrap; the drawing scheme and the season statistic are not said.
- **Options**: over games, (a) `run_comparison.bootstrap_mean`, the instrument of
  `nfl-predictor compare`, or (b) a new one. Over seasons, (c) the mean over every game of the
  drawn seasons, or (d) the mean of the drawn seasons' means.
- **Decision**: (a) and (c), each drawn in one `numpy.random.default_rng(42).integers` call, with
  2.5 and 97.5 percentiles.
- **Reasoning**: (a) is the repository's paired-interval instrument; (c) resamples the same
  statistic as the point estimate, every game weighing the same.
- **Data**: `test_paired_comparison_scores_only_the_window` checks the game interval equals
  `bootstrap_mean`'s. `test_season_bootstrap_pools_the_games_of_the_drawn_seasons` checks (c)
  against (d) on five seasons of unequal size (one of a single game at +10, four of nine games at
  0): pooling puts the upper bound at 30/21, averaging the drawn seasons' means at 6, and the test
  fails with (d) in place (the review found the earlier two-season test could not tell them
  apart).
- **How to reverse it**: `paired_comparison` and `season_bootstrap` in `ranking_evaluation.py`.

## 9. Games a candidate cannot rate, and the rank agreement

- **What differs**: the paired comparison needs one set of games; the hypothesis reports rank
  correlations and top-five differences without defining them.
- **Decision**: every candidate is scored on the games all of them rate, and if any scored game
  lacks a rating the decision is not applied and goes to the user with the numbers (`AGENTS.md`
  rule 12). The rank agreement is per scored week, over the teams both rate: the Spearman
  correlation (average ranks for ties) and whether the two top-five sets differ (ties broken by
  team code), summarized by window.
- **Reasoning**: a candidate missing games would be compared on other games than the reference.
  Sets rather than orders keep "how often the top 5 differ" about who is in it.
- **Data**: on the candidate build the ported rating and the composite rate all 4,400 scored
  games.
- **How to reverse it**: `score_candidates`, `decision` and `agreement_results` in
  `rankings_eval.py`; `rank_agreement` in `ranking_evaluation.py`.

## 10. The ranking methods in production

- **What differs**: the rankings command and the weekly run ranked on the composite or a
  Bradley-Terry fit only.
- **Options**: (a) three methods behind the existing setting (`ported`, `model`, `blend`), the
  default unchanged; (b) the ported rating only, the others left to a later step.
- **Decision**: (a). Each ranks through the composite's mapping (points from the rated teams'
  mean, then the model's margin curve, then the 1-10 and 0-10 scales) and publishes
  `team_rating`, `points_vs_average`, `rating_week` and `rating_source`. `ported` reads the game
  rows first and the weekly history for byes (its last week for a postseason week). `model` and
  `blend` read walk-forward fold checkpoints (`--model-predictions`), which must cover every
  completed regular-season game the fit reads, checked by `game_id` against the schedule (after
  the review: a count let predictions of other games in equal number pass); `blend` needs
  `--blend-weight`. An unavailable source skips the weekly rankings with a warning, as a missing
  snapshot does, and stops the rankings command with a one-line error (a missing snapshot week
  now does the same instead of a traceback); since the review, predictions that cannot be read
  (a missing or foreign directory, an empty checkpoint directory, a game in two sources, a fold
  file without predictions) and a week-1 carryover factor that cannot be fit count as an
  unavailable source too, named in the message. The weekly run's reports-stage key gains the new
  settings only when one of the three methods ranks, so the default's key is unchanged.
- **Reasoning**: one implementation serves the measurement and production, so whichever candidate
  the rule picks can be switched on by the setting alone (`AGENTS.md` rule 11).
- **Data**: `tests/test_power_rankings_sources.py`, `tests/test_cli_rankings_sources.py`,
  `tests/test_weekly_run_rating_sources.py`; the CLI surface snapshot records the new flags.
- **How to reverse it**: `POINTS_METHODS` and `load_team_ratings` in `power_rankings.py`.
- **Left open**: where a weekly run would take the model rating's predictions if the model or
  blend were chosen (its own stage-1 folds for the season so far, and the floor-sigma reference
  runs for the previous season, are the natural sources); the setting takes explicit paths today.

## 11. The second seed

- **What differs**: the hypothesis reads the model rating from the anchored ported-ratings run on
  seed 42; a result that would change a default holds on two seeds (`AGENTS.md` rule 13).
- **Options**: (a) fit the model rating to each game's predicted margin averaged over the seed-42
  and seed-7 runs; (b) fit each seed's rating and average the ratings; (c) score each seed and
  average the per-game errors, as `nfl-predictor compare` pairs seeds.
- **Decision**: (a), given by the delegating session after the review: `rankings_eval.py` takes
  `--run-dir` once per run, seed 42 first, and reports the two-seed result as a supplement beside
  the seed-42 result, which alone stays the rule's input.
- **Reasoning**: the ridge is linear in its targets, so (a) and (b) give the same ratings for a
  week (the averaged targets fit to the averaged strengths), and (a) keeps one fit per week; (c)
  would average two blends with two sets of chosen weights, which is not one ranking anyone could
  publish.
- **Data**: `rating_sources.average_predictions` and its tests (`test_average_predictions_*`):
  margins summed in the given order, runs of different games or of a game with other teams
  refused.
- **How to reverse it**: give `--run-dir` once.

## Review

An independent review found the measurement faithful to the design and its numbers reproducible,
and asked for the fixes now recorded in decisions 8 and 10, the week-1 fit's regular-season filter
(decision 2) and the second seed (decision 11).
