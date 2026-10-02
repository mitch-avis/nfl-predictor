# Roadmap step 4, feature values: plan and ladder

Status: accepted by the user on 2026-10-01 (all three questions at the end answered yes). It is
the ladder under `AGENTS.md` rule 4: every rung up to the cap is approved, and anything beyond the
cap or outside this text is a fresh question. Phase A (code) is in progress on
`feat/step4-feature-values`.

## What is measured

| Arm | Change | How it differs from the reference | Task |
| --- | --- | --- | --- |
| R0 | none (the reference) | current code, a fresh build | - |
| QB | defense-adjusted quarterback rate | new columns, switchable group, same build as R0 | 53.7 |
| NO | drop the `*_next_opponent_abbr` identity pair | `--disable-feature-groups next_opponent_identity` on R0's build | 2026-09-25 review |
| S2 | season-to-date stat blend `K = 2` (today `4`) | its own build (`--stat-prior-blend-games 2`) | 55.3 |
| S8 | season-to-date stat blend `K = 8` | its own build | 55.3 |
| T2 | strength-snapshot blend `K = 2` (today the same constant `4`) | its own build, new ETL option | 55.3 with the 2026-09-25 review |
| PT | the pick-time market line (nfelo opener, fitted spread-to-moneyline map) | its own build | 56.6 |
| F | the adopted combination | its own build | all |

Why `T2`: today one constant, `PRIOR_BLEND_GAMES`, sets both the stat blend and the strength
snapshot's blend, but only the stat blend has a command-line option. The 2026-09-25 review found
the strength snapshot shrunk twice (ridge penalty and blend), so a smaller strength `K` is the
direction that review points to; `K = 2` is the one candidate tested.

## Phase A: code first, no data, no walk-forwards

Each chunk is an implementer subagent with an independent review, merged into
`feat/step4-feature-values` under its own version. All code lands before any run (rule 4: edits
under `ml/` change every checkpoint fingerprint).

1. 53.7, first step only: the defense-adjusted quarterback rate (career and last-8 windows, the
   `K = 300` shrinkage of `qb_dropback_epa`) as a new feature group. Proof that the group switched
   off gives the same feature matrix as today, so R0 is "QB off" on the same build. The ridge
   (its second step) runs only if QB shows signal, in a later ladder.
2. A `--strength-prior-blend-games` ETL option (default `4`, today's value), so `T2` needs no code
   change between builds.
3. A feature switch for the `*_next_opponent_abbr` pair, if `--disable-feature-groups` cannot
   already target it.
4. 56.6: the `nfelomarket_data` getter (cache, fallback, snapshot hash in the run metadata), team
   and game-id normalization, the per-game line order in the task text, the fitted
   spread-to-moneyline map, and an ETL option choosing the line source (`stored` today,
   `pick_time` the candidate), so both builds come from one code version. The largest chunk.

Any chunk that changes `NFLREADPY_SCHEDULE_COLUMNS`, `NFLREADPY_SCHEDULE_RENAME` or a derived
schedule column refetches every historical schedule file since `0.38.2`, which can pull revised
lines; such a change is asked about first (rule 5), and its scratch builds say so.

## Phase B: builds and runs

- Builds: six cold ETL builds (`R0`, `S2`, `S8`, `T2`, `PT`, then `F`), each about 11 minutes, made
  in scratch copies (`~/scratch/step4_builds/<arm>/`), never into `data/`. Each build's
  through-2025 cut is kept as the run input under `models/step4/inputs/<arm>.csv` with its hash,
  so every number stays auditable from disk. The leakage audit runs on each build.
- Runs: `nfl-predictor backtest` like the GPU reference: 2007-2025 from week 1
  (`--wf-start-week 1`, 328 folds, about 4,943 games), the floor with its pooled sigma, market
  anchor and transform on, 200 trees, unweighted, GPU, seeds `42` and `7`. About 32 minutes per
  run.
- Order: one driver script runs `R0`, `QB`, `NO`, `S2`, `S8`, `T2` and `PT` back to back (14 runs,
  about 7.5 hours, overnight), with a watcher that wakes the session when it exits. A check-in
  with each arm's governing numbers follows, after an independent reviewer's rescore (rule 3).
- Then the user picks what to adopt; `F` builds that combination and runs it on two seeds (2 runs).
  F is the new GPU reference and the floor's new sigma pool.
- Cap: 16 walk-forward runs (8 arms by 2 seeds) and 6 builds. Stopping rule: none early; every
  arm runs once on both seeds, and `F` runs only after the user's adoption decision.

Feature-group switches per arm. The QB columns (`qb_def_adj`) are in every build and are on
by default, so every arm except QB passes `--disable-feature-groups qb_def_adj` (NO passes
`qb_def_adj,next_opponent_identity`); QB is the only arm that trains on them. The QB columns
need a full-history build (the default `--min-season 1999`): a `--min-season` build counts the
defenses of earlier seasons as average.

Recorded before the runs: the QB adjustment is small next to the rate it adjusts (the adjusted
and raw career rates correlate about 0.9995 on the 2019-2026 scratch build, because
`adj_def_pass_epa_snap` is a shrunk per-snap coefficient applied per dropback). A QB tie
therefore says this attenuated form adds nothing, not that opponent adjustment of quarterbacks
has no signal; the next step after a tie is 53.7's joint ridge, not closing the task.

## Decision rules (written before any run)

For `QB`, `NO`, `S2`, `S8`, `T2`, each against `R0` on the same code: the candidate-minus-reference
loss per game, averaged over the two seeds (same seed paired), bootstrapped over games (rule 13).

- Governing window: all weeks, 2007-2025. Weeks 1, 2 and 3-18 are reported for every arm, with
  week-2 total MAE for the `K` arms (task 55.3), but they do not govern. Every interval that
  excludes zero, in any window and either direction, is listed (rule 9).
- Adopt: deterministic Brier improves (the interval excludes zero on the better side) and log loss
  does not worsen (its interval does not exclude zero on the worse side).
- Reject: deterministic Brier worsens (its interval excludes zero on the worse side).
- Tie (Brier's interval includes zero): goes to the user as a question with the tie-breakers
  (confidence-pool points, then margin MAE) and pick accuracy (reported, never selected on). The
  recommendation is the simpler setting: no new group for `QB`, the drop for `NO`, and for the
  `K` arms no recommendation from simplicity alone (no value is simpler than another).
- Several `K` winners: the one with the larger all-weeks Brier gain; within noise of each other
  (difference interval includes zero), a question.

For `PT` the question is different. Production can only use the pick-time line, so anchoring the
benchmark to the stored, near-closing line is the parity gap of `AGENTS.md` rule 11; `PT` is
expected to look worse on reported accuracy and is not an option to be beaten. The report gives:
`PT` against `R0` on every column and window as above; the market yardstick at the pick-time line
and at the close; the fallback-row count per season; and whether the moneyline features add
anything once the model is anchored to the spread (`PT` with and without the moneyline group, one
extra arm only if the user asks). The recommendation is to adopt `PT` for parity unless it breaks
something (a season or week window where it is far worse than its own market yardstick), and the
decision goes to the user (a default change).

## Questions for the user before it starts

1. Accept this ladder: 8 arms, 16 runs, 6 scratch builds, the rules above? (Recommendation: yes.)
2. The run inputs live under `models/step4/inputs/`, not `data/`, and production `data/` is
   untouched until the adoption decision; then one rebuild into `data/` (backed up first) with
   the adopted settings. (Recommendation: yes; it keeps `data/` out of the experiment.)
3. Out of scope here, for a second ladder after `F`: the remaining step-4 follow-ups that change
   values (the `adj_*` scale drift, a `sos_played_raw` fallback for weeks 1-2, the quarterback
   identity-chain items, the ridge form of 53.7) and the column cleanups (`strength_games_played_diff`,
   the unpublished `PBP_COUNT_COLUMNS`). (Recommendation: yes; folding them in would double the
   ladder before we know which direction pays.)
