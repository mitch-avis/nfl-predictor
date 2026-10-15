# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9), `.agents/benchmarks.md` before any walk-forward,
and `models/qb_elos_measure/HYPOTHESIS.md` (main checkout; `models/` is not in git) before reading
the measurement. `.agents/step6_plan.md` and `.agents/sos_port/DECISIONS.md` hold step 6's history.

## State (written 2026-10-15, 06:05 MDT; the user asleep until about 09:30)

- `main` is at `0.59.0` plus docs commits (not pushed): the step-6 port, adopted in production.
  `data/` was rebuilt on 2026-10-14 (record in `.agents/benchmarks.md`, "Data state and ETL
  rebuild records"; backups in `~/scratch/data_backups/2026-10-14_pre_step6_merge/` and
  `2026-10-14_pre_week6/`). The Week 6 run is `models/weekly_20261015_022416_9a4c092c`.
- The user submits Week 6 picks before SEA@DEN (2026-10-15 20:15 ET). Production-changing merges
  land after the Monday game (2026-10-19), with the user's approval (rule 14). Nothing is merged
  tonight.
- The user's web server runs on `:8001` from the main checkout (no `--reload`, the `0.45.4` code
  it loaded, the old `web/dist/`). Never stop or restart it, and don't build `web/dist/` on `main`
  while it runs; `scripts/gate.sh --web` and a browser look at the new pages wait until the user
  restarts it.
- `../nfeloqb` fork `main` is at `9bf32f2` (pushed): next-week projections now find a team stored
  under a legacy code (`OAK` for Las Vegas), which had inflated Las Vegas's `qb_adj` to `+160`.
  nfeloqb was not rerun; the user runs it each week (memory `nfeloqb-weekly-process`).
- **`chore/pyproject-standards` (`0.59.1`, `1bc21d3`, the main checkout, off `main`, not merged)**
  holds the user's `pyproject.toml` review: strict pytest with warnings as errors, coverage of
  `src/` and `tests/` plus the gate's `src coverage` step, PEP 639 metadata, ruff/ty/pyright on
  the configured roots, `update_requirements.sh` rewritten with `update_requirements.hooks.sh`,
  and `weekly_run/config.py` at 100% coverage (contract tests, loader errors as `ValueError`
  naming the file, relative single paths from the root; reviewed independently, findings fixed).
  The gate fails only on pyright: the user switched it to strict, about 3,150 errors in `src/`
  and `tests/`, a separate cleanup the user will start. `ml/feature_spec.py` changed (the `str`
  dtype), so the branch invalidates every checkpoint: merge it with other such changes (rule 14).
  The shared files also live in the new Copier template `~/workspace/python-template` (its own
  git repo, gate green), whose `update_requirements.sh` and update tests are identical to this
  repo's; adopting the template here (`copier copy`, its README's "Existing project") is open.
- **The `qb_adj` review is done.** `feat/qb-adj-features` (`0.60.0`, `c637fc0`, harness worktree
  `.claude/worktrees/agent-afad3cf465d5bcba7`) passed an independent review against
  `.agents/qb_adj_task.md` with minor findings, all docs and tests. They are fixed on
  `fix/qb-adj-review-findings` at `6bacd5e` (nine commits on `c637fc0`, worktree
  `~/scratch/qb_adj_docs`) and went through a re-check reviewer, whose one question (is the
  rule-11 approval on record?) is answered under "Decisions on record":
  - the reproduction commands in `.agents/guardrails.md` and `.agents/walk_forward_runbook.md`
    add `qb_adj` to the group lists recorded before `0.60.0`;
  - AAA labels; a season-build test carrying `qb_adj` (`tests/` fixture regenerated);
  - the changelog's order and its reproduction note; the rule-11 paragraph in
    `.agents/benchmarks.md` ("The `qb_adj` columns and these references").
  `src/`, `pyproject.toml` and `uv.lock` are identical to `c637fc0`, so the measurement below
  measures the reviewed code. `scripts/gate.sh` on `6bacd5e` (detached in `~/scratch/qb_adj_review`,
  its own venv, `-n 8`): exit `0`, 3,627 tests, 95.01% coverage
  (`~/scratch/gate_qb_adj_6bacd5e.log`). `feat/qb-adj-features` stays at `c637fc0`: moving it to
  `6bacd5e` is a fast-forward, which needs `git merge` and so waits for the user's approval.
- The review's adversarial tests stay untracked in
  `~/scratch/qb_adj_review/tests/test_qb_adj_review_adversarial.py`: 16 pass, 3 fail.
  - Two fail in `fill_future_qb_data` (the inherited `game_utils._fill_side_qb` behaviour): a
    played row with a named starter and a null adjustment gets the latest starter's value, and
    `qb_adj_diff` does not follow the filled sides. Fixing it changes upcoming-week feature values
    (rule 5), so it is on the user's list.
  - One: `week_rows._merge_elo` leaves `_right` duplicates when the frame already carries the
    columns. Its only call site (`utils/polars/week_rows.py:552`) never passes them, and the Elo
    columns behave the same way: no action proposed.
- **The grouped measurement is finished and reviewed** (`models/qb_elos_measure/`: `build.sh`,
  `check_cuts.py`, `driver.sh`, `launch.sh`, `build_record.txt`, `launch_record.txt`,
  `REVIEW.md`). Worktree `~/scratch/qb_measure` is detached at `c637fc0` with its own venv and
  `data/`; the main checkout's `data/` was only read.
  - Builds: new cut `b9772e3e...`, old cut `b59366e2...`, 7,261 rows and 544 columns each,
    leakage audits ok. `inputs/cut_diff.json` is ok: 12 `elo` columns differ, only in 2023-2025
    rows (from 2023-09-07); the `qb_adj` columns are identical in the two builds. The hypothesis
    stands unamended (sha256 `d682a0c0...`).
  - Runs: all eight exited `0` between 01:37 and 05:47 (`driver.log`, 1,845-1,942 s each), in
    `models/qb_elos_measure/<arm>_seed<seed>/`. The producer's two-seed `compare` outputs (not
    yet reviewed numbers) are in `checkins/` (`new_file_vs_old_file`, `qb_adj_vs_new_file`,
    `strength_vs_new_file`, `_two_seeds.{md,json}`).
  - `REVIEW.md` (independent reviewer, 2026-10-15 05:58): provenance OK (each candidate differs
    from its reference only in the input or the groups; old-file runs reproduce
    `models/step6/ratings_anchored_seed{42,7}` bit for bit), all three rules apply cleanly, two
    seeds, 10,000 resamples:
    - the new file against the old (seasons 2023-2025, 816 games): Brier `-0.00026`
      `[-0.00082, +0.00030]`, log loss `-0.00070` `[-0.00195, +0.00055]`: no loss, the new file
      stays (nothing changes);
    - `qb_adj` against the new file (all weeks): Brier `+0.00025` `[-0.00026, +0.00076]`, log
      loss `+0.00056` `[-0.00056, +0.00170]`: a tie, `qb_adj` stays off; the seeds disagree, and
      weeks 3-18 pool points lose (`-82.0` `[-156.5, -8.0]`);
    - the strength snapshot against the ported ratings (all weeks): Brier `+0.00102`
      `[+0.00045, +0.00159]`, log loss `+0.00238` `[+0.00111, +0.00366]`: the snapshot loses, the
      ported ratings stay (also losing pool points and margin MAE, both seeds);
    - no arm beats the pick-time line anywhere; week 1 and week 2 show nothing to flag.
    `compare` and the producer's check-ins agree with the rescore; `compare` cannot produce the
    2023-2025 window, which comes only from `REVIEW.md`.
  - The transition watcher grepped a relative `driver.log` from the wrong directory, so it never
    matched; the check-in the user asked for at the first run-to-run transition (02:09) did not
    happen. Watchers poll the PID and use absolute paths.
- **Step 3, the "no game lines" groups change, is reviewed and approved** (`0.61.0`, branch
  `feat/no-game-lines-groups` at `5c33978`, worktree `~/scratch/step3`, twelve commits on
  `6bacd5e`). `constants.NO_GAME_LINES_DISABLED_FEATURE_GROUPS` =
  `strength,elo,own_elo_wt,qb_adj`; one resolution in `cli/options.py` serves `backtest`, `train`
  and the weekly run: an explicit list (command line or config key, `''`/`[]` included) wins,
  otherwise a run without game lines (weekly `--wf-market-mode none`; `backtest`/`train` with
  market features, anchor and transform all off) gets that list and every other run production's.
  `config/weekly_run.yaml` no longer sets `disable_feature_groups`. Production is pinned
  unchanged (groups, stage-1 options, stage hash `ef442e1e`, run fingerprint `d9189345...`); the
  `constants.py` edit moves every walk-forward checkpoint fingerprint, so this merges only after
  the measurement finishes. The independent review approved with findings (its gate: exit `0`,
  3,675 tests, 95.01%; adversarial tests uncommitted in
  `~/scratch/step3_review/tests/test_step3_review_adversarial.py`, 13 pass and 2 fail, both the
  user's questions below); the wording and test-layout fixes (`022e555..5c33978`) passed a
  re-check reviewer and the gate on `5c33978` (exit `0`, 3,675 tests, 95.01%,
  `~/scratch/gate_step3_5c33978.log`, and the re-checker's own run).

## Next, in order

1. **The user's decisions on the measurement** (below): the floor-sigma reference, then a
   `.agents/benchmarks.md` record of the three outcomes from `REVIEW.md` (restating reviewed
   numbers), and `TODO.md` follow-ups that concern only the strength snapshot (closure is the
   user's call).
2. **Step 3** waits for the user's answers below and the merge after the Monday game. The
   older roadmap labels the re-check found outside the branch (`README.md:897` "task 55.8",
   `README.md:1067` and `config/weekly_run.yaml:144` "decision 12") go with the step-6 docs
   close-out.
3. **`../nfeloqb`, a backup's first start** (approved 2026-10-15, test first): the next-week
   projection rates the starter with `current_value`, while the played row uses
   `get_value(team)`, which applies the backup rule (Baltimore's Week 5 projection was off by
   `0.8`). Read `get_value` first and confirm it uses only pre-game information; then a failing
   test in `tests/test_elo_qb_adjustment.py`, the fix in `elo_file_constructor.py`, the fork's
   gates (`.venv/bin/ruff format .`, `ruff check .`, `ty check .`, `pyright .`,
   `markdownlint-cli2 "**/*.md" "#.venv"`, `.venv/bin/python -m pytest`), a `fix:` commit, and a
   push to the fork (`GUARD_OK=1`). Before the user's next nfeloqb run (Week 7).
4. **The fork's win-total file** (approved 2026-10-15 as its own measurement after the grouped
   one, landing between game weeks): `nfeloqb/Manual Data/wt_ratings.csv` ends at 2023, so the
   QB-adjusted Elo has had no win-total pull since 2024 (`elo.py` line ~136). Hypothesis and
   rule first, a fork change, a third build against the grouped measurement's new-file build.
5. **The step-6 docs close-out and the roadmap tidy** (approved 2026-10-15):
   - regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (a `models/` write: ask);
   - rename "market-free" to "no game lines" where it means the configuration with Elo,
     including `weekly_run/stage1.py:63,68`;
   - refresh the stale "Review:" fields in `.agents/sos_port/decisions/*.md`; commit the
     hypothesis checksums;
   - record the 2026-10-11, 2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and
     `TODO.md`; they and the open P3s from the integration reviews are in an earlier handoff,
     `git show 5b73686:.agents/next_agent_session_prompt.md` ("Decisions on record", "Open for
     the user" and the integration-progress bullets);
   - in `TODO.md`: refresh "Current validated baseline" (it still says `0.28.1`); mark step 6's
     phases done; retire the follow-ups that concern only the strength snapshot if the grouped
     measurement keeps the ported ratings (rule 5: the user approves each closure);
   - build the approvals list and the user's final HTML report for step 6.
6. `scripts/gate.sh --web` on `main` and a browser look once the user restarts the server; the
   2026 roof recheck before the Paris game on 2026-10-25
   (`.agents/sos_port/decisions/pbp_columns.md`).

## Decisions on record (2026-10-15)

- `../nfeloqb` Las Vegas fix: approved and done (`9bf32f2`).
- Adopt `qb1_adj`/`qb2_adj` as features, off in production until measured: approved.
- The grouped measurement: approved, with its ladder (two builds, eight runs).
- The backup first-start fork fix: approved, test first, after the picks.
- The stale win-total file: its own measurement after the grouped one, between game weeks.
- The roadmap: steps 7 (the Elo inputs) and 8 (the feature ideas, explored quickly, then
  accepted or dropped) go before the re-tune, now step 9 (`TODO.md`, "Roadmap Status").
- Week 6: the top-decile ranking is the two-seed mean (2026-10-14, in DECISIONS.md).
- Overnight session rules (2026-10-15, 01:20 MDT, the user asleep for about eight hours):
  - a review finding that needs the user holds the measurement only if its answer could change
    the `qb_adj` column values, the default feature groups or the backtest command; anything
    else goes on the morning question list and the runs proceed;
  - if the build check finds old-file/new-file differences beyond the 2023-2025 rows of the
    `elo` and `qb_adj` columns, an amendment to `HYPOTHESIS.md` (new checksum) proceeds when
    every extra difference traces to `qb_elos.csv`; anything unrelated (pre-2023 rows, market
    lines, non-Elo stats) holds;
  - step 3 (the "no game lines" groups change) may start during the runs, stacked on
    `feat/qb-adj-features`, pytest capped at `-n 8`, nothing merged;
  - the production-settings difference with the floor-sigma reference runs is recorded as
    harmless in `.agents/benchmarks.md` until the reference is replaced (rule 11 approved).

## Open for the user

- `chore/pyproject-standards`: approve the merge (with the checkpoint-invalidating group), and
  whether to adopt the python-template here now. Two config questions from its review:
  - path lists in a weekly config file (`floor_sigma_reference_runs`,
    `power_rankings_model_predictions`) stay relative to the working directory in
    `reporting/rating_sources.py:430` and `run_comparison.py:136`, against
    `config/weekly_run.yaml:37` ("relative to the repository root"); the blend ranking fails when
    `nfl-predictor weekly` starts outside the root. Recommended: resolve them from the root at
    read time; it changes recorded config strings and reruns the reports stage once;
  - config files accept feature-group lists only, not comma strings. Recommended: keep it;
  - `~` in a config path is not expanded (`~/x` becomes `<root>/~/x`).
- Approve fast-forwarding `feat/qb-adj-features` to `6bacd5e`, together with its merge after the
  Monday game.
- Step 3's list includes `qb_adj` (`strength,elo,own_elo_wt,qb_adj`), where the 2026-10-14
  approval named `strength,elo,own_elo_wt`. Recommended: keep it, since `qb_adj` is off in
  production until measured and the `--wf-market-mode` help already advises this list; dropping
  it is a one-line change. If the measurement turns `qb_adj` on, the list follows.
- Step 3, `train`: its anchor and transform were already off by default, so
  `train --exclude-market` alone now counts as a run without game lines and leaves out `elo`
  (before: production's list). The implementer added `--no-market-anchor` and
  `--no-market-transform` to `train` (`BooleanOptionalAction`, still off by default) so the
  three-flag command matches `backtest`'s. Both reviewers recommend accepting it; nothing in
  `src/`, `scripts/` or the web API calls `train` with these flags.
- Step 3, `backtest --exclude-market --no-market-anchor` without `--no-market-transform` keeps
  production's list (the contract's "all three off", taken literally), although with no market
  columns the transform changes nothing. The reviewer recommends dropping the transform from the
  test (`cli/options.py:115`, one line plus tests; moderately-to-highly sure); the weekly run and
  `train` are unaffected either way. Your call, since the contract named all three.
- The inherited `_fill_side_qb` behaviour (above): a fix changes upcoming-week feature values.
- The floor-sigma reference (`HYPOTHESIS.md`'s closing paragraph, triggered): make
  `models/qb_elos_measure/new_file_seed{42,7}` the reference (`constants.FLOOR_SIGMA_REFERENCE_RUNS`
  and `config/weekly_run.yaml`), since production trains on the new file but takes its sigma
  from the old-file step-6 runs. The reviewer recommends yes, between game weeks, recorded in
  `.agents/benchmarks.md`; the sigma barely moves (`13.3184`/`13.3182` from `13.3195`/`13.3227`),
  and the `constants.py` edit invalidates every checkpoint, so it groups with other such changes
  (rule 14). If the win-total file measurement (item 4 of "Next, in order") will rebuild the
  reference soon, swap once, after it.
- The stats-only model: proposed 2026-10-15, not decided. A named track rather than one more
  step at the end, in three parts:
  - in step 8, measure each feature idea on the configuration without game lines as well as on
    production's, since the market already prices injuries and weather;
  - in step 9, tune it as its own study, since its best settings likely differ from the
    anchored model's;
  - a weekly shadow prediction (logged, never submitted), scored against the closing line, so it
    builds a live record.
  On the step-6 references it trails the pick-time line by `+0.00373` Brier with the own Elo and
  `+0.00466` without any Elo (`.agents/benchmarks.md`, "Step 6 reference arms").
- 58.1's exact survivor planner in place of the plan's beam search (acceptance pending).

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer` in a
  fresh context (adversarial tests, uncommitted), fixes back to an implementer, a re-check by a
  reviewer. A `--no-ff` merge (the commit hook needs a Conventional Commits subject on merge
  commits; `git merge` needs `GUARD_OK=1`, only after the user's approval). The delegating
  session writes `TODO.md`, `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file,
  and lints the Markdown it edits.
- Reviewers get a detached scratch worktree with a synced venv, cap pytest with
  `PYTEST_ADDOPTS="-n 8"` when several run at once, and write only their report; at most four at
  once. Every measured number goes through an independent rescore before it enters a doc.
- Run a worktree's gate with `VIRTUAL_ENV` set to that worktree's `.venv`. A backgrounded gate
  wrapper must `exit` the gate's code, never echo after it.
- The bash-guard hook blocks commands containing `git merge` in the main checkout (including
  `merge-tree`) and bare `python3`; use `git diff A...B`, `git rev-list --count A..B` and
  `.venv/bin/python`.
- Inspect another worktree with `git -C <path>` or absolute paths. Run
  `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly|data)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`.
- Worktrees: `~/scratch/step3` (`feat/no-game-lines-groups`), `~/scratch/step3_review`
  (detached at `5c33978`, the step-3 review's adversarial file),
  `~/scratch/qb_measure` (detached at `c637fc0`, the measurement's code and `data/`),
  `~/scratch/qb_adj_docs` (`fix/qb-adj-review-findings`), `~/scratch/qb_adj_review` (detached at
  `6bacd5e`, the review's adversarial file), `~/scratch/step6` (`feat/step6-sos-port`, merged)
  and `~/scratch/step6_runs` (detached at `d6064eb`, the step-6 runs' code); the
  `.claude/worktrees/agent-*` worktrees are the harness's, leave them.
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
