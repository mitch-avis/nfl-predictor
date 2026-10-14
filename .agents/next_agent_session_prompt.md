# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then
`.agents/step6_plan.md` (streams, authority, the user's decisions), `.agents/benchmarks.md` before
any walk-forward, and this file. The main checkout's `.agents/ultracode_session_audit_2026-10-10.md`
(on `main` only) is the audit of the unattended session that built this branch.

## State (written 2026-10-14, after `0.59.0` landed)

- This branch, `feat/step6-sos-port`, is at `0.59.0` (worktree `~/scratch/step6`). `main`
  (`0.45.4` plus the handoff merges, up to `5087a5c`, not pushed) is merged in (`d288907`,
  `1d9f84c`), so the instruction-file restructure, the suppression rule and the three-role testing
  standard apply here. The main checkout stays on `main`; its handoff still reads 2026-10-12 and
  points here.
- Phase 4 is fully reviewed. `models/step6/REVIEW.md` (main checkout; `models/` is not in git)
  now holds three reviews, each by a reviewer that did not produce the runs, each with its
  scripts embedded:
  - Decision 1 (ported ratings in place of the strength snapshot): rule 1's first branch,
    replace. Comparison 1 all weeks, two seeds: Brier −0.00084 [−0.00144, −0.00024].
  - The new-build reference diagnostic, which splits comparison 1: the play-by-play corrections
    are a tie on every column, window and seed (Brier +0.00011 [−0.00040, +0.00062]); the ported
    ratings carry the whole gain (Brier −0.00095 [−0.00153, −0.00039], log loss −0.00218
    [−0.00349, −0.00092], a win on each seed alone, a tie in weeks 1 and 2).
  - Decision 2 (own Elo against the `qb_elos.csv` columns): rule 2's second branch. Comparison 3
    (anchored) loses: Brier +0.00069 [+0.00017, +0.00122], log loss +0.00154, pool points −114.
    Keep the `qb_elos.csv` columns; the own Elo stays optional and off. Comparison 4
    (market-free) wins: Brier −0.00093 [−0.00177, −0.00005], log loss −0.00224, no column lost,
    so the market-free clause adds the own Elo's market-free variant (`own_elo`, no win totals)
    to the market-free configuration. Caveat (the reviewer's P2): the own Elo rates past games
    for the starter known only at kickoff; the previous starter costs it +0.00101 Brier, about
    the size of that win.
  - The user decided on these on 2026-10-12 (below); `0.59.0` adopts them in production.
- Integration progress, 2026-10-14. Each agent branch got one implementer fix round on its
  findings (test first) and a fresh reviewer's re-check; the audit refresh and the blend default
  each needed a second round. Re-check reports are in `~/scratch/step6_review/`
  (`<name>_recheck.md`, `<name>_recheck2.md`; adversarial tests in `<name>/`).
  - Betting-edge split: merged (`91e3eb8`), released as `0.57.2` (`9435542`), gate `0`; reports
    in `.agents/sos_port/review/betting_edges/`. Open P3 for the adoption chunk:
    `floor_sigma.home_win_prob` (`ml/floor_sigma.py:126`) accepts an infinite sigma
    (unreachable today; test first).
  - Phase-1 audit refresh: merged (`d6c9489`), released as `0.57.3` (`db13ae1`), gate `0` (3,357
    tests); 35 judged pairs; reports in `.agents/sos_port/review/audit_refresh/`. Open P3: no test
    covers the run loop's call to `check_opponent_rows` (`.agents/sos_port/parity.py:577-578`).
  - Open rating-definition tests: merged (`327f30d`, the `.agents/sos_port/README.md` conflict
    resolved to the audit's failure list plus the new section), released as `0.57.4` (`ccd4d45`),
    gate `0` (3,488 tests); reports in `.agents/sos_port/review/rating_tests/`, review record in
    `decisions/open_rating_tests.md`. Open P3: the quarterback test's calibration check reads the
    recorded row count, not the fit, and no test shows `run_integrity_checks` calls it; the
    information-set change leaves 1.75 and nulls as they were and the team check does not flip
    later sites; the recorded commit lacks a dirty mark; `season_t_interval` returns NaN bounds for
    one season.
  - Blend ranking default: merged (`fa52884`, no conflict: the branch had merged `1d9f84c`, which
    carries the betting edges), released as `0.58.0` (`cd1b2ac`), gate `0` (3,518 tests); reports
    in `.agents/sos_port/review/blend_default/`; `decisions/power_rankings.md` entry 12 now states
    the reviewed behavior. Accepted without asking, to report at the check-in:
    `power_rankings_method` stays commented out in `config/weekly_run.yaml` (unset ranks on the
    blend, the same report), the docs advise `--no-resume` instead of a new flag, and a
    `composite` skip takes the same error path. Open P3: check the single manifest match against
    the run itself (`wf_best.json` weeks or `created_at`); three older skips at
    `weekly_run/pipeline.py:533-539` log at info level; "market-free" wording in `stage1.py:63,68`
    (docs close-out).
  - Betting-label test, pre-registered in `.agents/sos_port/protocols/betting_labels.md`
    (`5023f45`, corrected in `f5ea2a9` before any scoring): merged (`f76e2cf`), released as
    `0.58.1` (`94fd18f`), gate `0` (3,553 tests); one independent round and a re-check (approve),
    whose own script reproduced the governing number; reports in
    `.agents/sos_port/review/betting_labels/`, record in `decisions/betting_edges.md` entry 3. On
    the 815 real-moneyline games (2022, 2024, 2025), the beyond-the-spread claim minus the model's
    Brier is -0.00036 [-0.00198, +0.00135], a tie, so the labels stay on the whole edge.
  - Production adoption: merged (`ca617a6`), released as `0.59.0` (`9f3ac06`, with the
    benchmarks record `2b556c1`), `scripts/gate.sh --web` `0` (3,603 tests, 283 web tests).
    Production, `train` and `backtest` leave out `strength,own_elo,own_elo_wt` by default (`''`
    or `[]` leaves none out, a list replaces the default, a YAML key with no value is refused);
    the floor sigma reads `models/step6/ratings_anchored_seed{42,7}`; `home_win_prob` refuses a
    sigma that is not finite and positive, not one number, or text; `train` and the weekly run
    stop, and `backtest` warns, when `strength` is left out and the data has strength columns but
    no ported-ratings column with a value (the weekly run checks right after its refresh, before
    the run directory; a dry run that would refresh only warns). One review and two re-checks
    (the last "approve"); reports in `.agents/sos_port/review/adoption/`; DECISIONS.md `0.59.0`.
    `.agents/benchmarks.md` "Step 6 reference arms" holds the reference arms, every step-6
    comparison and the kickoff-starter convention. Open P3s: group lists not normalized for
    order or duplicates; no snapshot of the default fit's predictions; the weekly refresh writes
    where `--data-collection-args` say while the run reads `--data-path` (older).
  - Every merged agent branch and its fix and review worktrees are deleted; the reviewers'
    untracked tests are in `~/scratch/step6_review/<name>/`.
- The win-total runs (approved 2026-10-12) ran on 2026-10-14 (`models/step6/nogamelines_wt/`,
  `driver.log`: both `exit 0`, no failed run). The independent rescore
  (`models/step6/nogamelines_wt/REVIEW.md`, approve with findings, provenance OK) applies the
  rule's second branch: in all weeks over two seeds the win-total variant ties on every column
  (Brier -0.00012 [-0.00080, +0.00055], log loss -0.00030 [-0.00179, +0.00119]), so the
  configuration without game lines keeps the own Elo without win totals (`own_elo`), its current
  setting. No week-1 or week-2 loss. The GPU is free of this project's jobs.
- Worktrees: `~/scratch/step6` (this branch), `~/scratch/step6_runs` (detached at `d6064eb`, the
  runs' code). The `.claude/worktrees/agent-*` worktrees are the harness's; leave them.
- The `/tmp` copies of the power-ranking evaluation (`RANKINGS_EVAL.md`) are snapshotted in
  `~/scratch/step6_review/tmp_rankings_eval_snapshot/`.

## Decisions on record

- Through 2026-10-10: `.agents/step6_plan.md` ("Authority", "User decisions during the run").
- 2026-10-11, on the audit: the blend default merges whole, last of the four, as a minor
  version, and reaches `main` only with the user's approval; rule 1 waits for the reviewed
  diagnostic (now reviewed); the ten-run ladder is ratified; the live-betting design and the
  margin-only switch stay archived at the merge; no rating-definition outcome (every wiped
  snap, every quarterback play, the botched-snap removal) is adopted before the merge, each
  logged as a post-merge measured change; the rating tests' skipped ESPN QBR extras and passer
  holdout are accepted departures; the betting report's three columns and sigma check are
  approved, labels unchanged; `feat/sos-port-inventory` deleted.

- 2026-10-12, on the review session's questions:
  - Replace the strength snapshot with the ported ratings in production (rule 1): yes. Production
    gets `disable_feature_groups: [strength, own_elo, own_elo_wt]`.
  - Keep the play-by-play corrections, which measure as a tie: yes, as correctness fixes; record
    the tie in `.agents/benchmarks.md`.
  - Keep the `qb_elos.csv` columns, the own Elo optional and off (rule 2): yes, tentatively; the
    user may revisit it soon.
  - The own Elo joins the configuration without game lines, in whichever variant measures
    stronger there, with or without the win-total pull; only the variant without win totals has
    been measured in the model. The configuration may need a new name.
  - The kickoff-starter point is a shared convention, not a caveat on adoption: `qb_elos.csv`
    also rates past games for the actual starter, and the user sets each week's projected
    starters by hand before running `../nfeloqb`. Record it once in `.agents/benchmarks.md`.
  - The blend default's failure handling: read the reference runs only in week 1; on a skip, log
    an error, keep the standings, do not mark the stage done; no fallback to `composite`.
  - `main`'s handoff merged into `main`.
  - Two walk-forwards approved (seeds 42 and 7, about 70 minutes on the GPU): the own Elo's
    win-total variant against `models/step6/elo_marketfree_seed{42,7}` in the configuration
    without game lines, on this build from `~/scratch/step6_runs` at `d6064eb`. They wait for the
    GPU, which another job holds until about 2026-10-13. The rule is pre-registered in
    `models/step6/nogamelines_wt/HYPOTHESIS.md` (sha256
    `08bc8c7fdd902e730c0f9de31f41cf657f35aaf11b8a75eeb94959a649970caf`): all weeks, two seeds; a
    Brier win with no log-loss loss adopts the win-total variant, anything else keeps the variant
    without win totals; a week-1 or week-2 loss is flagged, not overriding; cap two runs.
  - The configuration without the market's game lines, anchoring or transform is named "no game
    lines"; "market-free test bench" stays the name of the strict reference without game lines or
    Elo (`models/step4_reference/marketfree_all_seed{42,7}`). The rename in the docs is part of
    the integration session.

## Open for the user

Answered on 2026-10-14 (record in DECISIONS.md and `step6_plan.md` at the docs close-out):

- The missing-ratings check as narrowed (stops `train` and the weekly run, warns in `backtest`,
  fires only when the data has strength columns but no ratings): accepted. After the merge to
  `main` the weekly run stops until `data/` is rebuilt with the ratings.
- The "no game lines" mode (`--wf-market-mode none` and the three market options): make it leave
  out `strength,elo,own_elo_wt`, the reviewed list of `models/step6/elo_marketfree_seed{42,7}`
  (today it uses production's groups). Approved; implement test first after the Week 6 picks.
- The approval and merge session: the user wants it now, before the Week 6 picks (Thursday
  2026-10-15), not on 2026-10-20; a merge before Thursday's game counts as between game weeks.

Still open (asked 2026-10-14):

1. OK to merge this branch into `main` today (`0.59.0`, `gate.sh --web` `0`)?
2. OK to back up `data/` and rebuild it with the ratings (about 5 minutes, then the leakage
   audit)?
3. `../nfeloqb` for Week 6: the user runs it unchanged (sets projected starters by hand, runs
   `workflow.py run_now`, copies `qb_elos.csv` into `data/`), or tells me where the starters go
   and I run it. Touching `../nfeloqb` is must-ask. The win-total fix (its `handle_regression`,
   `nfeloqb/Resources/elo.py:120-142`, reads `Manual Data/wt_ratings.csv`, which ends at 2023;
   the upstream `nfelosrs` file covers 2003-2026) waits until after the picks: it changes
   `qb_elos.csv` for 2024-2026 and needs a build and a walk-forward.
4. Who launches the Week 6 weekly run (the first after the merge retrains stage 1, since every
   fingerprint changed; estimated under an hour on the GPU, not measured).
5. The top-decile ranking for future label reports (decides nothing now): explained on
   2026-10-14 as (a) one list ranked by the two-seed mean edge, recommended as the headline, or
   (b) one list per seed, as a robustness note; the user has not chosen.

## Next: the merge and the Week 6 picks, then the close-out

On the user's approvals, in order:

1. The merge into `main` (`--no-ff`, from the main checkout; check `ss -ltnp` first), then
   `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` there.
2. Back up `data/`, rebuild it (`nfl-predictor data`), run `nfl-predictor leakage-audit`.
3. `../nfeloqb` and the `data/qb_elos.csv` copy, as the user decides.
4. The Week 6 weekly run, as the user decides. Fallback: `main` at `5087a5c` with the backed-up
   `data/`.

After the picks, on this branch or a fresh one off `main`:

1. The "no game lines" groups change, test first through an implementer and a reviewer.
2. The docs close-out: regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (a `models/`
   write: ask), rename "market-free" to "no game lines" where it means the configuration with Elo
   (including `stage1.py:63,68`), refresh the stale "Review:" fields in
   `.agents/sos_port/decisions/*.md`, commit the hypothesis checksums, record the 2026-10-11,
   2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and `TODO.md`, list the open P3s above,
   and build the approvals list and the user's final HTML report.
3. `scripts/gate.sh --web` on the final tree; a browser look at the new pages; the 2026 roof
   recheck before the Paris game on 2026-10-25; the `../nfeloqb` win-total fix decision.

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer` in a
  fresh context (adversarial tests, uncommitted), fixes back to the implementer, a re-check by a
  reviewer. A `--no-ff` merge into this branch (the commit hook needs a Conventional Commits
  subject on merge commits). The delegating session writes `CHANGELOG.md` and the version
  (`pyproject.toml`, `uv lock`, then `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`),
  `TODO.md`, `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file, and lints the
  Markdown it edits.
- Reviewers get a detached scratch worktree with a synced venv, cap pytest with
  `PYTEST_ADDOPTS="-n 8"` when several run at once, and write only their report; at most four at
  once. Every measured number goes through an independent rescore before it enters a doc.
- The bash-guard hook blocks commands containing `git merge` in the main checkout (including
  `merge-tree`); use `git diff A...B` and `git rev-list --count A..B` there.
- Inspect another worktree with `git -C <path>` or absolute paths. Run
  `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`.
- The user runs `nfl-predictor web` from the main checkout only when they want it; check
  `ss -ltnp | grep ':80'` before touching that checkout, and never stop or restart it.
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
