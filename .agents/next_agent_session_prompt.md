# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then
`.agents/step6_plan.md` (streams, authority, the user's decisions), `.agents/benchmarks.md` before
any walk-forward, and this file. The main checkout's `.agents/ultracode_session_audit_2026-10-10.md`
(on `main` only) is the audit of the unattended session that built this branch.

## State (written 2026-10-14, during the integration session)

- This branch, `feat/step6-sos-port`, is at `0.57.3` (worktree `~/scratch/step6`). `main`
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
  - The user decided on these on 2026-10-12 (below); nothing is adopted on the branch yet.
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
  - Open rating-definition tests, `worktree-agent-a5c5909054257aa73`, tip `8df3932` (fix worktree
    `~/scratch/fix_rating_tests`, `1d9f84c` merged in as `2589b68`): fix round done; re-check
    running in `~/scratch/review_rating_tests`. Next: merge as `0.57.4` with a `DECISIONS.md`
    index entry and its reports under `.agents/sos_port/review/rating_tests/`.
  - Blend ranking default, `worktree-agent-a2071e4c492dbbdda`, tip `085211d` (fix worktree
    `~/scratch/fix_blend_default`, `1d9f84c` merged in as `6197f58`): two fix rounds, both
    re-checks "approve with findings"; the user's failure handling is in (reference runs read in
    week 1 only; a skip logs an error, keeps the standings and leaves the stage undone). Accepted
    without asking, to report at the check-in: `power_rankings_method` stays commented out in
    `config/weekly_run.yaml` (the same report hash), the docs advise `--no-resume` instead of a
    new flag, and a `composite` skip takes the same error path. Follow-ups: check the single
    manifest match against the run itself (`wf_best.json` weeks or `created_at`); three older
    info-level skips (`weekly_run/pipeline.py:532-539`); "market-free" wording in
    `stage1.py:63,68` (docs close-out). Merge last as `0.58.0`; `README.md` conflicts with the
    betting merge near the weekly-rankings bullet (keep this branch's bullet, the betting
    branch's two paragraphs after it and its removal of the old Totals paragraph), then rerun
    `tests/test_weekly_run_characterization.py`.
- The win-total runs (approved 2026-10-12) launched on 2026-10-14 from
  `models/step6/nogamelines_wt/launch.sh d6064eb` (`launch_record.txt`). Seed 42 finished
  (`end elo_wt_nogamelines_seed42 exit 0 seconds 3122`); seed 7 started at 14:23. Once its `end`
  line is in `driver.log`, an independent `reviewer` rescores the pair from the fold checkpoints
  into `models/step6/nogamelines_wt/REVIEW.md`, under the rule in `HYPOTHESIS.md`.
- Worktrees: `~/scratch/step6` (this branch), `~/scratch/step6_runs` (detached at `d6064eb`, the
  runs' code), `~/scratch/fix_*` and `~/scratch/review_*` for the two unmerged branches (remove
  each pair once its branch merges, after copying the review worktree's untracked tests into
  `~/scratch/step6_review/<name>/`). The `.claude/worktrees/agent-*` worktrees are the harness's;
  leave them.
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

1. The `../nfeloqb` fix, for the approval session: the fork's `handle_regression`
   (`nfeloqb/Resources/elo.py:120-142`) applies the win-total pull only for seasons in its local
   `Manual Data/wt_ratings.csv`, which ends at 2023, so 2024-2026 get plain regression. The
   upstream `nfelosrs` file covers 2003-2026; switching changes `qb_elos.csv` for 2024-2026 and
   needs a build and a walk-forward. Asked on 2026-10-14 with the recommendation to run
   `../nfeloqb` unchanged for week 6 and decide after the step-6 merge.
2. The approval and merge session: proposed for Tuesday 2026-10-20 to Wednesday 2026-10-21,
   between game weeks 6 and 7 (asked 2026-10-14).

## Next: finish the integration session on this branch

1. When the rating-tests re-check approves: merge `worktree-agent-a5c5909054257aa73` with
   `--no-ff`, commit its reports under `.agents/sos_port/review/rating_tests/` (lint them first),
   add the `DECISIONS.md` index entry, release `0.57.4` (`CHANGELOG.md`, `pyproject.toml`,
   `uv lock`, the CUDA sync), run the gate, delete the branch and its two worktrees.
2. Merge the blend default the same way as `0.58.0` (minor), resolving the `README.md` conflict
   as above.
3. When seed 7 ends: the independent rescore into `models/step6/nogamelines_wt/REVIEW.md`, then a
   check-in (rule 7) with the rule's verdict, and tell the user the GPU is free.
4. The adoption chunk (rule 11), grouped with `5284db2`'s fingerprint change (rule 14), per the
   2026-10-12 decisions: `disable_feature_groups: [strength, own_elo, own_elo_wt]` and
   `floor_sigma_reference_runs: models/step6/ratings_anchored_seed{42,7}` in
   `config/weekly_run.yaml`; the configuration without game lines (with the own Elo variant the
   reviewed win-total runs select); the finite-sigma guard in `floor_sigma.home_win_prob` (test
   first); the reference arms and the kickoff-starter convention in `.agents/benchmarks.md`.
5. Pre-register and run the betting-label test on the 815 real-moneyline games (a tie keeps
   today's labels).
6. Regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/`, rename "market-free" to "no game
   lines" where it means the configuration with Elo (including `stage1.py`), refresh the stale
   "Review:" fields in `.agents/sos_port/decisions/*.md`, commit the hypothesis checksums into
   git, record the 2026-10-11, 2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and
   `TODO.md`, list the follow-ups above, and build the approvals list and the user's final HTML
   report.
7. `scripts/gate.sh --web` on the final tree.

Then the approval and merge session with the user, between game weeks: the merge to `main`, the
`data/` rebuild with a backup, the CUDA sync, a browser look at the new pages, the 2026 roof
recheck before the Paris game on 2026-10-25, and the `../nfeloqb` fix decision.

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
