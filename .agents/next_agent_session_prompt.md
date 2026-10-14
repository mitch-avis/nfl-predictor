# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then
`.agents/step6_plan.md` (streams, authority, the user's decisions), `.agents/benchmarks.md` before
any walk-forward, and this file. The main checkout's `.agents/ultracode_session_audit_2026-10-10.md`
(on `main` only) is the audit of the unattended session that built this branch.

## State (written 2026-10-12, after the review session)

- This branch, `feat/step6-sos-port`, is at `0.57.1` (worktree `~/scratch/step6`), 520 commits
  ahead of `main` and 30 behind it. `main` is at `0.45.4` (`667cc2e`, not pushed): the
  instruction-file restructure, no inline lint suppressions, the prompt-audit fixes and the
  three-role testing standard are not on this branch yet. The main checkout stays on `main`.
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
- Four agent branches, each reviewed on 2026-10-12 in a detached scratch worktree with the gate
  re-run (all exit `0`). Each verdict is "approve with findings", no P0 or P1. Full reports, gate
  logs and the reviewers' uncommitted adversarial tests are in `~/scratch/step6_review/`
  (`<name>_review.md`, `<name>/`); the review worktrees are `~/scratch/review_<name>`.
  - Phase-1 audit refresh, `worktree-agent-ab92fdaea474da115`, tip `74641b8` (forked at
    `0.56.0`, 6 commits). Reports reproduce byte for byte (two diagnostic floats differ in the
    last digit); columns 30/526/1,180 to 1,236/500/0 and parity mismatches 40,694 to 25,564,
    none unexplained, verified. P2 (rule 10): the four allowed rates in `gl-defense-rates-same`
    and the opponent-side `gl-carries-faced` are judged "matched" without script output (the
    reviewer's scripts show 0 mismatches over 13,928 team-games each); the 10 web items in
    `same-components` and `trend-chart` are "matched" only by the decision log's silence. P3:
    `mapping.yaml:592` names the team's own column, `parity.py:438` raises `IndexError` on an
    empty schedule, float sums unstable across thread counts, wording in
    `decisions/phase1_refresh.md`, tests to the tip's shared loader `tests/sos_port_scripts.py`.
    No later step-6 commit makes the reports stale; rerun them (2 minutes) after the merge and
    after any `data/` rebuild. No file overlap with this branch.
  - Open rating-definition tests, `worktree-agent-a5c5909054257aa73`, tip `a668e7f` (forked at
    `0.55.1`, 19 commits). Every protocol precedes its script and report; all four reports
    reproduce bit for bit (about 25 minutes); outcomes follow the protocols. P2: the quarterback
    test prints "every calibration used earlier weeks only" without checking it
    (`qb_plays_test.py:2087`); three formatting fixes (`983bdb2`, `fe36d58`, `d368845`) have no
    test. P3: a cycle of wins falls back silently to the first in the order
    (`qb_plays_test.py:326`; rule 12 says ask), a play can count twice in "every quarterback
    play" (0 such plays today), the information-set checks leave some later-game columns
    untouched, a bare `KeyError` in `rating_tests_common.py:317`, the JSON lacks the commit
    (and, for the team tests, the command and runtime). Only `tests/sos_port_scripts.py`
    overlaps, the same blob on both sides.
  - Betting-edge split, `worktree-agent-aa56c16b91074022f`, tip `e9c7a09` (not `00f2d02`; forked
    at `0.57.0`, 23 commits). The fix range `07bd2ca..e9c7a09` addresses every round-two finding
    and nothing else; the three columns carry the approved names; no label column changes in the
    snapshot. P3: an infinite sigma passes the positive-sigma check
    (`reporting/betting_report.py:143`, `weekly_run/pipeline.py:550`: use
    `math.isfinite(sigma) and sigma > 0`, test first); the decision log's Review line should
    name the reviewers and this re-check.
  - Blend ranking default and "no game lines" wording, `worktree-agent-a2071e4c492dbbdda`, tip
    `5aa5c74` (forked at `0.57.0`, 9 commits, all present). The weekly blend is the measured
    blend (rule 11) apart from the prediction source the user approved; `composite` restores the
    old snapshot byte for byte; `5284db2` changes only a log level and message but changes every
    checkpoint fingerprint. P2: a missing or damaged reference run blocks the rankings every
    week, though only week 1 needs it (`reporting/power_rankings.py:1406-1413`,
    `reporting/rating_sources.py:661-684`); a skipped ranking also drops the projected standings,
    logs only a warning, exits `0` and marks the stage done so `--resume` never retries
    (`weekly_run/pipeline.py:528-530`, `:584`). P3: a fold without `neutral` is read as home games
    (`rating_sources.py:481-482`); `--power-rankings-method blend --legacy-franchise-fit` silently
    ranks on Bradley-Terry (`weekly_run/config.py:891-894`); the fixed 0.5 weight for re-ranked
    early seasons is undocumented; stale "market-free" wording (`config/weekly_run.yaml:56-57`,
    `weekly_run/config.py:719-723`); no stage-1 pointer on a resumed old run (`stage1.py:347`); a
    weak info-level test. Overlap: `README.md` conflicts with the betting-edge branch near the
    weekly-rankings bullet (keep this branch's bullet, the betting branch's two paragraphs after
    it, and its removal of the old Totals paragraph); `pipeline.py` should merge cleanly.
- Worktrees: `~/scratch/step6` (this branch), `~/scratch/step6_runs` (detached at `d6064eb`, the
  runs' code), the four `~/scratch/review_*` (detached, disposable once their branches merge).
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
   needs a build and a walk-forward.

## Next: the win-total runs, then the integration session on this branch

The win-total runs come first, once the GPU is free (`nvidia-smi` shows no other job, and
`pgrep -af "nfl-predictor (backtest|weekly|data)"` is empty): `uptime`, check the hypothesis
(`sha256sum --check models/step6/nogamelines_wt/HYPOTHESIS.md.sha256` from the main checkout),
then `models/step6/nogamelines_wt/launch.sh d6064eb` and note the launch in a `launch_record.txt`
beside it. The driver waits for GPU memory and other jobs, but not for another project's job
that holds the GPU with little memory, so the check is by hand. Watch `driver.log` with a
Monitor until both `end` lines arrive; then an independent `reviewer` rescores the pair from the
fold checkpoints into `models/step6/nogamelines_wt/REVIEW.md` and a check-in follows (rule 7).
The integration session need not wait for them: the runs read `~/scratch/step6_runs`, not
this worktree.

1. Bring `main` (`0.45.1`-`0.45.4`) into this branch first and run the gate: the new
   instructions, the reviewer and implementer definitions and the suppression rule then apply to
   everything after.
2. Merge the four branches in this order, each after one implementer fix round on its findings
   (test first) and a reviewer re-check, each as its own version with a `CHANGELOG.md` entry and
   its review report committed under `.agents/sos_port/review/` (lint the report first; two of
   the four fail markdownlint as written): the audit refresh (patch; add the missing parity
   pairs, regenerate the reports on the merged tip), the rating tests (patch; with a
   `DECISIONS.md` index entry), the betting edges (patch), the blend default (minor; its fix
   round includes the failure handling the user approved). Delete each agent branch once merged.
3. The adoption chunk (rule 11), grouped with `5284db2`'s fingerprint change (rule 14), per the
   2026-10-12 decisions: `disable_feature_groups: [strength, own_elo, own_elo_wt]` and
   `floor_sigma_reference_runs: models/step6/ratings_anchored_seed{42,7}` in
   `config/weekly_run.yaml`; the configuration without game lines (with the own Elo variant the
   reviewed win-total runs select) and the reference arms in `.agents/benchmarks.md`.
4. Pre-register and run the betting-label test on the 815 real-moneyline games (a tie keeps
   today's labels).
5. Regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/`, refresh the stale "Review:"
   fields in `.agents/sos_port/decisions/*.md`, commit the hypothesis checksums into git, record
   the 2026-10-11 and 2026-10-12 decisions in `step6_plan.md` and `TODO.md`, and build the
   approvals list and the user's final HTML report.
6. `scripts/gate.sh --web` on the final tree.

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
