# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding. Then read the
step-6 branch's handoff, `~/scratch/step6/.agents/next_agent_session_prompt.md` (every branch's
review verdict and findings, file by file), `.agents/benchmarks.md` before any walk-forward, and
this file. The step-6 plan and decision log live only on the step-6 branch:
`~/scratch/step6/.agents/step6_plan.md` and `~/scratch/step6/.agents/sos_port/DECISIONS.md`.
`.agents/ultracode_session_audit_2026-10-10.md` is the audit of the unattended session that built
that branch.

## State (written 2026-10-12, after the review session)

- `main` holds `0.45.4` plus this handoff (merged with the user's approval), ahead of
  `origin/main` and not pushed. `scripts/gate.sh` exited `0` on the handoff branch before the
  merge. The main checkout stays on `main`.
- Step 6 (absorbing `nfl-sos-ratings`) lives on `feat/step6-sos-port` (`0.57.1`, worktree
  `~/scratch/step6`), about 520 commits ahead of `main` and 30 behind it before this merge.
- The review session (2026-10-12) ran six independent reviewers, each in a fresh context; every
  verdict is "approve with findings", no P0 or P1, and every re-run gate exited `0`.
  - The phase-4 rescores are in `models/step6/REVIEW.md` (decision 1, the new-build reference
    diagnostic, decision 2), each by a reviewer that did not produce the runs.
  - The four agent branches' reports, gate logs and adversarial tests are in
    `~/scratch/step6_review/`; their findings and merge notes are in the step-6 handoff.
- Phase 4, all weeks, two seeds, 95% intervals (rule 3 satisfied):
  - Decision 1, the ported ratings in place of the strength snapshot, anchored: Brier −0.00084
    [−0.00144, −0.00024], rule 1's first branch (replace).
  - The diagnostic splits that: the play-by-play corrections are a tie on every column, window
    and seed (Brier +0.00011 [−0.00040, +0.00062]); the ported ratings carry the whole gain
    (Brier −0.00095 [−0.00153, −0.00039], log loss −0.00218 [−0.00349, −0.00092], a win on each
    seed alone, a tie in weeks 1 and 2).
  - Decision 2, the own Elo against the `qb_elos.csv` columns: anchored, a loss (Brier +0.00069
    [+0.00017, +0.00122], log loss +0.00154, pool points −114), so rule 2's second branch: keep
    the `qb_elos.csv` columns, the own Elo optional and off. Market-free, a win (Brier −0.00093
    [−0.00177, −0.00005], log loss −0.00224, no column lost), so the clause adds the own Elo's
    market-free variant to the market-free configuration. Caveat: the own Elo rates past games
    for the starter known only at kickoff, and the previous starter costs it +0.00101 Brier,
    about the size of that win.
- The four agent branches, all reviewed, none merged: the phase-1 audit refresh
  (`worktree-agent-ab92fdaea474da115`, `74641b8`), the open rating-definition tests
  (`worktree-agent-a5c5909054257aa73`, `a668e7f`), the betting-edge split
  (`worktree-agent-aa56c16b91074022f`, `e9c7a09`) and the blend ranking default with the "no game
  lines" wording (`worktree-agent-a2071e4c492dbbdda`, `5aa5c74`).
- Worktrees: `~/scratch/step6`, `~/scratch/step6_runs` (detached at `d6064eb`, the runs' code),
  and four disposable review worktrees `~/scratch/review_*`. Other branch: `feat/web-ui` (leave
  it alone).
- The user runs `nfl-predictor web` from the main checkout only when they want it (none running
  on 2026-10-12). Check `ss -ltnp | grep ':80'` before a merge or checkout there. Never stop or
  restart a running server yourself (rule 5). Never switch the main checkout off `main`.
- `data/` was rebuilt 2026-10-09 (backup in `data/backup_pre_step4_adopted_20261009/`) and was
  not touched by step 6. The GPU reference is `models/step4_reference/adopted_seed{42,7}`; the
  step-6 test bench is `models/step4_reference/marketfree_all_seed{42,7}`.
- `nfl-sos-ratings` is frozen at `9081bce`; read it only as
  `git -C ../nfl-sos-ratings show 9081bce:<path>`.
- Names: the adopted-settings reference (`adopted_*`), the market-free test bench
  (`marketfree_all_*`), the ported-ratings arms (`ratings_anchored_*`, `ratings_marketfree_*`),
  the new-build reference (`newref_anchored_*`) and the own-Elo arms (`elo_anchored_*`,
  `elo_marketfree_*`). Never use a code alone with the user.

## Decisions on record

- 2026-10-02: drop the defense-adjusted quarterback rate; drop the next-opponent identity pair;
  adopt pick-time lines.
- 2026-10-08: the season-to-date stat blend keeps `K = 4`; roadmap step 5 the market-free
  baseline, step 6 absorbing `nfl-sos-ratings`, step 7 the Optuna re-tune; power rankings combine
  the ported ratings with the model, chosen by measurement.
- 2026-10-09: measure both market-free variants; the market-free option as flags. Standing from
  the `nfl-sos-ratings` session: the quarterback all-plays tie-break order, the botched-snap rule,
  and a pre-registered test before kneel-downs and spikes leave the team ratings.
- 2026-10-10 (in `step6_plan.md`): the weekly run publishes the blend at weight 0.5; the
  stats-only mode may keep the win-total prior and means "no game lines"; a pre-registered
  betting-label test, a tie keeping today's labels; feature ideas move to a later session; the
  `../nfeloqb` fix waits for decision 2 and the user's go-ahead.
- 2026-10-11, on the audit: the blend default merges whole, last of the four, as a minor
  version, reaching `main` only with the user's approval; rule 1 waits for the reviewed
  diagnostic (now reviewed); the ten-run ladder is ratified; the live-betting design and the
  margin-only switch stay archived; no rating-definition outcome is adopted before the merge
  (each a post-merge measured change); the rating tests' skipped ESPN QBR extras and passer
  holdout are accepted departures; the betting report's three columns and sigma check are
  approved, labels unchanged; the three-session plan is accepted.

- 2026-10-12, on the review session's questions:
  - Replace the strength snapshot with the ported ratings in production (rule 1): yes.
  - Keep the play-by-play corrections, which measure as a tie: yes, as correctness fixes.
  - Keep the `qb_elos.csv` columns, the own Elo optional and off (rule 2): yes, tentatively; the
    user may revisit it soon.
  - The own Elo joins the configuration without game lines, in whichever variant measures
    stronger there, with or without the preseason win-total pull; only the variant without win
    totals has been measured in the model. The configuration may need a new name, since with win
    totals it is no longer free of market information.
  - The kickoff-starter point is a shared convention, not a caveat on adoption: `qb_elos.csv`
    also rates past games for the actual starter, and the user sets each week's projected
    starters by hand before running `../nfeloqb`. Record it once in `.agents/benchmarks.md`.
  - The blend default's failure handling: read the reference runs only in week 1; on a skip, log
    an error, keep the projected standings, do not mark the stage done; no fallback to
    `composite`.
  - Merge this handoff into `main` (done).

## Open for the user

1. Approve two walk-forwards (seeds 42 and 7, about 70 minutes on the GPU) that pit the own
   Elo's win-total variant against the variant without win totals in the configuration without
   game lines, on the step-6 candidate build from `~/scratch/step6_runs`. Proposed rule, written
   into a new `models/step6/HYPOTHESIS_*.md` before launch: all weeks, two seeds; a Brier win
   with no log-loss loss adopts the win-total variant, anything else keeps the variant without
   win totals.
2. The configuration's name. Proposed: "no game lines", the user's own definition of 2026-10-10,
   already used by the blend branch's help text; "market-free test bench" stays the name of the
   strict reference without game lines or Elo.
3. The `../nfeloqb` fix, for the approval session. The fork's `handle_regression`
   (`nfeloqb/Resources/elo.py:120-142`) pulls each preseason Elo toward the Vegas win-total
   rating only when its local `Manual Data/wt_ratings.csv` has that season, and that file ends
   at 2023. So 2024-2026 get plain regression only, unlike every earlier season. The upstream
   `greerreNFL/nfelosrs` `wt_ratings.csv` covers 2003-2026. Pointing the fork at it changes the
   `qb_elos.csv` values for 2024-2026, so it needs a new build and a walk-forward before it
   reaches production.

## Next

1. **Integration session on `feat/step6-sos-port`** (the step-6 handoff has the detail): bring
   `main` into the branch first; merge the four reviewed branches in order (audit refresh, rating
   tests, betting edges, blend default), each after one fix round on its findings and a
   reviewer re-check, each with its version, `CHANGELOG.md` entry and committed review report;
   the adoption chunk per the 2026-10-12 decisions (production disables `strength`, `own_elo`
   and `own_elo_wt`; `floor_sigma_reference_runs` points at
   `models/step6/ratings_anchored_seed{42,7}`), grouped with the blend branch's fingerprint
   change; the own-Elo variant for the configuration without game lines once item 1 under "Open
   for the user" is settled; the pre-registered betting-label test; the docs close-out, the
   approvals list and the final HTML report; `scripts/gate.sh --web` on the final tree.
2. **Approval and merge session with the user, between game weeks** (after Monday night, before
   Thursday): the approvals list and HTML report; the merge to `main` in a temporary worktree;
   the `data/` rebuild with a backup; `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`;
   a browser look at the new pages; the 2026 roof recheck before the Paris game on 2026-10-25;
   then the `../nfeloqb` fix decision.

Not scheduled yet: projected standings on the ported ratings, the source's prior and wp-filter
reruns and data notes, a week-1 quarterback carryover, and the ratings refit speed-up. Weekly
runs and picks are the user's.

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer` in a
  fresh context, fixes back to the implementer, a re-check by a reviewer. A `--no-ff` merge into
  the step's branch. The delegating session writes `CHANGELOG.md` and the version, `TODO.md`,
  `ARCHIVE.md`, `AGENTS.md`, `.agents/benchmarks.md` and this file, and lints the Markdown it
  edits.
- Reviews: a detached scratch worktree per branch with a synced venv
  (`uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`), at most four reviewers at once,
  pytest capped with `PYTEST_ADDOPTS="-n 8"` while several run. Two rescores that feed one
  `REVIEW.md` write separate section files that the main session splices in.
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Never pipe a check through `tail` or `head` in a way that hides its exit status.
  Never send the user's email address to any external service. Never use the system `python3`.
- Merges into `main` run in the main checkout, which stays on `main`, with `GUARD_OK=1` once the
  user has approved (the bash-guard hook blocks it otherwise, and blocks any command starting
  `git merge`, including `merge-base` and `merge-tree`; use `git rev-list --count A..B` or
  `git diff A...B`). Check that no web server or job runs from the main checkout first.
- Inspect another worktree with `git -C <path>` or absolute paths. Run
  `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`
  (rule 11).
- Re-run `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` after every version bump and
  every merge or checkout that changes the version or the layout.
- Elo research is in `~/scratch/elo_research/`.
