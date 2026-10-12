# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding. Then read
`.agents/ultracode_session_audit_2026-10-10.md` (the audit of the unattended step-6 session: what
landed, what is half done, the four unmerged branches, the breaches), `.agents/benchmarks.md`
before any walk-forward, and this file. The step-6 plan and decision log live only on the step-6
branch: `~/scratch/step6/.agents/step6_plan.md` and `~/scratch/step6/.agents/sos_port/DECISIONS.md`.

## State (written 2026-10-12, after the step-6 session audit)

- `main` holds `0.45.4`, ahead of `origin/main` and not pushed: the instruction-file restructure
  (`0.45.1`), no inline lint suppressions (`0.45.2`), the prompt-audit fixes (`0.45.3`) and the
  three-role testing standard (`0.45.4`), plus the step-6 session audit and this handoff (docs
  only). Under the three-role standard the task text carries the test contract, the
  `implementer` never weakens a contracted case, and the `reviewer` writes adversarial tests
  against the diff and reports which fail without committing them (`.claude/agents/`).
  `scripts/gate.sh` exited `0` on `main` at `0.45.4`.
- Step 6 (absorbing `nfl-sos-ratings`) lives on `feat/step6-sos-port`, tip `2efbd3e` (`0.57.1`),
  520 commits ahead of `main`, worktree `~/scratch/step6`. Fifteen versions landed there on
  2026-10-10, each after an independent review. That branch's own handoff is stale on two points:
  the blend-default branch did not die (all 9 commits are there), and the betting-edge branch's
  tip is `e9c7a09`, not `00f2d02`.
- Four branches hold finished, unmerged work (worktrees removed; the audit has their contents):
  - `worktree-agent-ab92fdaea474da115` (6 commits, tip `74641b8`): the refreshed phase-1
    inventory and parity audit. Unreviewed.
  - `worktree-agent-a5c5909054257aa73` (19, tip `a668e7f`): the open rating-definition tests.
    Unreviewed; it skipped the pre-registered ESPN QBR extras and the passer holdout.
  - `worktree-agent-aa56c16b91074022f` (23, tip `e9c7a09`): the betting-edge split, including
    three new columns in `reporting/betting_report.py`. Reviewed in two rounds; the last fixes
    `07bd2ca..e9c7a09` were never re-checked.
  - `worktree-agent-a2071e4c492dbbdda` (9, tip `5aa5c74`): the blend ranking default
    (`6c0c285`, breaking) and the "no game lines" wording (`5284db2` edits `ml/feature_spec.py`,
    which changes every checkpoint fingerprint). Review stopped at its start.
- `feat/sos-port-inventory` was fully contained in the step-6 branch and was deleted 2026-10-11.
- Phase 4 walk-forwards: all ten runs under `models/step6/` exited `0` (`driver.log`), built
  from `d6064eb` (`0.55.0`) into `~/scratch/step6/data`. `models/step6/REVIEW.md` covers
  decision 1 only (ported ratings replace the strength snapshot: rule 1's first branch). The
  new-build reference diagnostic (`newref_anchored_seed{42,7}`) and decision 2, the own Elo
  (`elo_anchored_seed{42,7}`, `elo_marketfree_seed{42,7}`), are unreviewed; the session's own
  check-ins in `models/step6/checkins/` are not a second key. Rules and the diagnostic's
  amendment are in `models/step6/HYPOTHESIS.md`.
- Worktrees: `~/scratch/step6` (`feat/step6-sos-port`) and `~/scratch/step6_runs` (detached at
  `d6064eb`; the runs' code). Other branch: `feat/web-ui` (leave it alone).
- The user runs `nfl-predictor web` from the main checkout only when they want it (last seen on
  port 8001 without `--reload`); none was running on 2026-10-11. Check `ss -ltnp | grep ':80'`
  before a merge or checkout there. Never stop or restart a running server yourself (rule 5).
  Never switch the main checkout off `main`.
- `data/` was rebuilt 2026-10-09 (backup in `data/backup_pre_step4_adopted_20261009/`) and was
  not touched by step 6. The GPU reference is `models/step4_reference/adopted_seed{42,7}`; the
  step-6 test bench is `models/step4_reference/marketfree_all_seed{42,7}`; both are reviewed
  (`models/step4_reference/REVIEW.md`).
- `nfl-sos-ratings` is frozen at `9081bce`; read it only as
  `git -C ../nfl-sos-ratings show 9081bce:<path>`. Its open items are in
  `.agents/sos_port_source.md`.
- The user's main model is Opus 5.5 at medium effort; subagents run Opus 5.5 at medium with turn
  caps.
- Names: the adopted-settings reference (`adopted_*`), the market-free run without betting-line
  or Elo columns, which is the test bench (`marketfree_all_*`), the ported-ratings arms
  (`ratings_anchored_*`, `ratings_marketfree_*`), the new-build reference (`newref_anchored_*`)
  and the own-Elo arms (`elo_anchored_*`, `elo_marketfree_*`). Never use a code alone with the
  user.

## Decisions on record

- 2026-10-02: drop the defense-adjusted quarterback rate; drop the next-opponent identity pair;
  adopt pick-time lines.
- 2026-10-08: the season-to-date stat blend keeps `K = 4`; roadmap step 5 the market-free
  baseline, step 6 absorbing `nfl-sos-ratings`, step 7 the Optuna re-tune; power rankings combine
  the ported ratings with the model, chosen by measurement.
- 2026-10-09: measure both market-free variants; the market-free option as flags. Standing from
  the `nfl-sos-ratings` session: the quarterback all-plays tie-break order, the botched-snap rule,
  and a pre-registered test before kneel-downs and spikes leave the team ratings.
- 2026-10-10 (recorded in `~/scratch/step6/.agents/step6_plan.md`): the weekly run publishes the
  blend at weight 0.5; the stats-only mode may keep the win-total prior and means "no game
  lines"; a pre-registered betting-label test, a tie keeping today's labels; feature ideas
  (referee, weather, attendance, team health) move to a later session; the `../nfeloqb` fix waits
  for decision 2 and the user's go-ahead.
- 2026-10-11, on the audit's recommendations:
  - The blend default: review `worktree-agent-a2071e4c492dbbdda`, then merge it whole into the
    step-6 branch, last of the four, as a minor version. It reaches `main` only with the user's
    final approval.
  - Rule 1 of decision 1 waits for the reviewed new-build diagnostic before anything acts on it.
  - The session's ten-run ladder is ratified, and the live-betting design (58.3) and the
    margin-only switch (58.6) stay archived at the merge.
  - None of the rating-definition outcomes (every wiped snap, every quarterback play, the
    botched-snap removal) is adopted before the merge; log them as a post-merge measured change
    (each needs a new build and walk-forward).
  - `feat/sos-port-inventory` deleted (done).
  - The rating tests' skipped ESPN QBR extras and passer holdout are accepted as logged
    departures.
  - The betting report's three new columns (`market_home_prob_spread`,
    `edge_beyond_spread_home_prob`, `spread_moneyline_gap_home_prob`) and its sigma check are
    approved, labels unchanged.
  - The audit's three-session plan (below) is accepted.

## Next

1. **Review session, no implementers** (this session). `reviewer` subagents with turn caps, at
   most four at once, each in a fresh context:
   - Rescore the new-build diagnostic from the fold checkpoints against the amendment in
     `models/step6/HYPOTHESIS.md`, and append it to `models/step6/REVIEW.md`.
   - Rescore decision 2 (own-Elo arms against the ported-ratings arms, anchored and
     market-free) under rule 2 of the same file, and append it to `REVIEW.md`. No new runs.
   - Review `worktree-agent-ab92fdaea474da115` and `worktree-agent-a5c5909054257aa73` in full,
     `worktree-agent-a2071e4c492dbbdda` in full, and `07bd2ca..e9c7a09` on
     `worktree-agent-aa56c16b91074022f`. Each reviewer re-runs the gate in a scratch worktree of
     its branch and writes adversarial tests without committing them.
   - The main session rewrites the step-6 branch's handoff with the true branch states and the
     verdicts, then rewrites this file. Merges wait for the next session. Check in after each
     rescore with the governing-window numbers (rule 7).
2. **Integration session on `feat/step6-sos-port`.**
   - Merge the reviewed branches in order: the audit refresh, the rating tests, the betting edges,
     then the blend default. Each gets its own incremented version and `CHANGELOG.md` entry; the
     blend default is the minor bump. Delete each agent branch once merged.
   - The adoption chunk (rule 11), grouped with `5284db2`'s fingerprint change (rule 14): set
     `disable_feature_groups` and `floor_sigma_reference_runs` in `config/weekly_run.yaml` to the
     adopted arms per the outcomes of rules 1 and 2.
   - Pre-register and run the betting-label test on the 815 real-moneyline games.
   - Regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (it exists only in `/tmp`),
     refresh the stale "Review:" fields in `.agents/sos_port/decisions/*.md`, commit the
     hypothesis checksums into git, record the 2026-10-11 decisions in `step6_plan.md` and
     `TODO.md`, and build the approvals list and the user's final HTML report.
   - `scripts/gate.sh --web` on the final tree.
3. **Approval and merge session with the user, between game weeks** (after Monday night, before
   Thursday): the approvals list and HTML report; the merge to `main` in a temporary worktree;
   the `data/` rebuild with a backup; `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`;
   a browser look at the new pages; the 2026 roof recheck before the Paris game on 2026-10-25;
   then the `../nfeloqb` fix decision.

Not scheduled yet (the audit's "Not started" has the full list): projected standings on the
ported ratings, the source's prior and wp-filter reruns and data notes, the "no game lines"
measurement with the win-total Elo, a week-1 quarterback carryover, and the ratings refit
speed-up. Weekly runs and picks are the user's.

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer, and a second review round for anything with logic. A fix round is
  re-checked by a reviewer, not by the delegating session. A `--no-ff` merge into the step's
  branch (the commit hook needs a Conventional Commits subject on merge commits too). The
  delegating session writes `CHANGELOG.md` and the version (`pyproject.toml`, `uv lock`, then
  `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`), `TODO.md`, `ARCHIVE.md`,
  `AGENTS.md`, `.agents/benchmarks.md` and this file, and lints the Markdown it edits.
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Hypotheses and decision rules are written before launch and committed so they are
  tamper-evident; long runs go through one detached driver script with a PID watcher.
- Never pipe a check through `tail` or `head` in a way that hides its exit status. Never send the
  user's email address to any external service, including in a User-Agent header. Never use the
  system `python3`; use `.venv/bin/python`. Closing or archiving roadmap items is must-ask.
- Merges into `main` run in the main checkout, which stays on `main`: `git worktree add` refuses
  a branch that is checked out elsewhere. Build and commit the branch in a scratch worktree
  first, check that no web server or job runs from the main checkout, and run the merge there
  with `GUARD_OK=1` once the user has approved it (the bash-guard hook blocks it otherwise).
- Inspect an agent's worktree with `git -C <path>` or absolute paths; a `cd` into it moves this
  session's working directory there.
- Run `.venv/bin/pre-commit run --all-files` before ending a turn (the Stop hook runs it). Stop
  at a natural point well before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the GPU reference, and passes `--wf-start-week 1`
  (rule 11).
- Re-run `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)` after every version bump and
  every merge or checkout that changes the version or the layout.
- The bash-guard hook denies any command starting `git merge` (including `git merge-base`) while
  the project is on `main`; use `git rev-list --count A..B` or `git diff A...B` instead.
- Elo research (538's code and methodology, `nfelo`, upstream `nfeloqb`, the win-total file) is in
  `~/scratch/elo_research/`.
