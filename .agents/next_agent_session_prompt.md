# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9) and `.agents/benchmarks.md` before any
walk-forward ("Upstream win totals in the quarterback-adjusted Elo" is the newest section).
`.agents/step6_plan.md` and `.agents/sos_port/DECISIONS.md` hold step 6's history.

## First: the user's decisions

The overnight session of 2026-10-16 finished its plan and left decisions for the user. Before any
other work, walk the user through the open ones below, one line each on what the thing does today
(naming the file), the recommendation first with its cost, and how sure you are. The full list,
with the evidence, is `~/scratch/overnight_2026-10-16/morning_list.md`; the step-6 approvals are
section 5 of `~/scratch/overnight_2026-10-16/step6_report.html` (sent to the user 2026-10-16).

1. **Step 6's fourteen approvals** (the HTML report). Headline: the anchored production model ties
   the closing line at pick time over all weeks but trails it in week 1 (Brier `+0.00506`
   `[+0.00001, +0.01012]`, `benchmarks.md`), so "better early-season calibration" is not met.
2. **Upstream's win-total file as maintenance** (`TODO.md` step 7, the open item). Measured a tie
   on 2023-2025 (Brier `-0.00006` `[-0.00057, +0.00044]`, `models/qb_wt_measure/REVIEW.md`), so the
   rule kept the fork's file; the fork's file ends at 2023, so from 2024 the quarterback-adjusted
   Elo only regresses to the mean at each season start. Recommended: adopt, between game weeks,
   through the fork's `main` (branch `exp/upstream-wt-ratings` `847b6b9` holds the change), with
   `models/qb_wt_measure/upstream_wt_seed{42,7}` as the new floor-sigma reference (a `constants.py`
   change; every checkpoint fingerprint moves).
3. **Step 8's open questions** (`.agents/step8_pick_time.md`): the lagged team-health proxy, both
   tracks (with and without game lines), retiring attendance and parking the referee crew, and the
   pick-time line (the notes assume the Tuesday or Wednesday weekly run).
4. **Retractable roofs** (`TODO.md` step 8): one constant category for those venues in every row
   (a feature-value change; two builds; groups with the next one).
5. **The Python version.** `requires-python = ">=3.14"` with no `.python-version`, so a new venv
   (every worktree's) gets 3.15.0 while the main checkout runs 3.14.8; a 3.15-only pass hid seven
   failing tests until the pre-push gate (fixed, `d39cc4b`). Which Python CI gets is unconfirmed.
   Recommended: pin 3.14 now (pyright's `pythonVersion` is `3.14`), move to 3.15 as its own change.
6. **Smaller calls:**
   - the power-ranking method outside the weekly run (`nfl-predictor rankings` and the web job
     default to the composite, the weekly run uses the blend);
   - who wrote `data/` at 2026-10-16 02:00:07 (544 columns, no run record);
   - deleting `data/cache/nflreadpy/backup_pre_m54_*` and
     `~/scratch/data_backups/2026-10-16_nfl_elo/`;
   - carrying setup-uv 10.3.0 and Dependabot's minor/patch grouping into the Copier template;
   - the QB identity chain follow-up (recommended: keep open).

Until the user answers, take the conservative default: no production default, input or feature
value changes.

## State (written 2026-10-16, 07:50 MDT)

- **`main` is `0.63.8`, pushed** (`origin/main` `3bb59de`, pre-push gate passed), plus the docs
  merge of this handoff (see the last commit). Landed overnight, each independently reviewed:
  - `0.63.1`-`0.63.2`: Paris and Munich roofs overridden to `outdoors`; `nfl_elo.csv` off the
    data status page (the file is unused and parked outside the repo);
  - `0.63.3`: web and CI dependencies (react-table 9, TypeScript 7, setup-uv 10.3.0, Dependabot
    groups only minor and patch updates);
  - `0.63.4`: the leakage audit card lists flags and failures; empty values sort last both ways;
  - `0.63.5`: step 6's docs close-out (follow-ups retired by script and review,
    `RANKINGS_EVAL.md` regenerated, the `models/` cleanup record in
    `.agents/models_cleanup/deleted_2026-10-16.txt`);
  - `0.63.6`: the data status page lists every data file and every source cache;
  - `0.63.7`: review follow-ups (sorted ranking records, skipped rankings logged as warnings,
    `season_t_interval` refuses one season, the rating tests' commit marks a dirty tree);
  - `0.63.8`: no inline lint suppressions left in `.agents/`, and
    `tests/test_no_inline_suppressions.py` guards every tracked `.py` and `.pyi` file;
  - `d39cc4b`: the m60 inventory tests register the module while it loads (Python 3.14).
- **The win-total measurement is done and recorded** (`models/qb_wt_measure/`, `REVIEW.md`,
  `benchmarks.md`): a tie, the fork's file stays (decision 2 above).
- **`../nfeloqb`**: fork `main` `0db2e40` (the backup first-start fix, reviewed, pushed);
  `exp/upstream-wt-ratings` `847b6b9` is the win-total candidate, unmerged.
- `web/dist/` was rebuilt in the main checkout after `0.63.6`. No web server was listening at
  07:45 MDT.
- Worktrees kept on purpose: `~/scratch/qb_measure` (code `c637fc0`, the measurements' builds) and
  `~/scratch/qb_adj_nolines` (code `d791846`, the no-game-lines sigma reference's build).
- ComfyUI (the user's) was using the GPU from about 02:43; a walk-forward waits for 3 GB free and
  then shares the card (results are unaffected, wall time roughly doubles).

## Next work, after the decisions

Work only on the highest-priority blocking items (`TODO.md`, "Roadmap Status"). Step 6 closes when
58.1 and 58.2 are accepted; step 7 has only the adoption item; step 8's measured arms wait for
decision 3; step 9 (the re-tune) comes after steps 7 and 8. Production-changing steps land between
game weeks: the user's Week 7 runs (nfeloqb, then the weekly run) are due before Thursday
2026-10-22.

Small follow-ups ready to take without a decision:

- `_git_head` in `.agents/sos_port/elo_eval.py` (line 127) records the git commit without a
  dirty mark; point it at the shared `git_commit` in `.agents/sos_port/rating_tests_common.py`
  (test first).
- The win-total review's two P3s: `models/qb_wt_measure/build.sh` overwrote the check file the
  hypothesis cites (write a new file instead, in future build scripts); `nfl-predictor compare` has
  no seasons filter, so season-window rules rest on the standalone rescore.

## Decisions on record (2026-10-15 and 2026-10-16)

- Floor-sigma reference with game lines: `models/qb_elos_measure/new_file_seed{42,7}` (`0.62.0`);
  without game lines: `models/qb_adj_no_lines/no_lines_seed{42,7}` (`0.63.0`, chosen by the user).
- The win-total file: the user chose "replace the whole file, two runs" (measured: a tie).
- Overnight merges and pushes of reviewed, gate-green branches were approved for 2026-10-16 only.
- Roofs: Paris (`PAR00`) and Munich (`MUN01`) overridden to `outdoors`; Melbourne as nflverse has
  it; Madrid's blank roof falls under step 8's retractable-roof item.
- Step-6 follow-ups: only the verified-moot ones retired; `RANKINGS_EVAL.md` regenerated.
- `models/` cleanup: aggressive, the 2026 weekly runs kept (record above).
- No-game-lines check: market features and anchor off, the transform not counted; the
  no-game-lines list keeps `qb_adj` out (measured a tie).
- `.agents/m60/INVENTORY.md` and `PROPOSAL.md` stay frozen; `.agents/m60/inventory.py` no longer
  runs against the `src/` layout and stays as a record.
- Finished, independently reviewed roadmap items close without asking (AGENTS.md rule 6);
  reopening, reordering or retiring undone items still asks.

## Open for the user (no rush)

- The stats-only model: proposed 2026-10-15, not decided (measure each feature idea without game
  lines too; tune it as its own study; a weekly shadow prediction scored against the closing line).
- 58.1's exact survivor planner (acceptance pending) and 58.2's browser look at a team page.
- `.agents/` is excluded from Ruff (`pyproject.toml`); linting it would be a campaign
  (recommended: no).
- `origin/feat/web-ui` (fully merged) is still on GitHub for the user to delete.
- nfeloqb pre-existing defects, out of scope: `extract_starter_values` calls a missing
  `QBModel.get_qb_value`; the unknown-QB fallback passes a NaN draft number.

## How the last sessions worked (keep doing this)

- Each code chunk: test first, one file per commit (a signature change lands with its callers),
  `chore: release X` with `CHANGELOG.md`, `pyproject.toml` and `uv.lock`; a separate `reviewer`
  in a fresh context (adversarial tests, uncommitted); a non-producer agent writes measured
  numbers into the docs from `REVIEW.md`; a `--no-ff` merge with a Conventional Commits subject
  (`chore: merge <what> into main`: the commit hook rejects git's default merge message).
- The bash-guard hook blocks any command containing "merge" (`git merge-base` included) unless it
  starts with `GUARD_OK=1`, and bare `python3`; use `git rev-list --count A..B` for ancestry and
  `.venv/bin/python`. It blocks `git push` too (must-ask; `GUARD_OK=1` once approved), and the
  pre-push hook runs the gate.
- After every commit or version change, `uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`,
  or the gate's `uv sync --check` fails.
- A worktree needs its own synced venv
  (`VIRTUAL_ENV=$PWD/.venv uv sync $(<main>/.venv/bin/nfl-lightgbm-cuda-install uv-args)`) for
  the gate and the pre-commit ruff hooks. A backgrounded gate keeps its exit code
  (`...; rc=$?; echo "exit=$rc" >> log; exit $rc`), and its summary is read before any commit.
  Never pipe a lint or gate command through `| tail` in an `&&` chain: it masks the exit code.
  Check the worktree venv's `.venv/bin/python -V`: until a version is pinned, uv picks 3.15 for a
  new venv while the main checkout runs 3.14; recreate it with `uv venv --clear --python 3.14`.
- Stage resolved files before committing a merge (`git add`), and never `git stash` with a
  pathspec while other files are staged: the stash takes the index too.
- Strict pyright in tests: `tests/fakes.returning(value)` for stubs, a typed nested
  `def _capture(message: str, *args: object)` for log capture, typed empty containers
  (`set[tuple[str, ...]]()`).
- Run `.venv/bin/pre-commit run --all-files` before ending a turn. Stop at a natural point well
  before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once.
- A backtest that judges probabilities runs from 2007 week 1 or passes
  `--floor-sigma-reference-runs` with the reference, and passes `--wf-start-week 1`.
- CI (`.github/workflows/validation.yml`) runs `scripts/gate.sh` in a fresh checkout with no
  `data/`, `models/` or `~/scratch`: a test that reads any of them passes locally and fails there.
  `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19 (a runner notice on the 2026-10-16 run).
- Elo research is in `~/scratch/elo_research/`; `nfl-sos-ratings` is frozen at `9081bce`, read
  only as `git -C ../nfl-sos-ratings show 9081bce:<path>`.
