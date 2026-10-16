# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-16) as binding, then this file,
`.agents/TODO.md` ("Roadmap Status", steps 6-9) and `.agents/benchmarks.md` before any
walk-forward ("The quarterback adjustment without game lines" is the newest section).
`.agents/step6_plan.md` and `.agents/sos_port/DECISIONS.md` hold step 6's history.

## First: ask everything up front, then work through the night

The user starts this session and then goes to bed: this is 8-10 hours of unattended work. Before
any other work, spend at most 20 minutes on the checks below, then ask every question in one go
(`AskUserQuestion`, at most four per call, so two calls back to back). Each question gets one line
on what the thing does today (naming the file), the recommendation first with its cost, and how
sure you are (global rules). After the answers, do not stop to ask again: anything new that needs
the user goes on the morning list, and the work moves to the next item. If a question goes
unanswered, take its "default" below, which is always the conservative choice.

Checks before asking (read-only):

- The fork's win-total file against upstream's: `../nfeloqb/nfeloqb/Manual Data/wt_ratings.csv`
  (2003-2023, read by `nfeloqb/Resources/elo.py:35` and applied at each season start, lines
  ~133-142, to the quarterback-adjusted `qbelo` track only) against `greerreNFL/nfelosrs`'s
  `wt_ratings.csv` (cached 2026-10-10 as `~/scratch/elo_research/wt_ratings_nfelosrs.csv`,
  2003-2026; refresh it if upstream moved). They differ before 2024 as well: Arizona 2003 has
  `wt_rating_elo` `1392.85` upstream against `1344.55` in the fork, so the methods differ. Count
  how many team-seasons differ and by how much, and confirm upstream's `wt_rating_elo` is on the
  scale `elo.py` expects.
- The Paris roof (`.agents/sos_port/decisions/pbp_columns.md`, entry 13): read the 2026 schedule
  from nflverse (`nflreadpy`, into a scratch directory, never `data/`) and check whether `PAR00`
  (Stade de France, Week 7, 2026-10-25) is still `dome`. It feeds `stadium_type`.
- The frontend Dependabot PR (#4): list which of its 14 updates are majors (known:
  `@tanstack/react-table` 8 to 9, TypeScript 6 to 7, `@types/node` 24 to 26), and read
  react-table 9's migration notes. Its `web` job fails only in
  `web/src/components/table/DataTable.tsx` (16 TypeScript errors: `useReactTable`,
  `getCoreRowModel` and `getSortedRowModel` are gone, `Table`/`Row`/`ColumnDef` take more type
  arguments).
- The step-6 follow-ups that concern only the strength snapshot (the "6, phases 3-4" row of the
  follow-up table in `TODO.md` and the groups it names), since the grouped measurement kept the
  ported ratings: the list the user is asked to retire.

Questions (recommendation first; default if unanswered):

1. **The win-total file.** Replace the fork's file with upstream's whole file (one maintained
   source and one method for every season; changes `qbelo` from 2003 on), append only upstream's
   2024-2026 rows (changes 2024-2025 only, but mixes two methods), or measure both (four runs in
   place of two). Recommend replacing the whole file: two runs, about 1.5 hours of compute. Rule 4
   makes a third and fourth run must-ask, hence the question. Default: replace, two runs.
2. **Overnight merges and pushes.** May each finished branch (gate green, independent review
   passed) be merged `--no-ff` into `main` and `main` pushed, without waking the user? Planned
   tonight: the frontend upgrade, the step-6 docs close-out, the roof fix if one is needed. The
   nfeloqb fix's push to the fork was approved on 2026-10-15. Nothing planned changes a
   production default; the win-total result is a recommendation only. Recommend yes. Default:
   merge nothing, leave the branches for the morning.
3. **The frontend majors** (also resolves Dependabot PRs #3 and #4): take all of Dependabot's
   updates locally on one branch (`astral-sh/setup-uv` 10.3.0 in `.github/workflows/validation.yml`,
   and the 14 frontend updates with `DataTable.tsx` moved to react-table 9 and TypeScript 7), so
   Dependabot closes both PRs once `main` has the versions; and in `.github/dependabot.yml` group
   only minor and patch updates, so a breaking major arrives as its own PR. Both are CI and
   dependency changes, which the global rules make ask-first. Recommend yes to both. Default: take
   only the minor and patch updates, and leave the majors and `dependabot.yml` for the user.
4. **Retire the strength-snapshot follow-ups** (rule 5): the list from the checks, one
   multi-select option per item. Default: retire none.
5. **`RANKINGS_EVAL.md`**: regenerate it under `models/step6/rankings/` (a write under `models/`,
   must-ask). Recommend yes. Default: no.
6. **Step 8, early start**: once steps 6 and 7's items below are done, write step 8's pick-time
   checks (what is known before Thursday for the referee crew, weather, attendance and injury
   reports): research notes under `.agents/`, no code, no runs. Recommend yes. Default: no.
7. **Only if the Paris roof is still `dome`**: give `PAR00` an `outdoors` override before the
   user's Week 7 run (a feature-value change for one game, landing before Thursday 2026-10-22),
   or leave it to nflverse. Recommend leaving it (one game; the source may still be corrected).
   Default: leave it.

## State (written 2026-10-16, 02:00 MDT)

- **`main` is `0.63.0` at `41f8902`, pushed to `origin`** (the user approved the merge and the
  push). Two merges this session:
  - `df6063b`: `fix/no-game-lines-sigma-reference`. A run without game lines (the weekly run's
    `--wf-market-mode none`, or `train` without market features and anchor) defaults its
    floor-sigma reference to `constants.NO_GAME_LINES_FLOOR_SIGMA_REFERENCE_RUNS`
    (`models/qb_adj_no_lines/no_lines_seed{42,7}`). Production (mode hybrid) resolves to the same
    runs and settings as before. `constants.py` changed, so every walk-forward checkpoint
    fingerprint moved; the user's Week 7 run retrains. Reviewed in a fresh context; the reviewer's
    25 adversarial tests pass (uncommitted, `~/scratch/sigma_review/`).
  - `41f8902`: the CI fix. `Validation` had failed on every push since 2026-10-15 on one test,
    `tests/test_sos_port_audit.py::test_parse_args_without_data_dir_reads_the_main_checkout_data`,
    which needed the main checkout's `data/` (never in git, so absent in CI). Reproduced in a
    fresh clone without `data/`; the test is now two: the default is `MAIN / "data"`, and
    `parse_args` returns the default with `DATA` patched to a temporary directory. No version bump
    (test only).
  - Gate exit `0` on `main` before the push (4,407 tests, `src/` coverage 96%), and again in the
    pre-push hook; pre-commit clean. GitHub's `Validation` run on `41f8902`: `validate` and `web`
    both passed, the first green `main` run since 2026-10-15 (UTC).
- **Dependabot PRs, both still open**: #3 (`astral-sh/setup-uv` 10.2.0 to 10.3.0) failed only on
  the test fixed above; #4 (14 frontend updates) failed that test and the `web` job (react-table 9,
  see the checks). Neither reruns by itself: Dependabot rebases only on conflicts. The auto-mode
  classifier refused this session's `@dependabot rebase` comments and the remote branch deletion
  (GitHub writes); question 3 resolves both PRs from local commits instead.
- **Branches cleaned**: locally only `main` is left. The 20 harness worktrees under
  `.claude/worktrees/` were clean and every one of their commits was on `main` (`git cherry`
  showed all as already applied); they and their `worktree-agent-*` branches are removed, with
  `chore/pyproject-standards`, `docs/qb-elos-measurement-record`, `feat/qb-adj-features`,
  `feat/web-ui` and `fix/no-game-lines-sigma-reference`. `origin/feat/web-ui` (fully merged) is
  still on GitHub: the user deletes it (`git push origin --delete feat/web-ui`). Kept on purpose:
  the detached worktrees `~/scratch/qb_measure` (code `c637fc0`, the grouped measurement's builds,
  for the win-total measurement) and `~/scratch/qb_adj_nolines` (code `d791846`, whose runs are
  the sigma reference without game lines), and `~/scratch/step6/data`.
- `data/` in the main checkout is the 2026-10-14 build (no `qb_adj` columns, which production
  leaves out). The user's Week 7 weekly run rebuilds every season by default.
- `../nfeloqb` fork `main` is at `9bf32f2` (pushed). The user runs it each week (memory
  `nfeloqb-weekly-process`); the Week 7 run is due before Thursday 2026-10-22.
- No web server was listening at 01:36 MDT (`ss -ltnp`). Check before a `web/dist/` build or an
  edit, and never stop or restart one the user started.

## Tonight's plan, in order (about 8-10 hours)

Compute: one walk-forward at a time (`uptime`, `nvidia-smi`,
`pgrep -af "nfl-predictor (backtest|weekly|data)"` first). The build and both runs go through one
driver script launched with `nohup setsid`, watched by its PID (`while kill -0 <pid>`, never
`pgrep -f` on a path). Code work runs in parallel with the runs: delegate each code chunk to an
`implementer` in its own worktree, then a `reviewer` in a fresh context.

1. **`../nfeloqb`, a backup's first start** (approved 2026-10-15, test first; about 1-1.5 hours;
   must land before the user's Week 7 nfeloqb run). The next-week projection rates the starter
   with `current_value`, while the played row uses `get_value(team)`, which applies the backup
   rule (Baltimore's Week 5 projection was off by `0.8`). Read `get_value` first and confirm it
   uses only pre-game information; then a failing test in `tests/test_elo_qb_adjustment.py`, the
   fix in `elo_file_constructor.py`, the fork's gates (`.venv/bin/ruff format .`, `ruff check .`,
   `ty check .`, `pyright .`, `markdownlint-cli2 "**/*.md" "#.venv"`,
   `.venv/bin/python -m pytest`), a `fix:` commit, an independent review, and a push to the fork
   (`GUARD_OK=1`). It changes only unplayed rows, so it does not touch the win-total measurement
   (cut through 2025).
2. **The win-total measurement** (approved 2026-10-15 as its own measurement; about 4 hours wall,
   1.5 of compute; ask question 1 first).
   - Write `models/qb_wt_measure/HYPOTHESIS.md` before any build, following
     `models/qb_elos_measure/HYPOTHESIS.md`: its sha256 beside it, the runs, the instrument, the
     windows and the decision rule. The reference is that measurement's `new_file_seed{42,7}`
     (production's settings, code `c637fc0`); the candidate is the same command on a build whose
     `qb_elos.csv` differs only by the win-total file. Governing window: all weeks of 2007-2025 if
     the whole file is replaced, 2024-2025 if only rows are appended; report both. Rule:
     deterministic Brier selects, log loss must not lose; a win recommends the new file (a
     production input change: the user's call, landing between game weeks, with the candidate
     runs as the new floor-sigma reference); anything else keeps the fork as it is.
   - The fork change on a fork branch in its own scratch worktree, never the fork's `main` (which
     the user runs weekly) until the user decides: regenerate `qb_elos.csv` from the commit that
     produced the 2026-10-14 file (`27709b5`) with only the win-total file changed, and check by
     script that every row the change should not touch is identical to
     `models/qb_elos_measure/inputs/new_file.csv`'s source.
   - Build in `~/scratch/qb_measure` (code `c637fc0`, so candidate and reference share code) as
     `models/qb_elos_measure/build.sh` does: the through-2025 cut, its sha256, the pick-time line
     check, the leakage audit. Then the two runs (seeds `42` and `7`, about 31 minutes each).
   - Then a `reviewer` writes `REVIEW.md` (rule 3), and an agent that did not produce the runs
     records it in `.agents/benchmarks.md` and `TODO.md` step 7 (closing the item once reviewed:
     rule 6). The producer writes check-ins only.
3. **The frontend and CI dependencies** (in parallel with the runs; about 1.5-2 hours; scope from
   question 3). One branch, `build/web-deps`: the updates through `pnpm` in `web/` (never
   hand-edit the lock), `DataTable.tsx` moved to react-table 9 with vitest coverage of its sorting
   and rendering first, `setup-uv` 10.3.0, and `dependabot.yml` if approved. Gates: `scripts/gate.sh
   --web` and the `web` job's steps (`pnpm run lint`, `pnpm run typecheck`, `pnpm exec vitest run`,
   `pnpm run build`). A browser look only on a server this session starts itself on a free port,
   without `--reload`. A changelog entry under its own patch version.
4. **The step-6 docs close-out and the roadmap tidy** (in parallel; about 2-3 hours, in chunks):
   - regenerate `RANKINGS_EVAL.md` under `models/step6/rankings/` (question 5);
   - rename "market-free" to "no game lines" where it means the configuration with Elo,
     including `weekly_run/stage1.py:63,68`;
   - the old roadmap labels in user docs: `README.md` ("task 55.8", "decision 12") and
     `config/weekly_run.yaml` ("decision 12"); find them by search, the line numbers moved;
   - refresh the stale "Review:" fields in `.agents/sos_port/decisions/*.md`; commit the
     hypothesis checksums;
   - record the 2026-10-11, 2026-10-12 and 2026-10-14 decisions in `step6_plan.md` and `TODO.md`;
     they and the open P3s from the integration reviews are in an earlier handoff,
     `git show 5b73686:.agents/next_agent_session_prompt.md`;
   - in `TODO.md`: refresh "Current validated baseline" (it still says `0.28.1`); mark step 6's
     phases done; retire the follow-ups the user approved (question 4);
   - build the approvals list and the user's final HTML report for step 6 (send it with
     `SendUserFile` when done).
5. **The Paris roof**: whatever question 7 decided (default: nothing beyond a dated note in
   `pbp_columns.md` entry 13).
6. **Step 8's pick-time checks** if question 6 was yes, and only after items 1-4.
7. **Before stopping**: rewrite this file, run `.venv/bin/pre-commit run --all-files`, and leave
   the user a phone-friendly morning summary: what merged and pushed, the measurement's verdict
   and recommendation, every gate result, and the morning list of decisions.

## Decisions on record (2026-10-15 and 2026-10-16)

- Floor-sigma reference with game lines: `models/qb_elos_measure/new_file_seed{42,7}` (`0.62.0`);
  without game lines: `models/qb_adj_no_lines/no_lines_seed{42,7}` (`0.63.0`, chosen by the user).
- No-game-lines check: market features and anchor off, the transform not counted.
- `train`'s `--no-market-anchor`/`--no-market-transform` and its `--exclude-market` default list:
  accepted.
- The no-game-lines list keeps `qb_adj` out (measured: a tie, `models/qb_adj_no_lines/REVIEW.md`).
- `.agents/m60/INVENTORY.md` and `PROPOSAL.md` stay frozen with their old helper names.
- Finished, independently reviewed roadmap items close without asking (AGENTS.md rule 6);
  reopening, reordering or retiring undone items still asks.
- Earlier on 2026-10-15: the Las Vegas fork fix (done); `qb_adj` as features, off until measured;
  the grouped measurement and its ladder; the backup first-start fork fix; the stale win-total
  file as its own measurement; steps 7 and 8 before the re-tune (step 9); the Week 6 top-decile
  ranking is the two-seed mean.

## Open for the user (not needed tonight)

- The stats-only model: proposed 2026-10-15, not decided. A named track in three parts: in step 8
  measure each feature idea without game lines as well as with them; in step 9 tune it as its own
  study; a weekly shadow prediction, logged and never submitted, scored against the closing line.
- 58.1's exact survivor planner in place of the plan's beam search (acceptance pending), and
  58.2's done-criterion, which waits on the production rebuild of `data/` and a browser look.
- The test suite: the user asked about pruning; recommended no blanket review (4,407 tests run in
  about 55 seconds on 6 workers).
- The reviewers' adversarial tests stay uncommitted in `~/scratch/review_archive/` and
  `~/scratch/sigma_review/`; the one failing case (a Python `bool` read as one round) is
  not fixed.

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
