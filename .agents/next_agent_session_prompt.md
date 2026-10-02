# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones"), `.agents/benchmarks.md` before any walk-forward, and this file.

## State (written 2026-10-02 03:40, step 4 Phase B running)

- Roadmap step 4 runs on `feat/step4-feature-values` (local only, not pushed), checked out in the
  main checkout at `0.42.1`; `uv sync` has run and the tree is clean. `scripts/gate.sh` exits `0`
  on `091d7c0` (1472 passed, coverage 93.25%). `main` is at `0.39.3` (merged 2026-10-01, pushed
  2026-10-02, both approved by the user). No agent worktrees remain except the gate worktree
  `~/scratch/gate_step4`.
- On `main` (reviewed, each with a `CHANGELOG.md` entry; details in `ARCHIVE.md`, "Roadmap step
  4 (in progress) - Resolved follow-ups"): `0.37.2`-`0.38.2` rebuild reproducibility, `0.39.0` the
  web weekly job over `config/weekly_run.yaml`, `0.39.1`-`0.39.2` the dependency refresh and
  `update_requirements.sh` repairs, `0.39.3` the `next_opponent_identity` group. Since `0.37.3` a
  step-4 build compares against a reference rebuilt with current code, not against today's
  `data/`.
- Only on the branch, Phase A of `.agents/step4_plan.md` (done, every chunk reviewed): `0.40.0` the
  `qb_def_adj` group (on by default once built; production has no group switch, so it must not
  reach `main` unless adopted); `0.40.1` `--strength-prior-blend-games`; `0.41.0` the QB defense
  term in per-dropback units; `0.42.0` `--line-source {stored,pick_time}`; `0.42.1`
  `nfl-predictor compare --market-from <run>` (scores every compared run against another run's
  market by `game_id`, warns on result mismatches; two review rounds, the two optional P3s left:
  the "any compared run" mismatch case is untested, and both-NaN results would count as a
  mismatch, which walk-forward checkpoints cannot hold).
- Phase B was launched on 2026-10-02 at about 03:45 as one detached job,
  `models/step4/phase_b.sh <commit>` (log `models/step4/phase_b.log`, PID in
  `models/step4/phase_b.pid`, commit in `models/step4/launch_record.txt`): `build.sh` makes the
  five cold full-history builds R0, S2, S8, T2, PT in `~/scratch/step4_builds/<arm>/` (logs
  there), cuts each to `models/step4/inputs/<arm>.csv` (+ `.sha256`) and runs the leakage audit
  (`models/step4/audit/<arm>.json`); it stops if a build fails or an audit is not ok. Then
  `driver.sh` runs the 14 walk-forwards (`models/step4/<arm>_seed{42,7}/`, order R0 QB NO S2 S8
  T2 PT, seed 42 then 7 per arm), checking before each run that `src/`, `pyproject.toml` and
  `uv.lock` are unchanged from the commit, the GPU answers, and nothing else runs. Expected end
  about 12:30. Rules: `models/step4/HYPOTHESIS.md` (restates the plan's decision rules).
  **Do not edit `src/`, `pyproject.toml` or `uv.lock`, and do not commit, while it runs** (the
  driver stops on a code change; a commit changes the recorded git commit between arms).

## Open questions for the user

None pending.

Decided 2026-10-02: the 56.6 map stays the per-season expanding fit; the `compare` market-from
option (landed as `0.42.1`); push `main` (done); the old scratch `data/` copies deleted (venvs and
logs kept). Decided 2026-10-01: the plan, the per-dropback QB units, the partial merge to `main`,
no incremental option on `etl_full`; earlier decisions are in `ARCHIVE.md`. The user handles
weekly runs and picks; do not raise Week 5 for several days.

## Next

1. Watch Phase B (`tail models/step4/phase_b.log`; `kill -0 $(cat models/step4/phase_b.pid)`).
   If it stopped, read the log; the driver can be relaunched with the same commit and resumes
   from the fold checkpoints. After it ends: confirm every fold ran on `cuda`, then an
   independent `reviewer` rescores from the fold checkpoints and writes
   `models/step4/REVIEW.md` (rule 3), with `nfl-predictor compare` per arm against R0 (two seeds)
   and, for PT, a second compare with `--market-from` R0. Then the check-in with each arm's
   governing numbers, every interval that excludes zero, and the ranks (rule 9).
2. After Phase B: the user's adoption decisions, then `F` (the adopted build, two seeds; the new
   GPU reference and the floor's sigma pool), one backed-up rebuild into `data/` (must-ask), and
   before any merge to `main` either adopt `qb_def_adj` or turn it off/remove it.
3. Weekly runs are the user's. If asked to run one, run it from `main` (the step-4 branch adds the
   unmeasured `qb_def_adj` columns to production), never while Phase B runs; after switching
   branches run `uv sync` with the LightGBM CUDA flags
   (`uv sync $(.venv/bin/nfl-lightgbm-cuda-install uv-args)`).

## How the last sessions worked (keep doing this)

- Each code chunk: an `implementer` subagent in its own worktree, a separate `reviewer`, fixes
  back to the implementer, and a `--no-ff` merge into the feature branch (the commit hook needs a
  Conventional Commits subject on merge commits too). The delegating session writes:
  - `CHANGELOG.md` and the version (`pyproject.toml`, `uv lock`, `uv sync`);
  - `TODO.md`, `ARCHIVE.md` (step-4 resolved items under "Roadmap step 4 (in progress)"),
    `AGENTS.md`, `.agents/benchmarks.md` and this file.

  It lints the Markdown it edits with `markdownlint-cli2` before committing.
- Small review-fix deltas (docs, a flag) may be verified by the delegating session from the diff;
  anything with logic goes back to the reviewer (a second-round reviewer caught a `total_yards`
  regression in `0.37.2`).
- Every measured number goes through an independent reviewer's rescore before it enters the docs
  (rule 3). Decision rules are written into the run directory before scoring.
- Run the gate in a separate worktree (`git worktree add --detach ~/scratch/gate_step4 HEAD`,
  `uv sync`, `VIRTUAL_ENV=<it>/.venv scripts/gate.sh`). Watch long runs by polling a PID, never
  with `pgrep -f` on a pattern that matches the watcher's own command line.
- Lint work: fix the code rather than suppress. Run `.venv/bin/pre-commit run --all-files`
  yourself before ending a turn (the Stop hook runs it).
- Stop at a natural point well before the context fills, and rewrite this file.

## Notes

- Never run two walk-forwards at once. Check `uptime`, `nvidia-smi` and
  `pgrep -af "nfl-predictor (backtest|weekly)"` first.
- A backtest that judges probabilities must run from 2007 week 1 or pass
  `--floor-sigma-reference-runs` with the GPU reference, and must pass `--wf-start-week 1`
  (rule 11).
- Re-run a plain `uv sync` after every version bump and after every merge or checkout that
  changes the version or the layout.
- The user runs `nfl-predictor web` from this checkout. Never stop or restart it yourself
  (rule 5).
