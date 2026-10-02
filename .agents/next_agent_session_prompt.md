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
- Phase B finished 2026-10-02 11:58: 5 builds, 14 runs, 0 failures (`models/step4/`, driver log
  `phase_b.log`). An independent reviewer rescored it (`models/step4/REVIEW.md`, provenance clean,
  `compare` matched its own rescore exactly); the numbers are in `.agents/benchmarks.md`, "Step 4,
  Phase B feature-value arms". Under the written rules QB, NO, S2, S8 and T2 tie R0; PT is worse on
  every column (expected) but not worse than its own pick-time market. The handoff commit
  `c4dd24a` is the run commit; code may change again now.

## Open questions for the user

Asked 2026-10-02 after Phase B (recommendations in the check-in): (1) QB tie, drop `qb_def_adj`
from the default features (the rule's simpler setting; 53.7's joint ridge is the next step); (2)
NO tie, drop the `*_next_opponent_abbr` pair; (3) S2/S8/T2 ties, keep `K = 4` for both blends (the
rule gives no recommendation; one shared value is fewer knobs); (4) PT, adopt the pick-time line
for parity (a default change; `nfl-predictor lines` needs the line source first). Then `F`.

Decided 2026-10-02: the 56.6 map stays the per-season expanding fit; the `compare` market-from
option (landed as `0.42.1`); push `main` (done); the old scratch `data/` copies deleted (venvs and
logs kept). Decided 2026-10-01: the plan, the per-dropback QB units, the partial merge to `main`,
no incremental option on `etl_full`; earlier decisions are in `ARCHIVE.md`. The user handles
weekly runs and picks; do not raise Week 5 for several days.

## Next

1. Wait for the user's answers to the four questions above; nothing runs until then.
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
