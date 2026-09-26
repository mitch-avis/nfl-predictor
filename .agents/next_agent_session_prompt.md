# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-15) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestones 55 and 56, and "Open follow-ups from
completed milestones") and this file.

## State (written 2026-09-25, after the `AGENTS.md` split)

- `main` is at version `0.28.3` and pushed to `origin` with the user's approval: Milestone 60
  (`0.28.1`), the pre-commit hooks (`0.28.2`) and the documentation split (`0.28.3`), each merged
  with `--no-ff`. Branch `docs/rule-2-corrections` carries `0.28.4` and `0.28.5` (below),
  committed and not yet merged: merging and pushing stay must-ask.
- `scripts/gate.sh --web` exits `0` on the `0.28.3` tree: 1032 passed, coverage 92.21% against
  the 90% floor, 22 frontend tests.
- `git commit` runs pre-commit hooks: file hygiene, ruff on staged Python, and a Conventional
  Commits check on the message (a merge needs a conventional message too, since `--strict` bars
  git's default "Merge branch" text). `git push` runs `scripts/gate.sh --quick`.
- `AGENTS.md` holds the rules; detail loads on demand. Read `.agents/benchmarks.md` before any
  walk-forward comparison, `.agents/modeling_spec.md` before changing prediction, evaluation or
  feature code, and `.agents/walk_forward_runbook.md` before launching a long run. Nested
  `AGENTS.md` files under `nfl_predictor/api/`, `nfl_predictor/ml/` and `web/` carry the rules
  for those directories.
- Milestone 60 is archived (`ARCHIVE.md`, "Milestone 60", with a close-out listing every
  acceptance item). One acceptance item was narrowed and two leftovers moved to `TODO.md`, "From
  Milestone 60", both assigned to step 3.
- One front door: `nfl-predictor <command>` runs every task (`--help` lists them). `scripts/`
  holds only `gate.sh`, which CI runs. `nfl-predictor compare` is the paired walk-forward
  comparison, and under rule 3(b) a reviewer may use it for rescoring (it reproduces the task 55.8
  independent rescore exactly), as long as the review also checks provenance and the reviewer did
  not produce the run.
- Checkpoint fingerprints changed in `0.20.0`, `0.22.0` and `0.26.0`, and no walk-forward has
  run since, so the next reference run retrains from scratch. The walk-forward input is still
  `data/completed_games_ml.m54_flip_through_2025.csv` (`2d4111a6...`).
- The user runs the web app from this checkout on port 8000 (`nfl-predictor web --reload`, last
  seen 2026-09-25). With `--reload`, every Python edit, checkout or merge you make restarts it, and
  a restart marks running web jobs failed. Before editing code, check `pgrep -af "nfl-predictor
  web"`. If a job is running from it, stop and ask. Never stop or restart the server yourself
  (rule 5).

## What the last session did (2026-09-25)

- `0.28.2`: the user's pre-commit configuration, with its commit-message hook fixed (the global
  exclude filtered out `.git/COMMIT_EDITMSG`).
- `0.28.3`: split `AGENTS.md` (80k to 38k characters) into the files named above, dropped the
  `.agents/skills` excludes (agent skills are installed globally), and pointed the living docs
  at the current modules and commands.
- `0.28.4`: rewrote Milestone 55's acceptance line for the 55.9 tune and the 56.3 shared source
  (the user's request), and gave rule 2 a correction clause: small factual errors noticed
  outside the current task may be fixed without asking, within its limits, and are reported
  under "Fixed without asking" in the next check-in.
- `0.28.5`: rule 15 (a non-obvious question to the user carries the agent's brief,
  code-grounded recommendation), Milestone 55's goal rewritten for the hypothesis-driven
  settings and the 55.9 tune, and the stale "not pushed" baseline line in `AGENTS.md`
  corrected under rule 2. `AGENTS.md` is at about 39.5k characters, near the 40k size at which
  Claude Code warns: trim or move detail out before adding to it.
- `0.25.0`-`0.28.0`: finished Milestone 60 (one thread option, the market model removed, the
  web API catch-all fix, web jobs on the front door, CI on the gate, the `compare` command).
- `0.28.1`: closed the milestone with the user's approval. It verified that every removed option
  has a changelog note (`.agents/m60/verify_removals.py`), deleted three spent `.agents/` docs, and
  documented when to use `--reload` and the Vite dev server (`web/README.md`, "Which mode to
  use"). It also documented what drives the default power rankings (`README.md`). Then it
  merged into `main`.
- Answered the user's questions on the Week 3 outputs and filed three follow-ups under "From the
  2026-09-25 review of the Week 3 outputs" in `TODO.md` (evidence in
  `.agents/findings_2026_09_25/`, with no second key yet):
  - The Model page's feature importance sums XGBoost's average gain per split over one-hot
    columns, which puts `*_next_opponent_abbr` and `stadium_surface` on top; by total gain they
    rank near the bottom. Step 3.
  - The next-opponent identity columns are a candidate to drop. Step 4.
  - The early-season strength snapshot (the default power rankings and the `adj_*` features)
    is dominated by last season, partly because the in-season solve is shrunk twice. Step 4,
    with task 55.3.

## Your task: roadmap step 3 (production/benchmark parity)

Create a new branch off `main` (for example `feat/step3-parity`). Scope, from "Roadmap Status"
and Milestones 55/56 in `TODO.md`:

1. Task 55.4: the GPU as the default device for every XGBoost run. First a one-fold CPU-vs-GPU
   timing and prediction-difference check, then a GPU determinism check (the same seed twice must
   give identical predictions). Add a CPU fallback and record the device in metadata. Then a new
   GPU reference arm on **two seeds** (rule 13); rule 4 needs a written hypothesis and decision
   rule first.
2. Task 56.5 with 56.7(b)/(c): how production probabilities are formed, one calibration path for
   every run type, and whether the weekly stage-1 re-selection is retired. Start by rescoring
   saved predictions. `models/wf_m55_8_review/probability_paths.py` has no second key and must
   be rescored independently before any of its numbers enter the docs. The choice of default is
   the user's (must-ask).
3. Task 56.6: the pick-time line yardstick (opening lines from `data/nfl_lines.csv` and
   `data/historic_odds.csv`, as the task describes).
4. The out-of-fold calibration pool (narrowed 59.2), the four calibration-window follow-ups, task
   55.6 (the per-season stability view in the standard report), the feature-importance fix, and
   the Milestone 60 leftovers: the per-template live web checks, and the `blend` power rankings
   (settle them in 56.5).

Batch every edit under `nfl_predictor/ml/` so they share one new reference (rule 14). Measure
success as "Roadmap Status" describes: Brier of the probabilities actually submitted against the
market's Brier, no loss on log loss, then pool points and margin MAE as tie-breakers.

Start by proposing to the user the order of the step-3 work, the first measurement's hypothesis
and decision rule, and the run budget (a ladder with a cap, rule 4). Do not launch anything
before they accept it. Production-changing steps land between game weeks (the user makes picks
before each Thursday game).

## How to launch a weekly run today

Run `.venv/bin/nfl-predictor weekly --run-id <id> --xgb-device cuda` through a `launch.sh` in the
run directory, started with `nohup setsid`. The run reads `config/weekly_run.yaml`, which holds
the code defaults. Do not run a weekly run or any
walk-forward unless the user asks.

## Open questions for the user

- None open from the last session.

## Notes

- Never run two walk-forwards at once; check `uptime` and `pgrep -af "nfl-predictor
  (backtest|weekly|sweep)"` first.
- Editing a `.py` file under `nfl_predictor/ml/` (or `constants.py` or `ml_model.py`) changes
  every walk-forward checkpoint fingerprint; the fingerprint hashes those source files. The
  command-line code in `nfl_predictor/cli/` and `nfl_predictor/reporting/` sits outside it.
- Re-run a plain `uv sync` after every version bump and after every merge that bumps the version.
