# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status" and Milestone 60's open question and acceptance
list) and this file.

## State (written 2026-09-25, Milestone 60 with every task done)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.28.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh --web` exits `0` on the final tree
  (the counts are in the `0.28.0` check-in to the user). The working tree is clean after the
  `0.28.0` commits.
- Tasks 60.1-60.9 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); task 58.5 is
  archived under Milestone 58. The milestone itself stays open: closing it is must-ask, and two
  items wait on the user (below). Sign-off decisions: `.agents/m60/PROPOSAL.md`, "Sign-off".
  `.agents/m60/INVENTORY.md` is the signed-off 60.1 record; do not regenerate it over the
  signed-off copy.
- `scripts/` holds only `gate.sh`, which CI runs. Every web job runs `<python> -m nfl_predictor
  <command>`. `README.md` and `AGENTS.md` use the front door; `tests/test_readme_commands.py`
  parses every README command.
- `tests/fixtures/cli_surface.json` held 309 actions in 19 entrypoints when the milestone started
  and holds 221 in 16 parsers now (the `compare` command added 6).
- Checkpoint fingerprints changed in `0.20.0`, `0.22.0` and `0.26.0`; no walk-forward has run
  since, so the next reference run retrains from scratch.

## What this session did (2026-09-25)

- `0.25.0`: one thread option (`--xgb-n-jobs` for stage 1 and the final fit, default every
  core; `--wf-n-jobs` and `wf_n_jobs` removed). The fixture gave identical outputs at 1 and 24
  threads apart from the candidate key's `nj` label and hash.
- `0.26.0`: the market model removed entirely, with `market_only` and `train --tune-scope`.
  A tuned blend now gets the whole `--tune-timeout` (the default `both` had halved it).
- `0.26.1`: the API catch-all fix (task 58.5).
- `0.27.0`: task 60.7. Job templates on the front door, one model-kind vocabulary
  (`nfl_predictor.cli.options.MODEL_KINDS`, `blended_margin_total` accepted as an alias), the
  shims deleted, and the gate's and CI's smoke checks run every command's `--help`.
- `0.27.1`: task 60.8. CI runs `scripts/gate.sh` (with `uv sync --locked`); README and AGENTS.md
  on the front door, with "Reproducing an old run"; the README-command test, which caught a
  leakage-audit example that had always lacked its two required options.
- `0.28.0`: task 60.9, `nfl-predictor compare`. It reproduces all 1,928 numbers of the task 55.8
  independent rescore exactly (`.agents/m60/verify_compare.py`, output beside it).

## Your task

Wait for the user's answers below. Then, with the user's approval: close Milestone 60 (move it to
`ARCHIVE.md`), and merge `feat/m60-cli-consolidation` into `main` (must-ask). The next roadmap
step is step 3 (tasks 55.4, 56.5, 56.6 and the out-of-fold calibration pool) on a new branch off
`main`; see "Roadmap Status" in `TODO.md`.

## How to launch a weekly run today

`.venv/bin/nfl-predictor weekly --run-id <id> --xgb-device cuda` through a `launch.sh` with
`nohup setsid` (see `models/weekly_2026_week_03_full/launch.sh`, which still names the deleted
`scripts/weekly_run.py`; use the command above in its place). It reads
`config/weekly_run.yaml`, which holds the code defaults. Stage 1 on the CPU uses every core by
default; with `--xgb-device cuda` it runs on the GPU. Do not run a weekly run or any walk-forward
unless the user asks.

## Open questions for the user

- Blend power rankings (`TODO.md`, Milestone 60): the rankings read `model.feature_spec`, which
  a blend keeps on `team_model`, so ranking with a blend run has never worked. Read the team
  model's spec, or reject blend runs for rankings with a clear message?
- The milestone acceptance asks for one live check per web job template. Several templates
  rebuild `data/` or run a weekly run or walk-forward (must-ask). Which live checks, and where
  does the web instance run from?
- The characterization-snapshot acceptance item holds except for `0.23.0`'s intended rewrite:
  accept it as met?
- Closing Milestone 60 and merging `feat/m60-cli-consolidation` into `main` (both must-ask).

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
- Editing under `nfl_predictor/ml/` changes every walk-forward checkpoint fingerprint (the
  fingerprint hashes every file there); the command-line code lives in `nfl_predictor/cli/`, and
  `nfl_predictor/reporting/` is outside the fingerprint too.
