# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).
Read `AGENTS.md` first and treat its delegation guardrails (rules 1-14) as binding, then
`.agents/TODO.md` (above all "Roadmap Status", Milestone 60 tasks 60.8-60.9 and its open
question and acceptance items, and task 56.7) and this file.

## State (written 2026-09-25, Milestone 60 after closing 60.7)

- Branch `feat/m60-cli-consolidation`, cut from `main` at `703ea25`; version `0.27.0`. Not
  pushed; merging and pushing are must-ask. `scripts/gate.sh --web` exits `0` on the final tree
  (the counts are in the `0.27.0` check-in). The working tree is clean after the `0.27.0`
  commits.
- Tasks 60.1-60.7 are done and archived (`ARCHIVE.md`, "Milestone 60 (partial)"); task 58.5 is
  archived under Milestone 58. Sign-off decisions: `.agents/m60/PROPOSAL.md`, "Sign-off".
  `.agents/m60/INVENTORY.md` is the signed-off 60.1 record; do not regenerate it over the
  signed-off copy.
- `scripts/` holds only `gate.sh`. Every web job runs `<python> -m nfl_predictor <command>`.
  `README.md` and `AGENTS.md` still name the deleted `scripts/*.py` paths (31 places when
  counted by grep): that is task 60.8, next.
- Characterization snapshots must not change during a move; an intended output change rewrites
  them with `NFLP_UPDATE_SNAPSHOTS=1` in the same commit and says so in the changelog.
  `tests/fixtures/cli_surface.json` changes with every flag change (intended; audit the diff).
  It held 309 actions when the milestone started and holds 215 in 15 parsers now.
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

## Your task

1. Task 60.8: CI (`.github/workflows/validation.yml`) calls `scripts/gate.sh` instead of
   repeating its steps (its CLI smoke step already mirrors the gate's); `README.md` (the Scripts
   section, every command example, and a "Reproducing an old run" note: `git worktree add
   <commit>`, then the launcher as written); `AGENTS.md` ("Repo scripts", "Project Shape", "Dev
   Workflows", launch instructions; the `launch.sh` guidance should name `nfl-predictor`);
   `CHANGELOG.md`; `ARCHIVE.md`; this file. Find every stale path by grep, not by memory
   (rule 10).
2. Task 60.9: the `compare` command; it must reproduce an existing `REVIEW.md` rescore exactly
   before it is trusted.
3. Each chunk small, versioned, gated (`--web` when the web code changes) and committed; rewrite
   this file at every landed chunk (rule 8).

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
- Merging `feat/m60-cli-consolidation` into `main` is must-ask once Milestone 60 closes (or
  earlier, if the user wants the chunks on `main` between game weeks).

## Notes

- `models/wf_m55_8_review/probability_paths.py` has no second key; none of its numbers go into
  the docs until task 56.5 rescores it independently.
- Never run two walk-forwards at once; check `uptime` and `pgrep -af walk_forward` first.
- Editing under `nfl_predictor/ml/` changes every walk-forward checkpoint fingerprint (the
  fingerprint hashes every file there); the command-line code now lives in `nfl_predictor/cli/`.
