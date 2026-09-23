# Next Agent Session Prompt

You are resuming work in the `nfl-predictor` workspace (`/home/mitch/workspace/nfl-predictor`).

Read `AGENTS.md` first and treat its delegation guardrails as binding.
Then read `.agents/TODO.md` and this file.

## Branch and version state

- Branch: `feat/m55-8-season-weighting`
- Version on this branch: `0.17.0`
- Status of task 55.8: closed and gated on this branch
- `scripts/gate.sh` passed on the `0.17.0` tree (`889 passed`, coverage `92.45%`)

## What landed in task 55.8

The four-arm six-season season-weighting ladder is complete and reviewed.

Run directories:

- `models/wf_m55_8_2020_2025_unweighted/`
- `models/wf_m55_8_2020_2025_half_life4/`
- `models/wf_m55_8_2020_2025_half_life8/`
- `models/wf_m55_8_2020_2025_half_life16/`

Checkpoint directories:

- unweighted: `models/wf_checkpoints/a75322959a223e38576c/`
- half-life 4: `models/wf_checkpoints/41cb4904e5e1a8878fd1/`
- half-life 8: `models/wf_checkpoints/2b9b6a6cb83107f66dfa/`
- half-life 16: `models/wf_checkpoints/d51959f8b4a8b33f5dbf/`

Review files exist beside each run directory.

Decision:

- Half-life `16` was the best raw Brier arm.
- But every governing-window paired interval against both the unweighted reference and the shipped
  half-life `4` arm still covered zero.
- The ladder therefore reads as flat within noise, not as a default-changing win.
- `train_recency_half_life_seasons: 4` stays in place.
- `config/weekly_run.yaml` now also sets `wf_recency_half_life_seasons: 4`, so Stage 1 measures
  the same season weighting Stage 2 already trains.

Docs/config updated for this outcome:

- `config/weekly_run.yaml`
- `README.md`
- `.agents/TODO.md`
- `.agents/ARCHIVE.md`
- `AGENTS.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `uv.lock`

## Milestone 60 status

The read-only task 60.1 audit is landed locally at:

- `.agents/m60_cli_flag_audit.md`

TODO references it, but Milestone 60 is still open because removals require the user's sign-off.

Headline findings already recorded there:

- `scripts/weekly_run.py` still exposes an inert walk-forward early-stopping flag
- `scripts/weekly_run.py` and `nfl_predictor/ml/ml_model_cli.py` expose tuning-only
  early-stopping flags whose names now overstate their effect
- Calibration, recency, market-probability blending, XGBoost runtime, and power-rankings options
  are duplicated with naming drift across entrypoints

## Next work to pick up

1. Task 55.9: Optuna re-tune, when the user wants to spend the machine time
2. Milestone 60: get user sign-off on the audit's removal list, then land the CLI cleanup
3. Milestone 53 task 53.7 after the above higher-priority work

## Must-ask items still in force

1. Merging this branch to `main` or pushing it
2. Any default change beyond the closed 55.8 alignment that is already reflected here
3. Any Milestone 60 flag removals before the user signs off on the audit list
4. Any rebuild under `data/`, or any touch to `../nfeloqb`, `../nfl-sos-ratings`, or the web API
   on port `8765`

## Working tree expectations

If this prompt is being read before the local commit is made, expect the 55.8 close-out docs and
the 60.1 audit file to still be unstaged or uncommitted. If the local commit has already been
made, the tree should be clean except for future user work.
