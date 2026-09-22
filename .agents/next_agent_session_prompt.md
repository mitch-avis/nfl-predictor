# Next Agent Session Prompt

You are the orchestrating agent for a session in the `nfl-predictor` workspace
(`/home/mitch/workspace/nfl-predictor`). The user has delegated the remaining roadmap to a
sequence of agent sessions like this one. Your job this session is to run **two independent
tracks in parallel**: task 55.8 (season weighting) and Milestone 60 task 60.1 (a read-only CLI
flag audit). Stop for the user whenever a decision is theirs.

**The goal.** The user wants to use this project every week of the regular season to make picks
and bets on that week's games, and to trust what it produces. Every task is judged against that:
does it make the weekly run more correct, more reproducible, more honest about its uncertainty,
or easier to operate.

**Read `AGENTS.md` first, all of it, and treat its "Delegation guardrails" section as binding.**
The short form: `scripts/gate.sh` decides "done"; narrowing is never a checkbox; a number goes
into the docs only after a separate rescore names its run directory (the two-key rule); two
walk-forward runs per task before you ask, unless they are rungs of an accepted ladder (this
session's 55.8 ladder is pre-accepted below, cap 4); the must-ask list means stop and wait.

## Why two tracks, and how to actually run them in parallel

Task 55.8 is walk-forward-bound: **only one walk-forward runs at a time** (`AGENTS.md`, rule 4),
so its four arms run sequentially no matter what. Milestone 60 task 60.1 is a read-only
inventory (grep every `add_argument` across the CLIs, write a table, propose removals) that
touches no code, no `data/`, no `models/`, and needs no gate run of its own — it is explicitly
called out in `TODO.md` as able to "run in parallel with any walk-forward as a subagent task."

So: launch 55.8's first walk-forward arm in the background (`launch.sh` + `nohup setsid`, per
`AGENTS.md`'s walk-forward operating notes), and while it runs, spawn a subagent (the `Agent`
tool, `subagent_type: "Explore"` or `"general-purpose"`, `isolation: "worktree"` to keep it from
touching your checkout mid-run) to produce the 60.1 inventory. Check in on both, land 60.1's
write-up first if the walk-forward is still running, then keep cycling 55.8's remaining arms.
Never run two walk-forwards at once, and never let the 60.1 subagent touch `nfl_predictor/ml/`,
`data/`, or `models/` — it only reads and writes one new `.agents/` file.

## Starting state (2026-09-22)

- **`main` is at version `0.16.2`, commit `c3acc39`.** `scripts/gate.sh` exits `0`
  (`889 passed`, coverage `92.45%`). `feat/m54-0-landing` merged 2026-09-22 and every fully-merged
  stale branch was deleted; only `main` and `feat/web-ui` (the live worktree behind the web API
  on port 8765) remain.
- **A documentation cleanup pass landed the same day**: `AGENTS.md`, `.agents/TODO.md`,
  `.agents/ARCHIVE.md`, `CHANGELOG.md` (structural ordering fixes) and `README.md` were reviewed
  and brought current — stale baseline numbers, a fully-closed Milestone 54 section moved out of
  `TODO.md` into `ARCHIVE.md`, three oversized `[x]` task bodies (55.7, 56.2, 58.4) compacted into
  pointers with their full record moved to `ARCHIVE.md`, and several resolved-but-still-marked-
  open follow-up items corrected. Check `git status --short` first: if those files show modified
  and uncommitted, that is this pass — commit it first (`docs(agents): ...`, conventional
  commit, one commit is fine for a docs-only pass) before starting 55.8 or 60.1. If it is already
  committed, skip this.
- **A material discovery from that same pass, not yet acted on**: the shipped
  `config/weekly_run.yaml` sets `train_recency_half_life_seasons: 4` for final production
  training, but has no `wf_recency_half_life_seasons` key, so its walk-forward comparison stage
  evaluates candidates **unweighted**. Today's weekly run trains production with half-life-4
  recency weighting while never measuring that choice in walk-forward. This is folded into task
  55.8's scope in `TODO.md` (read the task text there in full before planning the ladder) rather
  than treated as a separate default change, since the weighted value is already live.
- **Data**: `data/completed_games_ml.csv` is `edd6b852e910...` (`7292` completed rows, `513`
  columns, `pbp`-default sources). The through-2025 walk-forward input is
  `data/completed_games_ml.m54_flip_through_2025.csv` (`2d4111a6b3b2...`, `7261` rows). This is
  the current build; the tree-budget ladder's six-season checkpoints
  (`models/wf_m55_7_2020_2025_trees*/`) predate both the schedule-skeleton rebuild and the
  pbp-default flip, so they cannot be reused as a season-weighting reference — every 55.8 arm is
  a fresh run.
- **No walk-forward is running.** Machine is idle (load average under 1) as of this writing;
  check `pgrep -af walk_forward` and `uptime` again before launching anything, since time will
  have passed.
- `config/weekly_run.yaml` currently: `tune: false`, `wf_include_postseason: false`,
  `include_postseason: false`, `wf_n_estimators: 200`, `wf_max_depth: 5`,
  `wf_learning_rate: 0.0165` (aligned with production `DEFAULT_XGB_PARAMS` since task 55.7), and
  the `train_recency_half_life_seasons: 4` / missing `wf_recency_half_life_seasons` gap above.
- The Week 2 package is `models/weekly_2026_week_02_refresh/`. The user runs
  `scripts/weekly_run.py` for weekly picks themselves; not this session's job unless asked.
- The web API worktree `../nfl-predictor-web` on port 8765 and `../nfeloqb` / `../nfl-sos-ratings`
  are untouched by either track this session; leave them alone (must-ask to touch).

## Read first, in this order

1. `AGENTS.md`: the delegation guardrails in full (especially rule 4's ladder language and rule
   5's must-ask list), the walk-forward operating notes (launch.sh/nohup, OpenMP policy,
   "one at a time"), and the "Fit-noise floor on the same build" paragraph (measured on
   three-season arms — read the caveat about applying it to six-season arms).
2. `.agents/TODO.md`: task 55.8 in full (the recency-mismatch finding is folded into its text),
   Milestone 60 task 60.1 in full, and the "Roadmap Status" section for how these two fit the
   rest of the order.
3. `config/weekly_run.yaml`: confirm the `train_recency_half_life_seasons` / missing
   `wf_recency_half_life_seasons` gap described above still holds before planning the ladder.
4. `README.md`, "Backtesting": the recency ablation table, marked superseded — this is what 55.8
   replaces.
5. For 60.1: `nfl_predictor/ml/ml_model_cli.py`, `scripts/weekly_run.py`,
   `scripts/walk_forward_backtest.py`, `scripts/betting_pipeline.py`,
   `scripts/golden_command.py`, `nfl_predictor/data_collection.py`, `scripts/validate_*.py`,
   `scripts/power_rankings.py` — the task text in `TODO.md` names exactly what to inventory per
   flag (inert, duplicated across entrypoints, config-only).

## First check-in (before any code or run)

1. `git status --short`, `git log --oneline -3`, `scripts/gate.sh --quick`. Handle the docs
   commit from "Starting state" if still pending; ask about nothing else until the tree is clean
   and the quick gate passes.
2. `pgrep -af walk_forward`, `uptime`, `ps -eo pcpu,args --sort=-pcpu | head -4`.
3. Report to the user, in one message: tree state, confirmation of the two-track plan below, and
   the 55.8 ladder as written (four six-season arms, cap 4) so it counts as accepted under
   guardrail rule 4 before you launch the first one.

## Track A: task 55.8 (season weighting)

1. Resolve the recency mismatch in scope (see "Starting state"): read the full task text in
   `TODO.md` and decide the exact six-season config for every arm — seasons `2020-2025`, from
   week 1, `auto` calibration, `market_anchor` on, `4` calibration weeks, a fixed seed (`42`,
   matching the tree-budget ladder), on the current build's through-2025 cut
   (`data/completed_games_ml.m54_flip_through_2025.csv`).
2. Write one `HYPOTHESIS.md` (in the first arm's run directory, e.g.
   `models/wf_m55_8_2020_2025_unweighted/`) declaring the **ladder**: four arms — unweighted
   reference, `--recency-half-life-seasons 4`, `8`, `16` — cap 4, no fifth rung without asking.
   Name the governing window (weeks 3-18) and the exact columns (deterministic Brier, margin
   MAE, the paired deterministic-minus-market interval), and state up front that the
   three-season fit-noise floor is an approximate guide here, not a measured six-season
   threshold.
3. Launch the unweighted reference arm first (needed regardless of outcome, since no current
   six-season reference exists on this build), then the three weighted arms, one at a time,
   each via its own `launch.sh` + `nohup setsid`. About 100 minutes idle per arm assuming the
   machine stays quiet; check load before each launch and use `OMP_WAIT_POLICY=PASSIVE` if
   anything else (including the 60.1 subagent, which is cheap, or the web API's `--reload`
   watcher) is loading the machine.
4. After all four: an independent reviewer (a separate subagent or your own rescore from the
   checkpoints, following the two-key rule) rescores every arm from disk, writes `REVIEW.md`
   beside each `HYPOTHESIS.md`, and only then may numbers move into `AGENTS.md` / `TODO.md`.
5. Decision: if `4` (the value already shipped in `train_recency_half_life_seasons`) is at least
   as good as unweighted and the other two, no default change is needed — just align
   `wf_recency_half_life_seasons: 4` into `config/weekly_run.yaml` so the walk-forward stage
   finally measures what production trains, and replace the README's superseded ablation table
   with the new one. If a different half-life (or unweighted) wins beyond the noise floor,
   report it and ask before changing `train_recency_half_life_seasons`'s shipped value — that is
   a default change under guardrail rule 5.
6. Versioned chunk, changelog entry, `pyproject.toml` bump, `uv lock && uv sync`, gate green,
   commit on a branch named for the task (for example `feat/m55-8-season-weighting`), before
   reporting done. Merging to `main` stays must-ask.

## Track B: Milestone 60 task 60.1 (CLI flag audit)

1. Spawn a subagent (worktree isolation recommended) with the full task text from `TODO.md`
   copied into its prompt — it starts cold and needs the context, including which entrypoints to
   cover and what to record per flag (inert since when, duplicated where, config-file-only).
2. It is read-only research: no production code changes, no gate run required for the audit
   itself. Its deliverable is a written inventory file under `.agents/` (name it something like
   `.agents/m60_cli_flag_audit.md`) with a proposed removal list.
3. Once it lands, this session reviews it, folds a summary into `TODO.md` under Milestone 60,
   and stops: the acceptance criterion is explicit that removals wait for the user's sign-off on
   the list, so do not act on the proposed removals this session.
4. If a commit is wanted for the new audit file, do it on its own branch (for example
   `docs/m60-cli-audit`) separate from 55.8's branch, since the two tracks are unrelated.

## Documentation debts to clear as you go

- `README.md`, "Backtesting": replace the superseded recency ablation table with 55.8's result.
- `.agents/TODO.md`, "Current validated baseline": restate at whichever version lands last this
  session (55.8's chunk, most likely).
- `config/weekly_run.yaml`: align `wf_recency_half_life_seasons` once 55.8 resolves.
- `CHANGELOG.md`: one entry per landed chunk, new version each time, never `[Unreleased]`.
- Rewrite this file at the end of the session (or at a natural pause) with whichever track
  finished, whichever is mid-flight, and the next task.

## Open questions waiting on the user

1. The recency-mismatch discovery itself: confirm the plan above (fold it into 55.8, no separate
   ask) is the right call, or say otherwise before the ladder launches.
2. Any default change coming out of 55.8 (if a half-life other than the shipped `4` wins): must-
   ask with the numbers in hand, per guardrail rule 5.
3. A fifth walk-forward rung beyond the four-arm 55.8 ladder, or any six-season run outside it:
   fresh ask.
4. Milestone 60's removal list, once 60.1 lands: the user signs off before any removal.
5. Merging either branch to `main`, and pushing: must-ask, every time, however small.
6. Standing must-asks from `AGENTS.md` rule 5 otherwise unaffected this session: rebuilding
   `data/`, touching `../nfeloqb` / `../nfl-sos-ratings` / the web API on port 8765, reopening
   Milestone 57, reordering the roadmap.

## How each chunk runs

- TDD: characterization or failing tests first for any production code touched (55.8 should not
  need any; 60.1 touches none). No new `noqa`, `type: ignore` or `pragma: no cover` without a
  reason in the code.
- One versioned changelog entry per landed chunk, `pyproject.toml` to the same version, `uv
  lock`, `uv sync`. Never a tag.
- `scripts/gate.sh` exits `0` on the final tree before a chunk is reported done; markdownlint
  covers `.md` notes under `models/` too, so keep `HYPOTHESIS.md` and `REVIEW.md` clean.
- Walk-forward runs: `HYPOTHESIS.md` with the decision rule and exact command written before
  launch; a `launch.sh`; one run at a time; an independent rescore before any number reaches the
  docs; a check-in after every run naming the directory, the numbers, the floor comparison and
  the decision.
- Subagents for read-only parallel work only (60.1 fits exactly); never over anything on the
  must-ask list.

## Final report for a session

1. What landed on each track, by version, with commit subjects.
2. Every walk-forward run started under 55.8: directory, hypothesis, result on the deterministic
   and market columns against the fit-noise floor, and the decision it produced.
3. The 60.1 inventory: where it landed, its headline findings.
4. Anything in this file or in `AGENTS.md` that turned out to be wrong.
5. The state of the tree (branch(es), version(s), uncommitted files) and the next task, also
   written into this file.
6. The questions waiting on the user.
