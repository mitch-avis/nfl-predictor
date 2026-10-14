# Delegation guardrails: the long form

The binding rules are listed in `AGENTS.md` ("Delegation guardrails"); this file holds their full
definitions, limits and examples, and `.agents/guardrail_history.md` records the incidents behind
each one. The rules were added on 2026-09-19 after the Milestone 59 audit and amended on
2026-09-21, 2026-09-23, 2026-09-24 and 2026-09-25. An autonomous session has no other supervision,
so these rules are not advisory.

1. **One gate.** `scripts/gate.sh` is the definition of "checks pass". No task, chunk or version
   is reported done, and no changelog entry is written as landed, until it exits `0` on the
   final tree. Reporting a subset of checks as the gate is a defect; `--quick` is for iteration,
   never for the report.
2. **Narrowing is never a checkbox.** If what landed differs from the task text (smaller scope, a
   substitute method, a skipped acceptance criterion), the task stays `[ ]` with a `Narrowed:`
   note giving the difference, the reason and where the remainder now lives, and the difference
   goes on the user's question list. Rewriting the acceptance text to fit the delivery is not
   allowed. **Correcting is not narrowing:** an error noticed outside the current task may be
   fixed without asking first when all of these hold; if any fails, it is a question.
   - It is a fact, not a choice: a stale path, command or option name, a broken pointer, a
     typo, or a bug whose correct behavior the repo already states (a docstring, test,
     documented contract or error message), with exactly one plausible fix.
   - Meaning is unchanged. Corrected task or acceptance text names the current equivalent of
     what was asked, keeps the same bar and every criterion, and no checkbox changes state. A
     code fix changes no output of a weekly run, walk-forward or ETL, and touches no
     fingerprinted file (`src/nfl_predictor/ml/*.py`, `constants.py`, `ml_model.py`).
   - Nothing under rule 5 is involved, no measured number changes (rule 3), and records stay as
     written: `.agents/ARCHIVE.md`, past `CHANGELOG.md` entries, run directories, benchmark
     provenance.
   - It is small and verified: one logical change of a few lines, a failing test first for
     code, `scripts/gate.sh` green.
   - The user is told: the next check-in lists it under "Fixed without asking" (before, after,
     why), and a code or tooling fix gets a changelog line.
3. **Two keys on every number.** The agent that produced a run never writes its numbers into
   `AGENTS.md`, `.agents/benchmarks.md`, `.agents/ARCHIVE.md`, `.agents/TODO.md` or
   `CHANGELOG.md`. A reviewer (a separate subagent, or the next session) rescores the artifact
   from disk, writes the run directory and the reproduction command beside the number, and only
   then may the docs change. A number in the docs without a run directory is a defect. The review
   is independent only if (a) its `REVIEW.md` names who reviewed and states that the reviewer did
   not produce the run; (b) it recomputes the metrics from the fold checkpoints
   (`models/wf_checkpoints/<fingerprint>/`), not from `metrics_report.json` or only through the
   producer's comparison script; and (c) it checks provenance: dataset hash, git commit, fold
   count, `best_iteration`/`early_stopped`, and that only the intended setting differs between
   arms. If a reviewer subagent fails or returns without numbers, the producing agent reports
   that and stops; it never writes the review itself. `nfl-predictor compare` reproduces the
   task 55.8 independent rescore exactly (`.agents/m60/verify_compare.py`), so a reviewer can use
   it under (b) as long as the review also checks provenance and the reviewer did not produce the
   run.
4. **Compute budget.** Before each walk-forward run, write the hypothesis and the decision rule
   (which result changes what). The decision rule names the exact window or windows it reads
   (week 1, week 2, weeks 3-18, all weeks), the exact columns (deterministic Brier, log loss,
   pick accuracy, margin MAE, the paired deterministic-minus-market interval), and how the
   windows combine when they disagree, so a rung cannot read as a tie on one window and a win
   on another without the rule saying which one governs. After two runs on one task without a
   decision, stop and ask. A **ladder** (several runs of one hypothesis family, with a stated
   cap on the number of rungs and a stopping rule) written into the check-in and accepted by
   the user counts as approval for every rung up to that cap; only a rung beyond the cap, or
   outside the ladder as written, needs a fresh ask. One walk-forward at a time; `uptime` and
   `pgrep -af "nfl-predictor (backtest|weekly)"` first; OpenMP policy by load. Any edit under
   `src/nfl_predictor/ml/` changes every checkpoint fingerprint, so a rerun after a code change
   retrains from scratch; plan runs after the code is stable.
5. **Must ask first** (stop and wait; never assume):
   - rebuilding anything under `data/` (an ETL rerun), or deleting or overwriting any file under
     `data/` or `models/`;
   - changing a default (CLI, config, constants) that alters what a weekly run produces, or any
     change that alters feature values at ETL time;
   - closing a milestone, reopening a parked one (Milestone 57), or reordering the roadmap;
   - merging to `main` or pushing (tags and releases: never);
   - touching `../nfeloqb`, `../nfl-sos-ratings`, or the running web API (`nfl-predictor web`,
     port 8000 by default);
   - a third walk-forward run on one task, or any six-season run, unless it is a rung of a
     ladder the user has already accepted under rule 4 and is within that ladder's cap;
   - anything the task text says to decide with the user.
6. **May proceed without asking:**
   - commits on the working feature branch after each versioned chunk (Conventional Commits, one
     logical change per commit);
   - fixes with a failing test first within the current task, corrections under rule 2, and doc
     updates that restate numbers already verified under rule 3;
   - one walk-forward run per written hypothesis, within rule 4;
   - creating the milestone's feature branch off `main` when none exists.
7. **Check-ins.** Report at every landed version and after every walk-forward run: what landed,
   the run directory, the gate result, the open questions. Stop for a question whenever rule 5
   triggers; a session that ends blocked on a question has done the right thing. The check-in
   after a walk-forward run carries its governing-window numbers (not only "it finished"), and it
   is written before the next rung is launched. A session that launches long runs says in the
   same check-in how they will be sequenced and supervised: either one driver script that runs
   the accepted rungs back to back, or a watcher that wakes the session when the run exits.
   Never end a turn while promising to "keep monitoring" with nothing actually watching.
8. **Handoff hygiene.** Rewrite `.agents/next_agent_session_prompt.md` at every landed chunk
   (branch, version, uncommitted state, the next task, open questions) so a restart after a
   closed terminal or a context summary resumes without re-deriving anything.
9. **Read every interval; the incumbent gets no benefit of the doubt.** A decision or close-out
   lists every paired interval that excludes zero, in every window reported and in either
   direction, plus where each arm ranks on every named column and tie-breaker (deterministic
   Brier, log loss, margin MAE, total MAE, pick accuracy, confidence-pool points). "No arm beat
   X" is incomplete without how X compares to the others. A shipped setting that was never
   measured is an assumption, not a baseline: when a measurement ties, the simpler setting (fewer
   knobs, fewer assumptions) is the recommendation, and the tie goes to the user as a question
   rather than closing in the incumbent's favor.
10. **Inventories and audits are generated, not written.** Any inventory of code (flags, files,
    columns, consumers, call sites) is produced by a script checked in beside it, and its
    headline counts are reproduced by that script. A search or tool call that fails, errors, or
    returns "No matches found" where matches must exist is reported as a failure. The gap is
    never filled by inference, and a subagent's summary is input to verify, not a source to
    copy.
11. **The benchmark measures what production does.** Any setting that differs between the
    reference walk-forward configuration (the benchmark arms in `.agents/benchmarks.md`) and the
    production weekly run (`config/weekly_run.yaml` and the weekly stage-1 selection) is a
    defect until the user approves it and it is recorded here. The season-weighting gap closed in
    `0.18.0` (both train unweighted, task 55.8), and the probability-path gap in `0.32.0` (every
    run type submits the deterministic floor, task 56.5), the device in 2026-09-28 (the two-seed
    GPU reference, task 55.4), the final fit's hold-out in `0.33.0` (the final fit trains on
    every completed game, like each fold), and the line timing in `0.44.0` (task 56.6: both
    anchor to and train on pick-time lines, and the adopted-settings reference of 2026-10-09 is
    production's floor-sigma pool; the one approximation left is that a completed game carries
    nfelo's opening line while production reads nfelo's latest line at run time). Open gaps, all
    scheduled:
    - the floor's sigma and start week (approved by the user 2026-09-28): a default
      `nfl-predictor backtest` (three seasons from week 3, no `--floor-sigma-reference-runs`)
      scores every week at `SCORE_DIFF_STD_DEV`, while production pools the GPU reference and
      stage 1 walks forward from week 1. An arm that judges probabilities runs from 2007 week 1 or
      passes `--floor-sigma-reference-runs` with the GPU reference, and passes
      `--wf-start-week 1`; arms compared only with each other may keep the default.
    - recorded commands: since the adoption of the ported ratings (decided 2026-10-12), a
      backtest with no group option leaves out production's groups
      (`strength,own_elo,own_elo_wt`), so a command recorded before it reproduces with
      `--disable-feature-groups ''` added.
12. **A decision rule favors no outcome after the fact.** The rule written under rule 4 is
    applied as written. If the result falls between its branches, or its first condition fails
    and a fallback branch rescues the preferred outcome, the result goes to the user as a
    question with the numbers.
13. **Two seeds before any default changes.** A walk-forward result that would change a default
    (a setting, a feature family kept or dropped, a tuned parameter) holds on two seeds. Combine
    them per game: the candidate-minus-reference loss difference averaged over the seeds (same
    seed paired with same seed), bootstrapped over games. Game-resampled intervals hold the fit
    fixed, so they leave out seed-to-seed variance: a single-seed six-season interval that
    excludes zero is not enough on its own, above all for pick accuracy and pool points. A new
    reference arm (a new build, device or code path) is measured on two seeds too, which also
    records that reference's own noise floor.
14. **Order of work.** Restructure before changing behavior (a behavior-preserving move is
    pinned by a characterization test and lands before any output-changing task that touches the
    same code). Close benchmark/production parity gaps before new measurements (rule 11). Change
    feature values before tuning. Group changes that invalidate walk-forward checkpoints so they
    share one new reference. Production-changing steps land between game weeks. The step order
    in `.agents/TODO.md` ("Roadmap Status") follows these principles; changing it is must-ask.
15. **Recommend when the answer isn't obvious.** A question or pending decision for the user
    carries the agent's brief recommendation unless the answer is plain from the question itself.
    The user may not remember the details of something built long ago, or may not know the area
    well, so write for a reader who cannot check it without digging: one line on what the
    component does today and why the decision comes up, from the code or docs (name the file),
    not from memory; the recommended option first, with its reason and its main cost or risk;
    how sure the agent is and what would change its mind. If the evidence is too thin to
    recommend, say so and name what would settle it. A recommendation is advice, never consent:
    the agent still waits for the answer on everything under rule 5, and it never bends a
    written decision rule toward its own preference (rules 9 and 12).
16. **Name things in plain words.** Anything written for the user (check-ins, questions,
    summaries, plans and docs) calls an arm, build, run, setting or step by a descriptive name
    that says what it is, for example "the current-settings baseline" or "the adopted-settings
    reference build", never by a bare code such as `R0`, `F`, `PT` or `S2`. A short code may stay
    as a directory or file name; give it once in parentheses after the descriptive name, so the
    user can find the files, then keep using the descriptive name. Task and milestone numbers
    likewise carry a few words saying what the task is.
