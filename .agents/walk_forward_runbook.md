# Walk-forward runbook: timings and machine load

The timings and machine-load measurements behind the walk-forward rules in `AGENTS.md` (one
walk-forward at a time, `launch.sh` with `nohup setsid`, the OpenMP wait policy).

- Run **one walk-forward at a time**. XGBoost uses every core, and on 2026-09-10 two concurrent
  from-week-1 runs each burned more CPU than a whole solo run (42 CPU-hours against a solo run's
  roughly 30) without finishing, so both were stopped and rerun in sequence. Measured 2026-09-20
  on an idle machine: a from-week-1 run over `--eval-last-n-seasons 3` takes about 50 minutes and
  over `--eval-last-n-seasons 6` about 100 minutes. The earlier observation, kept as a record, put
  a from-week-1 three-season run at about 75 minutes alone and a week-3 start at about 40.
- Choose the OpenMP wait policy by machine load at launch. Under other load, use
  `OMP_WAIT_POLICY=PASSIVE`: with the default policy XGBoost's threads spin while a preempted peer
  catches up (on 2026-09-10 one week took `730s` by default and `185s` with `PASSIVE`). On an idle
  machine keep the default, because sleeping threads cost more to wake than they save on this
  small dataset (an idle `PASSIVE` week took `~142s` against `~82s` for the default). The setting
  changes scheduling only, so it neither alters results nor invalidates fold checkpoints; switching
  mid-run means stop, relaunch with the other policy, and resume. Since `0.24.0` the walk-forward
  commands (`nfl-predictor weekly` and `backtest`) set
  `PASSIVE` themselves unless `OMP_WAIT_POLICY` is already set, because load that arrives mid-run
  stalls the default policy; on a machine known to stay idle, launch with `OMP_WAIT_POLICY=`
  (empty) to keep the library default.
- Load observation from 2026-09-20: with the web API running `--reload` (its file watcher takes
  about half a core continuously) a three-season from-week-1 run took about 110 minutes instead of
  about 50 idle, so the seed-7 arm was relaunched with `OMP_WAIT_POLICY=PASSIVE` and resumed from
  its checkpoints. Check `uptime` and `ps -eo pcpu,args --sort=-pcpu | head -4` before a launch.

## Launching a run

- Every walk-forward starts through a small `launch.sh` in its own run directory: it `cd`s to the
  repo root, runs `.venv/bin/nfl-predictor <command>` with the run's options and echoes the exit
  code. Start it with `nohup setsid`, never from a harness-bound shell, which stops at 10 minutes.
  Several accepted rungs run back to back through one driver script; check the driver's log after
  its first run-to-run transition, not only at the end.
- Any logic added to a `launch.sh` (a load probe, for example) is tested under the script's own
  strict-mode header, because `IFS=$'\n\t'` changes how `read` splits. A probe written without it
  failed and stopped the whole queue overnight on 2026-09-23.
- Never execute a fragment cut from a `launch.sh`: a cut that includes the `nfl-predictor backtest`
  line starts a second walk-forward with default settings. Never `pkill -f` a pattern that can
  match your own shell.
- Since the adoption of the ported ratings (decided 2026-10-12), a backtest with no group option
  leaves out production's groups (`strength,own_elo,own_elo_wt,qb_adj`), so a command recorded
  before it reproduces with `--disable-feature-groups ''` added.
- Progress: every finished week prints a `Walk-forward fold N/M done` line with elapsed and
  remaining time. After a stop, re-running the identical command resumes at the next unfinished
  week (`--no-resume` retrains everything). A `weekly_run`'s own walk-forward of the production
  configuration is in `wf_compare/wf_compare.csv`.
