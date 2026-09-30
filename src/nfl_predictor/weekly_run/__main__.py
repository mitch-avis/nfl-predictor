"""Run the weekly pipeline: ``python -m nfl_predictor.weekly_run``."""

from nfl_predictor.cli.openmp import prefer_passive_wait_policy


def _run() -> int:
    """Set the OpenMP wait policy, then import and run the pipeline (which loads XGBoost)."""
    prefer_passive_wait_policy()
    from nfl_predictor.weekly_run.pipeline import main

    return main()


raise SystemExit(_run())
