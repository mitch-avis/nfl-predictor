"""Run the weekly pipeline: ``python -m nfl_predictor.weekly_run``."""

from nfl_predictor.cli.openmp import prefer_passive_wait_policy
from nfl_predictor.cli.threads import prefer_one_polars_thread


def _run() -> int:
    """Set the thread defaults, then import and run the pipeline (it loads Polars and XGBoost)."""
    prefer_one_polars_thread()
    prefer_passive_wait_policy()
    from nfl_predictor.weekly_run.pipeline import main

    return main()


raise SystemExit(_run())
