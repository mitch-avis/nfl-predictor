#!/usr/bin/env python
"""Run ``nfl-predictor sweep``; the code lives in ``nfl_predictor.cli.sweep``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor sweep`` directly.
"""

from nfl_predictor.cli.openmp import prefer_passive_wait_policy

if __name__ == "__main__":
    # The wait policy must be set before the command's imports load XGBoost.
    prefer_passive_wait_policy()
    from nfl_predictor.cli.sweep import main

    raise SystemExit(main())
