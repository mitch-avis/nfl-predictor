#!/usr/bin/env python
"""Run the weekly pipeline; the code lives in ``nfl_predictor.weekly_run``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor weekly`` directly.
"""

from nfl_predictor.weekly_run.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
