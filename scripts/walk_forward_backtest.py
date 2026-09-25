#!/usr/bin/env python
"""Run ``nfl-predictor backtest``; the code lives in ``nfl_predictor.cli.backtest``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor backtest`` directly.
"""

from nfl_predictor.cli.backtest import main

if __name__ == "__main__":
    raise SystemExit(main())
