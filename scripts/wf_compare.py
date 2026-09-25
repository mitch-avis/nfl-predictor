#!/usr/bin/env python
"""Run ``nfl-predictor sweep``; the code lives in ``nfl_predictor.cli.sweep``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor sweep`` directly.
"""

from nfl_predictor.cli.sweep import main

if __name__ == "__main__":
    raise SystemExit(main())
