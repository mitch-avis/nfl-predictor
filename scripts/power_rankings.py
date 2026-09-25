#!/usr/bin/env python
"""Run ``nfl-predictor rankings``; the code lives in ``nfl_predictor.cli.rankings``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor rankings`` directly.
"""

from nfl_predictor.cli.rankings import main

if __name__ == "__main__":
    raise SystemExit(main())
