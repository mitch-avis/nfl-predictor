#!/usr/bin/env python
"""Run ``nfl-predictor validate``; the code lives in ``nfl_predictor.cli.validate``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor validate`` directly.
"""

from nfl_predictor.cli.validate import main

if __name__ == "__main__":
    raise SystemExit(main())
