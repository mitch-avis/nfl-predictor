#!/usr/bin/env python
"""Run ``nfl-predictor explain``; the code lives in ``nfl_predictor.cli.explain``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor explain`` directly.
"""

from nfl_predictor.cli.explain import main

if __name__ == "__main__":
    raise SystemExit(main())
