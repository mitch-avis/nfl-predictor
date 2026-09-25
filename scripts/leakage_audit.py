#!/usr/bin/env python
"""Run ``nfl-predictor leakage-audit``; the code lives in ``nfl_predictor.cli.leakage_audit``.

This path keeps working for the web job runner and existing launchers until they call
``nfl-predictor leakage-audit`` directly.
"""

from nfl_predictor.cli.leakage_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
