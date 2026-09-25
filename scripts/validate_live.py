#!/usr/bin/env python
"""Run ``nfl-predictor validate --live``; the code lives in ``nfl_predictor.cli.validate``.

This path keeps working for the web job runner until it calls ``nfl-predictor validate
--live`` directly.
"""

import sys

from nfl_predictor.cli.validate import main

if __name__ == "__main__":
    raise SystemExit(main(["--live", *sys.argv[1:]]))
