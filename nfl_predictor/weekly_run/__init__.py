"""The weekly run: data refresh, stage-1 selection, training, predictions and reports.

``pipeline.main`` is the entry point (``nfl-predictor weekly``, ``python -m
nfl_predictor.weekly_run``). ``config`` holds the configuration file and the parser, ``inputs``
the prediction-file and output-path rules, and ``stage1`` the walk-forward comparison.
"""
