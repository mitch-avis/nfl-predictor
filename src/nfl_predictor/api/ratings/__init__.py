"""The data behind the team and quarterback pages: each season's ratings and statistics tables.

The ETL writes every season's tables to ``data/ratings/{season}_{kind}.parquet``, the kinds and
columns named as in the ``nfl-sos-ratings`` project that the ratings (:mod:`nfl_predictor.ratings`)
and season statistics (:mod:`nfl_predictor.season_stats`) were ported from. These modules only
read them: they choose each page's columns, describe every column from the metric registry
(:mod:`nfl_predictor.season_stats.metrics`), and never write a file.
"""
