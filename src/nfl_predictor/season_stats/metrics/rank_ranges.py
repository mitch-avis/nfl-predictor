"""The rank-range settings the registry's quantile columns describe.

The ratings' rank ranges redraw a season's games ``BOOTSTRAP_RESAMPLES`` times and publish each
rating's and rank's quantiles at ``RANGE_QUANTILES`` as ``<column>_q025`` through
``<column>_q975``. The ratings package keeps these settings in ``ratings.rating_ranges`` beside the
bootstrap itself, which loads numpy; the registry needs only the three below, so they are restated
here and the registry, which the web API reads, imports nothing from the ratings code.
``tests/test_season_stats_registry.py`` requires them to equal the ratings package's and the
registry to import without numpy.
"""

from __future__ import annotations

BOOTSTRAP_RESAMPLES = 1000
RANGE_QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)


def quantile_suffix(level: float) -> str:
    """Return the column suffix for a quantile level, ``_q025`` for 0.025 through ``_q975``."""
    return f"_q{round(level * 1000):03d}"


__all__ = ["BOOTSTRAP_RESAMPLES", "RANGE_QUANTILES", "quantile_suffix"]
