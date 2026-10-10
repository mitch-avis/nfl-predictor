"""The metric registry as the glossary reads it.

The glossary describes the columns these pages show, so it leaves out the garbage-time filter's
metrics (its win-probability bins and kept-play shares) and its ``filtered_`` prefix: the filter
was an exploration view of the ``nfl-sos-ratings`` app that this app does not port.
"""

from __future__ import annotations

from nfl_predictor.api.schemas.ratings import MetricRegistryOut
from nfl_predictor.season_stats.metrics import get_registry

FILTER_PREFIX = "filtered_"
_FILTER_METRIC_MARK = "wp_"


def _is_filter_metric(name: str) -> bool:
    """Return whether a metric belongs to the garbage-time filter (``wp_*``, ``qb_wp_*``)."""
    return name.startswith((_FILTER_METRIC_MARK, f"qb_{_FILTER_METRIC_MARK}"))


def metric_registry() -> MetricRegistryOut:
    """Return every category, metric and column prefix the glossary describes."""
    full = MetricRegistryOut.model_validate(get_registry().payload())
    return full.model_copy(
        update={
            "metrics": {
                name: metric for name, metric in full.metrics.items() if not _is_filter_metric(name)
            },
            "prefix_rules": [rule for rule in full.prefix_rules if rule.prefix != FILTER_PREFIX],
        }
    )
