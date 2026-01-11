"""Unit tests for quarterback ID constants.

This module is intentionally lightweight: it primarily verifies that the
`nfl_predictor.qb_ids` data module can be imported and that its schema is
consistent for a few representative entries.
"""

from __future__ import annotations

from nfl_predictor import qb_ids


def test_active_qb_ids_import_and_shape() -> None:
    """Importing qb_ids exposes a non-empty ACTIVE_QB_IDS mapping."""

    assert isinstance(qb_ids.ACTIVE_QB_IDS, dict)
    assert qb_ids.ACTIVE_QB_IDS

    sample_key, sample_value = next(iter(qb_ids.ACTIVE_QB_IDS.items()))
    assert isinstance(sample_key, str)
    assert isinstance(sample_value, dict)
    assert {"name", "draft_year", "draft_number"}.issubset(sample_value.keys())


def test_active_qb_ids_contains_tom_brady() -> None:
    """The QB ID mapping contains a few well-known canonical entries."""

    # This ID is present near the top of the file; it should be stable.
    brady = qb_ids.ACTIVE_QB_IDS["00-0019596"]
    assert brady["name"] == "Tom Brady"
    assert brady["draft_year"] == 2000
    assert brady["draft_number"] == 199
