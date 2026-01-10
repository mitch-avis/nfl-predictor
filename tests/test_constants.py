"""Tests for nfl_predictor.constants.

These tests focus on schema/mapping invariants relied upon by Polars ETL and
feature engineering.
"""

from __future__ import annotations

from nfl_predictor import constants


def test_team_division_mapping_covers_all_canonical_teams() -> None:
    """Ensure that all canonical teams are mapped to divisions/conferences."""
    canonical = set(constants.TEAM_ABBR)
    mapped = set(constants.TEAM_TO_DIVISION.keys())

    assert canonical == mapped
    assert set(constants.TEAM_TO_CONFERENCE.values()) == {"AFC", "NFC"}


def test_team_alias_mapping_normalizes_to_canonical() -> None:
    """Ensure team aliases normalize to canonical abbreviations."""
    for canonical_abbr, meta in constants.TEAM_MAPPING.items():
        assert constants.normalize_team_abbr(canonical_abbr) == canonical_abbr
        for alias in meta["aliases"]:
            assert constants.normalize_team_abbr(alias) == canonical_abbr
            assert constants.normalize_team_abbr(alias.upper()) == canonical_abbr
            assert constants.normalize_team_abbr(alias.lower()) == canonical_abbr


def test_polars_metadata_columns_are_unique_and_include_new_feature_columns() -> None:
    """Ensure that METADATA_COLUMNS has no duplicates and includes all feature columns."""
    cols = constants.METADATA_COLUMNS

    assert len(cols) == len(set(cols)), "METADATA_COLUMNS contains duplicates"

    # Spot-check that feature groups are wired into the schema.
    for required in (
        *constants.RECORD_FEATURE_COLUMNS,
        *constants.DIVISIONAL_FEATURE_COLUMNS,
        *constants.LOOKAHEAD_FEATURE_COLUMNS,
        *constants.MOTIVATION_FEATURE_COLUMNS,
    ):
        assert required in cols


def test_feature_column_groups_have_no_duplicates() -> None:
    """Ensure that feature column groups have no duplicates and are all non-empty strings."""
    groups = (
        constants.RECORD_FEATURE_COLUMNS,
        constants.DIVISIONAL_FEATURE_COLUMNS,
        constants.LOOKAHEAD_FEATURE_COLUMNS,
        constants.MOTIVATION_FEATURE_COLUMNS,
    )

    for group in groups:
        assert len(group) == len(set(group))
        assert all(isinstance(c, str) and c for c in group)
