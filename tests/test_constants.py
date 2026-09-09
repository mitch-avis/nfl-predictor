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


def test_pbp_columns_contract_is_unique_and_covers_required_fields() -> None:
    """PBP selection list is unique, non-empty, and carries the join/aggregation keys."""
    cols = constants.PBP_COLUMNS

    assert len(cols) == len(set(cols)), "PBP_COLUMNS contains duplicates"
    assert all(isinstance(c, str) and c for c in cols)
    for required in (
        "game_id",
        "season",
        "season_type",
        "week",
        "posteam",
        "defteam",
        "home_team",
        "away_team",
        "epa",
        "success",
        "yards_gained",
        "qb_dropback",
        "qb_kneel",
        "qb_spike",
        "rush",
        "down",
    ):
        assert required in cols


def test_pbp_special_teams_flag_candidates_are_ordered_and_unique() -> None:
    """Special-teams flag candidates are an ordered, unique tuple of known column names."""
    candidates = constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES

    assert isinstance(candidates, tuple)
    assert len(candidates) == len(set(candidates))
    assert candidates == ("special", "special_teams_play")
    for candidate in candidates:
        assert candidate in constants.PBP_COLUMNS


def test_feature_group_column_markers_shape() -> None:
    """Feature-group markers map group names to tuples of column-name substrings."""
    markers = constants.FEATURE_GROUP_COLUMN_MARKERS

    assert isinstance(markers, dict)
    assert "pbp" in markers
    for group, group_markers in markers.items():
        assert isinstance(group, str) and group
        assert isinstance(group_markers, tuple)
        assert len(group_markers) == len(set(group_markers))
        assert all(isinstance(m, str) and m for m in group_markers)


def test_pbp_count_columns_match_the_aggregation_output() -> None:
    """Published count contract stays in sync with the aggregation module's output."""
    from nfl_predictor.utils.polars import pbp

    identity = {"season", "week", "team_abbr", "opponent_abbr"}
    produced = [c for c in pbp.PBP_TEAM_GAME_COLUMNS if c not in identity]

    assert produced == constants.PBP_COUNT_COLUMNS


def test_pbp_stats_are_unique_and_name_allowed_metrics_explicitly() -> None:
    """Published play-by-play stats are unique and pair each rate with an allowed variant."""
    stats = constants.PBP_STATS

    assert len(stats) == len(set(stats))
    assert all(isinstance(s, str) and s for s in stats)
    for required in (
        "off_pass_epa_per_snap",
        "off_rush_epa_per_snap",
        "def_pass_epa_allowed_per_snap",
        "def_rush_epa_allowed_per_snap",
        "epa_per_dropback",
        "epa_per_carry",
        "epa_margin_per_play",
        "pass_success_rate",
        "rush_success_rate",
        "success_rate_allowed",
        "explosive_pass_rate",
        "explosive_rush_rate",
        "stuffed_rush_rate",
        "early_down_pass_rate",
        "st_epa_margin_per_play",
    ):
        assert required in stats


def test_pbp_count_columns_are_excluded_from_opponent_generation() -> None:
    """Raw play-by-play counts never get a generic opponent mirror.

    The allowed columns already are the opponent's offense for the same game, so a
    generic `opponent_` mirror would duplicate them.
    """
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)

    for col in constants.PBP_COUNT_COLUMNS:
        assert col in excluded, f"{col} would be duplicated as opponent_{col}"


def test_pbp_feature_group_markers_catch_new_columns_only() -> None:
    """The ablation group matches every published play-by-play stat and nothing older."""
    markers = constants.FEATURE_GROUP_COLUMN_MARKERS["pbp"]

    assert markers, "the play-by-play group must define markers"

    for stat in constants.PBP_STATS:
        for prefix in ("away_", "home_", ""):
            column = f"{prefix}{stat}"
            assert any(m in column for m in markers), f"{column} escapes the group"
        assert any(m in f"{stat}_diff" for m in markers)

    pre_existing = (
        *constants.NFLREADPY_STATS,
        *constants.TR_RATINGS,
        *constants.TR_STATS,
        *constants.ELO_COLUMNS,
        *constants.TREND_FEATURE_COLUMNS,
    )
    for stat in pre_existing:
        if stat in constants.PBP_STATS:
            continue
        for column in (f"away_{stat}", f"home_{stat}", f"{stat}_diff", f"opponent_{stat}"):
            assert not any(m in column for m in markers), f"{column} wrongly joined the group"

    for column in ("away_passing_epa", "home_passing_epa", "passing_epa_diff"):
        assert not any(m in column for m in markers)


def test_rushing_epa_is_published() -> None:
    """Rushing EPA is published alongside passing EPA rather than dropped."""
    assert "rushing_epa" in constants.NFLREADPY_STATS
    assert "passing_epa" in constants.NFLREADPY_STATS
