"""Tests for nfl_predictor.constants.

These tests focus on schema/mapping invariants relied upon by Polars ETL and
feature engineering.
"""

from __future__ import annotations

import pytest

from nfl_predictor import constants


def test_team_division_mapping_covers_all_canonical_teams() -> None:
    """Ensure that all canonical teams are mapped to divisions/conferences."""
    # Arrange
    canonical = set(constants.TEAM_ABBR)

    # Act
    mapped = set(constants.TEAM_TO_DIVISION.keys())

    # Assert
    assert canonical == mapped
    assert set(constants.TEAM_TO_CONFERENCE.values()) == {"AFC", "NFC"}


def test_historical_division_map_handles_pre_realignment_alignment() -> None:
    """Season-aware division helpers should switch to the pre-2002 alignment."""
    # Act
    old_map = constants.division_map_for_season(2001)

    # Assert
    assert old_map["ARI"] == "NFC East"
    assert old_map["SEA"] == "AFC West"
    assert "HOU" not in old_map


def test_division_map_uses_the_realigned_divisions_from_2002() -> None:
    """From 2002 the season-aware division helper returns the realigned divisions."""
    # Act
    new_map = constants.division_map_for_season(2002)

    # Assert
    assert new_map["ARI"] == "NFC West"
    assert new_map["SEA"] == "NFC West"


# Each (raw, canonical) pair once: an alias that is already upper or lower case repeats itself.
_ALIAS_CASES = list(
    dict.fromkeys(
        case
        for canonical_abbr, meta in constants.TEAM_MAPPING.items()
        for case in [
            (canonical_abbr, canonical_abbr),
            *(
                (variant, canonical_abbr)
                for alias in meta["aliases"]
                for variant in (alias, alias.upper(), alias.lower())
            ),
        ]
    )
)


@pytest.mark.parametrize(("raw", "canonical_abbr"), _ALIAS_CASES)
def test_team_alias_mapping_normalizes_to_canonical(raw: str, canonical_abbr: str) -> None:
    """Ensure team aliases normalize to canonical abbreviations."""
    # Act
    normalized = constants.normalize_team_abbr(raw)

    # Assert
    assert normalized == canonical_abbr


def test_polars_metadata_columns_are_unique_and_include_new_feature_columns() -> None:
    """Ensure that METADATA_COLUMNS has no duplicates and includes all feature columns."""
    # Act
    cols = constants.METADATA_COLUMNS

    # Assert
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
    # Act
    groups = (
        constants.RECORD_FEATURE_COLUMNS,
        constants.DIVISIONAL_FEATURE_COLUMNS,
        constants.LOOKAHEAD_FEATURE_COLUMNS,
        constants.MOTIVATION_FEATURE_COLUMNS,
    )

    # Assert
    for group in groups:
        assert len(group) == len(set(group))
        assert all(isinstance(c, str) and c for c in group)


def test_pbp_columns_contract_is_unique_and_covers_required_fields() -> None:
    """PBP selection list is unique, non-empty, and carries the join/aggregation keys."""
    # Act
    cols = constants.PBP_COLUMNS

    # Assert
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
    # Act
    candidates = constants.PBP_SPECIAL_TEAMS_FLAG_CANDIDATES

    # Assert
    assert isinstance(candidates, tuple)
    assert len(candidates) == len(set(candidates))
    assert candidates == ("special", "special_teams_play")
    for candidate in candidates:
        assert candidate in constants.PBP_COLUMNS


def test_feature_group_column_markers_shape() -> None:
    """Feature-group markers map group names to tuples of column-name substrings."""
    # Act
    markers = constants.FEATURE_GROUP_COLUMN_MARKERS

    # Assert
    assert isinstance(markers, dict)
    assert "pbp" in markers
    for group, group_markers in markers.items():
        assert isinstance(group, str)
        assert group
        assert isinstance(group_markers, tuple)
        assert len(group_markers) == len(set(group_markers))
        assert all(isinstance(m, str) and m for m in group_markers)


def test_pbp_count_columns_match_the_aggregation_output() -> None:
    """Published count contract stays in sync with the aggregation module's output.

    Every team-game column is either an identity key, a context flag that is
    carried but never aggregated, or a count that season-to-date aggregation
    averages. ``is_home`` is the only context flag: it says who hosted the game,
    which opponent-adjusted solves need and no rate ever divides by.
    """
    # Arrange
    from nfl_predictor.utils.polars import pbp

    identity = {"season", "week", "team_abbr", "opponent_abbr"}
    context = {"is_home"}

    # Act
    produced = [c for c in pbp.PBP_TEAM_GAME_COLUMNS if c not in identity | context]

    # Assert
    assert produced == constants.PBP_COUNT_COLUMNS
    assert context <= set(pbp.PBP_TEAM_GAME_COLUMNS)
    assert not context & set(constants.PBP_COUNT_COLUMNS)


def test_pbp_stats_are_unique_and_name_allowed_metrics_explicitly() -> None:
    """Published play-by-play stats are unique and pair each rate with an allowed variant."""
    # Act
    stats = constants.PBP_STATS

    # Assert
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
    # Act
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)

    # Assert
    for col in constants.PBP_COUNT_COLUMNS:
        assert col in excluded, f"{col} would be duplicated as opponent_{col}"


def test_pbp_columns_include_box_score_source_fields() -> None:
    """The cached PBP shape includes the raw fields the play-by-play box-score overlay needs."""
    # Arrange
    required = {
        "fumble",
        "penalty",
        "penalty_team",
        "penalty_yards",
        "first_down_pass",
        "first_down_rush",
        "pass_attempt",
        "rush_attempt",
        "qb_epa",
    }

    # Act
    cols = set(constants.PBP_COLUMNS)

    # Assert
    assert required <= cols


def test_sack_mirror_inputs_are_excluded_from_opponent_generation() -> None:
    """Sack events should be excluded before ETL builds duplicate opponent mirrors."""
    # Act
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)

    # Assert
    assert "def_sacks" in excluded
    assert "times_sacked" in excluded


def test_opponent_mirror_intermediates_are_excluded_from_publication() -> None:
    """Every intermediate mirror is an excluded stat, so the published schema never carries it."""
    # Act
    excluded = set(constants.EXCLUDE_FROM_OPPONENT_STATS)

    # Assert
    assert constants.OPPONENT_MIRROR_INTERMEDIATES == ("times_sacked",)
    assert set(constants.OPPONENT_MIRROR_INTERMEDIATES) <= excluded


def test_rare_events_feature_group_markers_cover_the_noise_family() -> None:
    """The rare-event ablation group should name the exact noise-family columns."""
    # Act
    markers = set(constants.FEATURE_GROUP_COLUMN_MARKERS["rare_events"])

    # Assert
    assert markers == {
        "special_teams_tds",
        "def_fumbles",
        "fumble_recovery_tds",
        "2pt_conversions",
        "def_safeties",
        "def_tds",
    }


def test_pbp_feature_group_markers_catch_new_columns_only() -> None:
    """The ablation group matches every published play-by-play stat and nothing older."""
    # Act
    markers = constants.FEATURE_GROUP_COLUMN_MARKERS["pbp"]

    # Assert
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
    # Act
    stats = constants.NFLREADPY_STATS

    # Assert
    assert "rushing_epa" in stats
    assert "passing_epa" in stats


def test_games_played_is_pruned_on_both_sides() -> None:
    """The record count duplicates `strength_games_played`, so neither side reaches the model.

    `away_/home_games_played` count the team's completed games this season, which
    `away_/home_strength_games_played` already publish per side with a diff. Keeping a second
    copy only gave the model two columns to split on for one fact.
    """
    # Act
    sides = ("away", "home")

    # Assert
    for side in sides:
        assert f"{side}_games_played" in constants.RECORD_FEATURE_COLUMNS
        assert f"{side}_games_played" in constants.PRUNED_FEATURE_COLUMNS


def test_dead_weight_and_mirror_columns_are_pruned() -> None:
    """Columns that carry no usable signal or duplicate another column never reach the model.

    The season-phase flags repeat `week_in_season_norm`; `away_ties` and the remaining
    divisional-standing proxies are near-constant; the next-game flags and the game-count diff
    split on almost nothing; `is_divisional_matchup` equals the schedule's `division` flag from
    2002 on; and the `opponent_def_sacks` / `opponent_times_sacked` mirrors are `times_sacked` /
    `def_sacks` seen from the other sideline.
    """
    # Arrange
    expected = {
        "is_divisional_matchup",
        "season_phase_early",
        "season_phase_mid",
        "season_phase_late",
        "away_ties",
        "home_division_eliminated_proxy",
        "away_division_rank",
        "strength_games_played_diff",
        "away_next_is_home",
        "home_next_is_home",
        "away_next_is_divisional_matchup",
        "home_next_is_divisional_matchup",
        "away_days_to_next_game",
        "home_days_to_next_game",
        "away_opponent_def_sacks",
        "home_opponent_def_sacks",
        "opponent_def_sacks_diff",
        "away_opponent_times_sacked",
        "home_opponent_times_sacked",
        "opponent_times_sacked_diff",
    }

    # Act
    pruned = constants.PRUNED_FEATURE_COLUMNS

    # Assert
    assert expected <= set(pruned)
    assert len(pruned) == len(set(pruned))
