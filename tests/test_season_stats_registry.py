"""Contract tests for the metric registry, the single source of truth for published columns."""

import json
from typing import TYPE_CHECKING

import pytest

from nfl_predictor.season_stats.metrics import (
    CategoryDef,
    MetricDef,
    RegistryValidationError,
    get_registry,
)
from nfl_predictor.season_stats.metrics.rank_ranges import (
    BOOTSTRAP_RESAMPLES,
    RANGE_QUANTILES,
    quantile_suffix,
)
from nfl_predictor.season_stats.metrics.registry import (
    DEFAULT_PREFIX_RULES,
    DEFAULT_SUFFIX_RULES,
    MetricRegistry,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nfl_predictor.season_stats.metrics.schema import Entity, PrefixRule, SuffixRule

# A representative sample of every column shape the pipeline writes or the API serves, with at
# least one column for every affix rule. The full guarantee is enforced at write time by
# main._write_data_file and by the published-data test below.
_OUTPUT_COLUMN_SAMPLES = (
    *(f"team_rank{quantile_suffix(level)}" for level in RANGE_QUANTILES),
    "team",
    "game_id",
    "is_home",
    "games_played",
    "win_pct",
    "point_margin",
    "passing_yards",
    "offensive_snaps",
    "offensive_epa",
    "st_plays",
    "st_epa",
    "points_per_offensive_snap",
    "passing_yards_per_offensive_snap",
    "def_qb_hits_per_defensive_snap",
    "qb_dropbacks",
    "qb_epa_per_dropback",
    "wp_unit",
    "wp_bin",
    "wp_bin_plays",
    "wp_bin_epa",
    "qb_wp_bin_dropbacks",
    "qb_wp_bin_epa",
    "wp_kept_play_share",
    "wp_kept_dropback_share",
    "filtered_team_rank",
    "filtered_sos",
    "filtered_adj_qb_epa_per_dropback",
    "filtered_qb_rank_change",
    "qb_designed_epa_per_carry",
    "qb_attempts_total",
    "qb_completions_per_game",
    "qb_is_eligible",
    "opp_passing_yards",
    "opp_qb_epa_per_dropback",
    "qopp_qb_sack_rate",
    "team_rating",
    "offense_rating",
    "defense_rating",
    "special_teams_rating",
    "sos",
    "SRS",
    "adj_qb_epa_per_dropback",
    "qb_faced_pass_defense",
)


@pytest.fixture(scope="module")
def registry() -> MetricRegistry:
    """Return the validated project registry once per module."""
    return get_registry()


def _metric(name: str, category: str = "Test Category") -> MetricDef:
    """Return a minimal valid metric for synthetic-registry tests."""
    return MetricDef(
        name=name,
        label=name.title(),
        full_name=name.replace("_", " ").title(),
        description="A synthetic metric used only for validation tests.",
        entity="team",
        category=category,
        shape="count",
        polarity="higher",
        source="D",
    )


def _category() -> CategoryDef:
    """Return a minimal category for synthetic-registry tests."""
    return CategoryDef(name="Test Category", entity="team", description="Synthetic category.")


def test_every_rate_and_average_declares_a_denominator(registry: MetricRegistry) -> None:
    # Act
    missing = [
        metric.name
        for metric in registry.metrics.values()
        if metric.shape in ("rate", "avg") and not metric.denominator
    ]

    # Assert
    assert missing == []


def test_every_description_is_a_full_sentence(registry: MetricRegistry) -> None:
    # Act
    short = [
        metric.name
        for metric in registry.metrics.values()
        if len(metric.description) < 20 or not metric.description.endswith(".")
    ]

    # Assert
    assert short == []


def test_every_duplicate_of_target_exists(registry: MetricRegistry) -> None:
    # Act
    dangling = [
        metric.name
        for metric in registry.metrics.values()
        if metric.duplicate_of is not None and metric.duplicate_of not in registry.metrics
    ]

    # Assert
    assert dangling == []


@pytest.mark.parametrize("entity", ["team", "qb"])
def test_ratings_category_leads_each_entity(registry: MetricRegistry, entity: Entity) -> None:
    # Act
    first = registry.categories(entity)[0].name

    # Assert
    assert first == "Schedule-Adjusted Ratings"


@pytest.mark.parametrize("column", _OUTPUT_COLUMN_SAMPLES)
def test_output_column_resolves(registry: MetricRegistry, column: str) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None


def test_unknown_column_resolves_to_none(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("definitely_not_a_metric")

    # Assert
    assert resolved is None


def test_exact_match_wins_over_affix_decomposition(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("adj_qb_epa_per_dropback")

    # Assert
    assert resolved is not None
    assert resolved.base.name == "adj_qb_epa_per_dropback"


def test_quantile_suffix_keeps_the_base_metric_and_its_polarity(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("team_rank_q975")

    # Assert
    assert resolved is not None
    assert (resolved.base.name, resolved.polarity) == ("team_rank", "lower")
    assert (resolved.label, resolved.full_name) == (
        "Rank (97.5th %ile)",
        "Team Rating Rank, 97.5th Percentile",
    )


# The resampling clause every rank-range quantile note shares.
_REDRAWN = "in 1,000 seasons redrawn from the real games (picked at random, repeats allowed)"


@pytest.mark.parametrize(
    ("column", "start", "end"),
    [
        ("team_rank_q025", "Low end of the middle 95%", "2.5% of the results came out smaller."),
        ("team_rank_q100", "Low end of the middle 80%", "10% of the results came out smaller."),
        ("team_rank_q250", "Low end of the middle 50%", "25% of the results came out smaller."),
        ("team_rank_q500", "The middle result", "half of the results came out smaller."),
        ("team_rank_q750", "High end of the middle 50%", "25% of the results came out larger."),
        ("team_rank_q900", "High end of the middle 80%", "10% of the results came out larger."),
        ("team_rank_q975", "High end of the middle 95%", "2.5% of the results came out larger."),
    ],
)
def test_quantile_suffix_says_which_range_it_bounds(
    registry: MetricRegistry, column: str, start: str, end: str
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert resolved.description.endswith(f"{start}: {_REDRAWN}, {end}")


def test_filtered_change_column_names_both_the_filter_and_the_change(
    registry: MetricRegistry,
) -> None:
    # Act
    resolved = registry.resolve_column("filtered_team_rating_change")

    # Assert
    assert resolved is not None
    assert resolved.base.name == "team_rating"
    assert "Garbage-Time Filtered" in resolved.full_name
    assert "Change" in resolved.full_name
    assert "minus the same calculation with every play kept" in resolved.description
    assert "Exploration only" in resolved.description


@pytest.mark.parametrize(
    ("column", "contextual"),
    [
        ("filtered_sos", True),
        ("filtered_qb_faced_pass_defense", True),
        ("filtered_team_rating", False),
    ],
)
def test_filtered_prefix_keeps_the_base_contextual_flag(
    registry: MetricRegistry, column: str, *, contextual: bool
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert resolved.contextual is contextual


def test_per_game_suffix_keeps_the_base_metric(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("qb_attempts_per_game")

    # Assert
    assert resolved is not None
    assert resolved.base.name == "qb_attempts"
    assert resolved.polarity == resolved.base.polarity


@pytest.mark.parametrize(
    "column",
    [
        "filtered_team_rating_change",
        "filtered_team_rank_change",
        "filtered_adj_qb_epa_per_dropback_change",
        "filtered_qb_rank_change",
    ],
)
def test_change_suffix_reads_as_neutral(registry: MetricRegistry, column: str) -> None:
    """A change under the garbage-time filter shows sensitivity to the filter, not quality."""
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert resolved.polarity == "neutral"


@pytest.mark.parametrize(
    ("column", "category"),
    [
        ("opp_passing_yards", "Offense"),
        ("qopp_points_allowed", "Scoring, Clutch & Outcomes"),
        ("qopp_qb_sack_rate", "Pressure, Sacks & Pocket"),
    ],
)
def test_opponent_columns_stay_contextual_inside_the_taxonomy(
    registry: MetricRegistry, column: str, category: str
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert (resolved.contextual, resolved.category) == (True, category)


def test_qopp_prefix_inverts_qb_polarity(registry: MetricRegistry) -> None:
    # Act
    mirrored = registry.resolve_column("qopp_qb_epa_per_dropback")

    # Assert
    assert mirrored is not None
    assert mirrored.polarity == "lower"


def test_validate_columns_returns_exactly_the_unknown_names(registry: MetricRegistry) -> None:
    # Act
    unknown = registry.validate_columns(["team", "team_rating", "not_a_real_column"])

    # Assert
    assert unknown == ["not_a_real_column"]


def test_payload_survives_a_json_round_trip(registry: MetricRegistry) -> None:
    # Act
    payload = json.loads(json.dumps(registry.payload()))

    # Assert
    assert payload["metrics"]["team_rating"]["polarity"] == "higher"
    assert "Opponent Context" not in [
        category["name"] for category in payload["entities"]["team"]["categories"]
    ]


def test_payload_serves_the_prefix_rules(registry: MetricRegistry) -> None:
    """The web app labels the season_delta_ columns it derives with the registry's own rule."""
    # Act
    payload = json.loads(json.dumps(registry.payload()))

    # Assert
    rules = {rule["prefix"]: rule for rule in payload["prefix_rules"]}
    assert list(rules) == [rule.prefix for rule in DEFAULT_PREFIX_RULES]
    assert rules["season_delta_"]["label_template"] == "{label} vs Season"
    assert rules["qopp_"]["invert_polarity_for_qb"] is True


def test_column_metadata_carries_label_and_description(registry: MetricRegistry) -> None:
    # Act
    metadata = registry.column_metadata(["team_rating", "opp_passing_yards"])

    # Assert
    assert metadata["team_rating"]["label"] == "Team Rating"
    assert metadata["opp_passing_yards"]["contextual"] is True


def test_registry_rejects_a_metric_defined_twice() -> None:
    # Arrange
    metrics = [_metric("alpha"), _metric("alpha")]

    # Act & Assert
    with pytest.raises(RegistryValidationError, match="defined twice"):
        MetricRegistry(metrics, [_category()])


def test_registry_rejects_an_unknown_category() -> None:
    # Arrange
    metrics = [_metric("alpha", category="Nowhere")]

    # Act & Assert
    with pytest.raises(RegistryValidationError, match="unknown category"):
        MetricRegistry(metrics, [_category()])


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("opp_qb_offense_snaps", ("Offense", "Total")),
        ("opp_qb_rushing_yards", ("Offense", "Rushing")),
        ("opp_qb_fourth_quarter_comeback", ("Offense", "Scoring")),
        ("opp_qb_td_int_differential", ("Offense", "Turnovers")),
        ("opp_qb_passer_rating", ("Offense", "Passing")),
        ("qopp_points_allowed", ("Scoring, Clutch & Outcomes", None)),
        ("qopp_total_tds", ("Scoring, Clutch & Outcomes", None)),
        ("qopp_def_interceptions", ("Turnovers & Ball Security", None)),
        ("qopp_def_sacks", ("Pressure, Sacks & Pocket", None)),
        ("qopp_def_fumbles_forced", ("Pressure, Sacks & Pocket", None)),
        ("qopp_rushing_yards_allowed", ("Rushing", None)),
        ("qopp_passing_yards_allowed", ("Passing Efficiency", None)),
    ],
)
def test_cross_entity_context_columns_map_onto_the_viewing_taxonomy(
    registry: MetricRegistry, column: str, expected: tuple[str, str | None]
) -> None:
    # Act
    resolved = registry.resolve_column(column)

    # Assert
    assert resolved is not None
    assert (resolved.category, resolved.subcategory) == expected


@pytest.mark.parametrize(
    ("metric", "message"),
    [
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                subcategory="Nowhere",
                shape="count",
                polarity="higher",
                source="D",
            ),
            "unknown subcategory",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                shape="count",
                polarity="higher",
                source="D",
                duplicate_of="missing",
            ),
            "duplicates unknown metric",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                shape="rate",
                polarity="higher",
                source="D",
            ),
            "declares no denominator",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="Too short",
                entity="team",
                category="Test Category",
                shape="count",
                polarity="higher",
                source="D",
            ),
            "full-sentence",
        ),
        (
            MetricDef(
                name="alpha",
                label="Alpha",
                full_name="Alpha",
                description="A synthetic metric used only for validation tests.",
                entity="team",
                category="Test Category",
                shape="count",
                polarity="higher",
                source="D",
                percent=True,
            ),
            "percentage",
        ),
    ],
)
def test_registry_rejects_an_invalid_metric(metric: MetricDef, message: str) -> None:
    # Act & Assert
    with pytest.raises(RegistryValidationError, match=message):
        MetricRegistry([metric], [_category()])


def test_column_metadata_skips_unknown_columns(registry: MetricRegistry) -> None:
    # Act
    metadata = registry.column_metadata(["team_rating", "not_a_real_column"])

    # Assert
    assert list(metadata) == ["team_rating"]


def test_suffix_without_a_known_base_resolves_to_none(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("opp_not_a_metric_per_game")

    # Assert
    assert resolved is None


def test_qopp_prefix_keeps_neutral_polarity_neutral(registry: MetricRegistry) -> None:
    # Act
    resolved = registry.resolve_column("qopp_qb_offense_snaps")

    # Assert
    assert resolved is not None
    assert resolved.polarity == "neutral"


def test_drive_penalty_yards_rewards_net_yards_gained(registry: MetricRegistry) -> None:
    """Verify drive penalty yards grade higher as better for the offense.

    nflverse ``drive_yards_penalized`` is net penalty yards in the offense's favor: positive when
    the defense's fouls gave the offense yards, negative when the offense's own fouls cost it.
    """
    # Act
    resolved = registry.resolve_column("drive_penalty_yards")

    # Assert
    assert resolved is not None
    assert resolved.polarity == "higher"


def test_season_maximum_metrics_have_the_max_shape(registry: MetricRegistry) -> None:
    """The season row keeps a ``longest_`` column's largest game value (``team_stats``)."""
    # Act
    mismatched = [
        metric.name
        for metric in registry.metrics.values()
        if (metric.shape == "max") != metric.name.startswith("longest_")
    ]

    # Assert
    assert mismatched == []


@pytest.mark.parametrize("column", ["longest_pass", "longest_rush", "opp_longest_pass"])
def test_column_metadata_marks_longest_plays_as_maxima(
    registry: MetricRegistry, column: str
) -> None:
    # Act
    metadata = registry.column_metadata([column])

    # Assert
    assert metadata[column]["shape"] == "max"


@pytest.mark.parametrize(
    "metric",
    [
        "def_tackles_for_loss",
        "def_fumbles_forced",
        "def_sacks",
        "def_qb_hits",
        "def_interceptions",
        "def_pass_defended",
        "def_safeties",
    ],
)
def test_defense_player_stats_name_weekly_player_stats_as_source(
    registry: MetricRegistry, metric: str
) -> None:
    """team_stats sums these from nflverse weekly player stats, never from play-by-play."""
    # Act
    source = registry.metrics[metric].source

    # Assert
    assert source == "PLS"


@pytest.mark.parametrize(
    ("column", "percent"),
    [
        ("completion_pct", True),
        ("qb_sack_rate", True),
        ("havoc_rate", True),
        ("success_rate_margin", True),
        ("team_rank_top5_probability", True),
        ("opp_third_down_pct", True),
        ("qopp_qb_completion_pct", True),
        ("yards_per_attempt", False),
        ("qb_completion_percentage_above_expectation", False),
        ("giveaways_per_drive", False),
        ("passing_yards", False),
    ],
)
def test_column_metadata_marks_proportions_as_percentages(
    registry: MetricRegistry, column: str, *, percent: bool
) -> None:
    """0.653 means 65.3%; values already in percentage points (CPOE) are not proportions."""
    # Act
    metadata = registry.column_metadata([column])

    # Assert
    assert metadata[column]["percent"] is percent


def test_payload_marks_proportions_as_percentages(registry: MetricRegistry) -> None:
    # Act
    payload = json.loads(json.dumps(registry.payload()))

    # Assert
    assert payload["metrics"]["third_down_pct"]["percent"] is True
    assert payload["metrics"]["points_per_drive"]["percent"] is False


def _registry_with_rules(
    registry: MetricRegistry,
    prefix_rules: Sequence[PrefixRule],
    suffix_rules: Sequence[SuffixRule],
) -> MetricRegistry:
    """Return the project's metrics and categories under other affix rules."""
    return MetricRegistry(
        list(registry.metrics.values()),
        [*registry.categories("team"), *registry.categories("qb")],
        prefix_rules=prefix_rules,
        suffix_rules=suffix_rules,
    )


@pytest.mark.parametrize(
    "rule", DEFAULT_SUFFIX_RULES, ids=[rule.suffix for rule in DEFAULT_SUFFIX_RULES]
)
def test_every_suffix_rule_resolves_a_real_column(
    registry: MetricRegistry, rule: SuffixRule
) -> None:
    """A rule no real column needs would only describe, and admit, columns that do not exist."""
    # Arrange
    others = [other for other in DEFAULT_SUFFIX_RULES if other is not rule]
    without_rule = _registry_with_rules(registry, DEFAULT_PREFIX_RULES, others)

    # Act
    needing_rule = [
        column for column in _OUTPUT_COLUMN_SAMPLES if without_rule.resolve_column(column) is None
    ]

    # Assert
    assert needing_rule != []


@pytest.mark.parametrize(
    "rule", DEFAULT_PREFIX_RULES, ids=[rule.prefix for rule in DEFAULT_PREFIX_RULES]
)
def test_every_prefix_rule_resolves_a_real_column(
    registry: MetricRegistry, rule: PrefixRule
) -> None:
    """The web app's unique-opponent table derives season_delta_ columns from game-log stats."""
    # Arrange
    others = [other for other in DEFAULT_PREFIX_RULES if other is not rule]
    without_rule = _registry_with_rules(registry, others, DEFAULT_SUFFIX_RULES)
    columns = [*_OUTPUT_COLUMN_SAMPLES, "season_delta_passing_epa"]

    # Act
    needing_rule = [column for column in columns if without_rule.resolve_column(column) is None]

    # Assert
    assert needing_rule != []


def test_rank_range_settings_match_the_published_rank_ranges() -> None:
    # Act
    suffixes = [quantile_suffix(level) for level in RANGE_QUANTILES]

    # Assert
    assert BOOTSTRAP_RESAMPLES == 1000
    assert suffixes == ["_q025", "_q100", "_q250", "_q500", "_q750", "_q900", "_q975"]
