"""Tests for the ratings tables' column metadata, read from the metric registry."""

from __future__ import annotations

import math

import polars as pl
import pytest

from nfl_predictor.api.ratings.columns import metric_column, metric_table
from nfl_predictor.api.schemas.ratings import MetricColumnOut


def test_metric_column_describes_a_rating_from_the_registry() -> None:
    """A graded score gets its registry label, polarity, taxonomy and a gradient."""
    # Act
    meta = metric_column("team_rating", pl.Float64(), "ratings")

    # Assert
    assert meta is not None
    assert (meta.key, meta.label, meta.group, meta.kind) == (
        "team_rating",
        "Team Rating",
        "ratings",
        "float",
    )
    assert (meta.polarity, meta.heatmap, meta.contextual, meta.shape) == (
        "higher",
        True,
        False,
        "score",
    )
    assert meta.category == "Schedule-Adjusted Ratings"
    assert meta.full_name
    assert meta.description.endswith(".")
    assert meta.decimals is None


def test_metric_column_marks_schedule_strength_as_context() -> None:
    """Schedule strength is shaded as context, not as a grade."""
    # Act
    meta = metric_column("sos", pl.Float64(), "ratings")

    # Assert
    assert meta is not None
    assert (meta.contextual, meta.heatmap) == (True, True)


def test_metric_column_shows_a_proportion_as_a_percentage() -> None:
    """A registry proportion is a probability-style column the page shows as a percentage."""
    # Act
    meta = metric_column("completion_pct", pl.Float64(), "per_game_rates")

    # Assert
    assert meta is not None
    assert (meta.kind, meta.percent) == ("prob", True)


@pytest.mark.parametrize(
    ("column", "dtype", "kind"),
    [
        ("games_played", pl.Int64(), "int"),
        ("points_for", pl.Float64(), "float"),
        ("is_home", pl.Boolean(), "bool"),
        ("opponent_team", pl.String(), "text"),
    ],
)
def test_metric_column_kind_follows_the_shape_then_the_type(
    column: str, dtype: pl.DataType, kind: str
) -> None:
    """Identity text, flags, whole numbers and decimals each get their own kind."""
    # Act
    meta = metric_column(column, dtype, "identity")

    # Assert
    assert meta is not None
    assert meta.kind == kind


def test_metric_column_leaves_neutral_and_identity_columns_unshaded() -> None:
    """Only graded or contextual numbers carry a gradient."""
    # Act
    metas = [metric_column(name, pl.Int64(), "identity") for name in ("games_played", "week")]

    # Assert
    assert [meta.heatmap for meta in metas if meta is not None] == [False, False]


@pytest.mark.parametrize(
    ("column", "sticky"), [("team", True), ("qb_name", True), ("qb_id", False)]
)
def test_metric_column_pins_the_name_column_of_the_identity_group(
    column: str, *, sticky: bool
) -> None:
    """The team code or the quarterback's name stays visible while a wide table scrolls."""
    # Act
    meta = metric_column(column, pl.String(), "identity")

    # Assert
    assert meta is not None
    assert meta.sticky is sticky


def test_metric_column_is_none_for_a_column_the_registry_cannot_resolve() -> None:
    """An unknown column has no metadata."""
    # Act
    meta = metric_column("not_a_metric", pl.Float64(), "per_game_rates")

    # Assert
    assert meta is None


def test_metric_table_keeps_group_order_and_drops_unknown_columns() -> None:
    """Visible columns follow the groups; a column the registry cannot describe is left out."""
    # Arrange
    frame = pl.DataFrame(
        {
            "team": ["BUF", "KC"],
            "team_rating": [6.5, math.nan],
            "not_a_metric": [1.0, 2.0],
            "points_for": [27.0, None],
        }
    )
    groups = {
        "identity": ["team"],
        "ratings": ["team_rating", "not_a_metric"],
        "per_game_rates": ["points_for", "missing_column"],
    }

    # Act
    table = metric_table(frame, groups)

    # Assert
    assert table.visible_columns == ["team", "team_rating", "points_for"]
    assert table.column_groups == {
        "identity": ["team"],
        "ratings": ["team_rating"],
        "per_game_rates": ["points_for"],
    }
    assert table.rows == [
        {"team": "BUF", "team_rating": 6.5, "points_for": 27.0},
        {"team": "KC", "team_rating": None, "points_for": None},
    ]
    assert set(table.column_metadata) == {"team", "team_rating", "points_for"}


def test_metric_table_reference_groups_name_columns_without_moving_them() -> None:
    """A reference group lists columns that keep their own group and place."""
    # Arrange
    frame = pl.DataFrame(
        {"qb_name": ["J.Allen"], "adj_qb_epa_per_dropback": [0.2], "qb_epa_per_dropback": [0.25]}
    )
    groups = {
        "identity": ["qb_name"],
        "ratings": ["adj_qb_epa_per_dropback"],
        "per_dropback_rates": ["qb_epa_per_dropback"],
    }

    # Act
    table = metric_table(
        frame, groups, references={"rating_companions": ["qb_epa_per_dropback", "qb_sack_rate"]}
    )

    # Assert
    assert table.visible_columns == ["qb_name", "adj_qb_epa_per_dropback", "qb_epa_per_dropback"]
    assert table.column_groups["rating_companions"] == ["qb_epa_per_dropback"]
    assert table.column_metadata["qb_epa_per_dropback"].group == "per_dropback_rates"


def test_metric_table_describes_columns_the_registry_lacks_from_local_metadata() -> None:
    """A table's own columns (a leaderboard's ranks) carry the metadata it supplies."""
    # Arrange
    frame = pl.DataFrame({"team": ["BUF"], "softest_rank": [1]})
    local = {
        "softest_rank": MetricColumnOut(
            key="softest_rank",
            label="Softest Rank",
            description="Rank 1 is the softest schedule.",
            group="ranks",
            kind="int",
            full_name="Softest Schedule Rank",
        )
    }

    # Act
    table = metric_table(frame, {"identity": ["team"], "ranks": ["softest_rank"]}, local=local)

    # Assert
    assert table.visible_columns == ["team", "softest_rank"]
    assert table.column_metadata["softest_rank"].label == "Softest Rank"


def test_metric_table_keeps_list_cells() -> None:
    """A rank range's probabilities stay a list, with any NaN made null."""
    # Arrange
    frame = pl.DataFrame({"team": ["BUF"], "team_rank_probabilities": [[0.5, math.nan, 0.5]]})

    # Act
    table = metric_table(frame, {"identity": ["team"], "rank_chances": ["team_rank_probabilities"]})

    # Assert
    assert table.rows == [{"team": "BUF", "team_rank_probabilities": [0.5, None, 0.5]}]


def test_metric_table_applies_overrides_to_the_metadata() -> None:
    """A table can rewrite a column's description where its meaning differs from the registry's."""
    # Arrange
    frame = pl.DataFrame({"opp_team_rating": [1.0]})

    # Act
    table = metric_table(
        frame,
        {"opponent_ratings": ["opp_team_rating"]},
        overrides={"opp_team_rating": {"description": "This game's opponent.", "full_name": "X"}},
    )

    # Assert
    meta = table.column_metadata["opp_team_rating"]
    assert (meta.description, meta.full_name, meta.contextual) == (
        "This game's opponent.",
        "X",
        True,
    )
