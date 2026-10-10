"""Tests for rank ranges: bootstrap rating and rank quantiles per team or quarterback."""

import itertools

import numpy as np
import polars as pl
import pytest

from nfl_predictor.ratings.rating_ranges import (
    PAIR_QUANTILES,
    QB_PAIR_COLUMNS,
    QB_RANGE_COLUMNS,
    RANGE_QUANTILES,
    TEAM_PAIR_COLUMNS,
    TEAM_RANGE_COLUMNS,
    PairColumns,
    RangeColumns,
    quantile_suffix,
    summarize_rank_pairs,
    summarize_rank_ranges,
    summarize_unit_rank_ranges,
)
from nfl_predictor.ratings.team_rating import bootstrap_team_ratings, fit_team_ratings


def _draws(ratings: dict[int, dict[str, float]]) -> pl.DataFrame:
    """Return long draw rows from ``{draw: {unit: rating}}``."""
    return pl.DataFrame(
        [
            {"draw": draw, "unit": unit, "rating": rating}
            for draw, by_unit in ratings.items()
            for unit, rating in by_unit.items()
        ]
    )


_PUBLISHED = pl.DataFrame({"unit": ["A", "B", "C"], "rating": [3.0, 2.0, 1.0]})
_FOUR_DRAWS = {
    0: {"A": 3.0, "B": 2.0, "C": 1.0},
    1: {"A": 3.0, "B": 2.0, "C": 1.0},
    2: {"A": 1.0, "B": 3.0, "C": 2.0},
    3: {"A": 3.0, "B": 1.0, "C": 2.0},
}


def _summary(draws: dict[int, dict[str, float]], eligible: set[str] | None = None) -> pl.DataFrame:
    """Summarize ``draws`` against ``_PUBLISHED`` with plain column names."""
    return summarize_rank_ranges(
        _draws(draws),
        _PUBLISHED,
        RangeColumns(id="unit", rating="rating", rank="rank"),
        eligible=eligible,
    )


def _row(summary: pl.DataFrame, unit: str) -> dict[str, object]:
    """Return one unit's summary row."""
    return summary.filter(pl.col("unit") == unit).row(0, named=True)


def test_summarize_rank_ranges_gives_the_full_rank_distribution() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    assert _row(summary, "A")["rank_probabilities"] == pytest.approx([0.75, 0.0, 0.25])


def test_summarize_rank_ranges_reports_the_published_rank_and_top_probabilities() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    row = _row(summary, "C")
    assert (row["rank"], row["rank_top5_probability"]) == (3, pytest.approx(1.0))


def test_summarize_rank_ranges_takes_rank_quantiles_from_drawn_ranks() -> None:
    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    row = _row(summary, "B")
    assert (row["rank_q025"], row["rank_q500"], row["rank_q975"]) == (1, 2, 3)


def test_summarize_rank_ranges_has_every_quantile_column() -> None:
    # Arrange
    suffixes = [f"_q{round(level * 1000):03d}" for level in RANGE_QUANTILES]
    expected = {f"{base}{suffix}" for base in ("rating", "rank") for suffix in suffixes}

    # Act
    summary = _summary(_FOUR_DRAWS)

    # Assert
    assert expected <= set(summary.columns)


def test_summarize_rank_ranges_ranks_only_eligible_units_and_counts_missing_draws() -> None:
    # Arrange
    draws = {0: {"A": 3.0, "B": 2.0, "C": 9.0}, 1: {"A": 3.0, "C": 9.0}}

    # Act
    summary = _summary(draws, eligible={"A", "B"})

    # Assert
    assert summary.get_column("unit").sort().to_list() == ["A", "B"]
    row = _row(summary, "B")
    assert (row["rank_missing_share"], row["rank_probabilities"]) == (
        pytest.approx(0.5),
        pytest.approx([0.0, 0.5]),
    )


@pytest.mark.parametrize("columns", [TEAM_RANGE_COLUMNS, QB_RANGE_COLUMNS])
def test_summarize_rank_ranges_publishes_columns_in_order(columns: RangeColumns) -> None:
    # Arrange
    draws = _draws(_FOUR_DRAWS).rename({"unit": columns.id, "rating": columns.rating})
    published = _PUBLISHED.rename({"unit": columns.id, "rating": columns.rating})
    quantiles = [
        name
        for level in RANGE_QUANTILES
        for name in (
            f"{columns.rating}{quantile_suffix(level)}",
            f"{columns.rank}{quantile_suffix(level)}",
        )
    ]
    expected = [
        columns.id,
        columns.rank,
        *quantiles,
        f"{columns.rank}_missing_share",
        f"{columns.rank}_top5_probability",
        f"{columns.rank}_top10_probability",
        f"{columns.rank}_probabilities",
    ]

    # Act
    summary = summarize_rank_ranges(draws, published, columns)

    # Assert
    assert summary.columns == expected


# A synthetic league for the calibration check: true effects average to zero.
_OFFENSE = dict(zip("ABCDEFGH", (0.12, 0.07, 0.03, 0.0, -0.01, -0.04, -0.07, -0.10), strict=True))
_DEFENSE = dict(zip("ABCDEFGH", (-0.02, 0.06, 0.09, -0.05, 0.03, -0.08, 0.01, -0.04), strict=True))
_SPECIAL = dict(zip("ABCDEFGH", (0.10, -0.05, 0.0, 0.05, -0.10, 0.0, 0.05, -0.05), strict=True))
_PLAYS, _SPECIAL_PLAYS = 60, 12


def _true_ratings() -> dict[str, float]:
    """Return each team's true points-per-game rating under the generating model."""
    return {
        team: (_OFFENSE[team] + _DEFENSE[team]) * _PLAYS + _SPECIAL[team] * _SPECIAL_PLAYS
        for team in _OFFENSE
    }


def _noisy_league(rng: np.random.Generator) -> pl.DataFrame:
    """Return one season: every pair meets four times, game EPA per play with noise."""
    rows: list[dict[str, object]] = []
    games = itertools.product(range(2), itertools.permutations(_OFFENSE, 2))
    for number, (_, (home, away)) in enumerate(games):
        for team, opponent, sign in ((home, away, 1.0), (away, home, -1.0)):
            epa = 0.02 + _OFFENSE[team] - _DEFENSE[opponent] + 0.01 * sign + rng.normal(0, 0.2)
            special = _SPECIAL[team] / 2 - _SPECIAL[opponent] / 2 + rng.normal(0, 0.3)
            rows.append(
                {
                    "game_id": f"g{number:03d}",
                    "team": team,
                    "opponent_team": opponent,
                    "is_home": sign > 0,
                    "offensive_snaps": _PLAYS,
                    "offensive_epa": epa * _PLAYS,
                    "st_plays": _SPECIAL_PLAYS,
                    "st_epa": special * _SPECIAL_PLAYS,
                }
            )
    return pl.DataFrame(rows)


def test_team_rating_intervals_cover_the_true_ratings_near_the_nominal_rate() -> None:
    # Arrange
    rng = np.random.default_rng(0)
    truth = _true_ratings()
    covered = {0.80: 0, 0.95: 0}
    trials = 0

    # Act
    for replication in range(30):
        game_logs = _noisy_league(rng)
        draws = bootstrap_team_ratings(
            game_logs, fit_team_ratings(game_logs), resamples=200, seed=replication
        )
        for team, ratings in draws.group_by("team"):
            values = ratings.get_column("team_rating").to_numpy()
            for level in covered:
                low, high = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
                covered[level] += int(low <= truth[str(team[0])] <= high)
            trials += 1

    # Assert
    # Tolerances fixed in nfl-sos-ratings before this test first ran there.
    assert 0.85 <= covered[0.95] / trials <= 1.0
    assert 0.65 <= covered[0.80] / trials <= 0.95


_PAIR_COLUMNS = PairColumns(
    id="unit", other="other_unit", rating="rating", above="above", gap="gap", share="share"
)


def _pairs(draws: dict[int, dict[str, float]], eligible: set[str] | None = None) -> pl.DataFrame:
    """Summarize the pairs in ``draws`` against ``_PUBLISHED`` with plain column names."""
    return summarize_rank_pairs(_draws(draws), _PUBLISHED, _PAIR_COLUMNS, eligible=eligible)


def _pair(pairs: pl.DataFrame, unit: str, other: str) -> dict[str, object]:
    """Return one ordered pair's row."""
    return pairs.filter((pl.col("unit") == unit) & (pl.col("other_unit") == other)).row(
        0, named=True
    )


def _above(pairs: pl.DataFrame, unit: str, other: str) -> float:
    """Return the chance that ``unit`` is rated above ``other``."""
    return float(
        pairs.filter((pl.col("unit") == unit) & (pl.col("other_unit") == other))
        .select("above")
        .item()
    )


def test_summarize_rank_pairs_counts_the_draws_rating_a_unit_above_another() -> None:
    # Act
    pairs = _pairs(_FOUR_DRAWS)

    # Assert
    assert _pair(pairs, "A", "B")["above"] == pytest.approx(0.75)


def test_summarize_rank_pairs_chances_of_a_pair_add_up_to_one() -> None:
    # Act
    pairs = _pairs(_FOUR_DRAWS)

    # Assert
    for unit, other in itertools.permutations("ABC", 2):
        assert _above(pairs, unit, other) + _above(pairs, other, unit) == pytest.approx(1.0)


def test_summarize_rank_pairs_counts_a_tie_as_half() -> None:
    # Arrange
    draws = {0: {"A": 1.0, "B": 1.0, "C": 0.0}, 1: {"A": 2.0, "B": 1.0, "C": 0.0}}

    # Act
    pairs = _pairs(draws)

    # Assert
    assert _pair(pairs, "A", "B")["above"] == pytest.approx(0.75)


def test_summarize_rank_pairs_takes_gap_quantiles_from_the_drawn_differences() -> None:
    # Arrange
    differences = np.array([1.0, 1.0, -2.0, 2.0])
    gap_columns = [f"gap{quantile_suffix(level)}" for level in PAIR_QUANTILES]
    expected = np.quantile(differences, PAIR_QUANTILES).tolist()

    # Act
    pairs = _pairs(_FOUR_DRAWS)

    # Assert
    row = _pair(pairs, "A", "B")
    assert [row[column] for column in gap_columns] == pytest.approx(expected)


def test_summarize_rank_pairs_uses_only_draws_with_both_units() -> None:
    # Arrange
    draws = {
        0: {"A": 3.0, "B": 2.0, "C": 1.0},
        1: {"A": 1.0, "B": 2.0},
        2: {"A": 3.0, "C": 4.0},
        3: {"A": 2.0, "B": 1.0, "C": 3.0},
    }

    # Act
    pairs = _pairs(draws)

    # Assert
    row = _pair(pairs, "A", "C")
    assert (row["above"], row["share"]) == (pytest.approx(1 / 3), pytest.approx(0.75))


def test_summarize_rank_pairs_without_shared_draws_leaves_the_chance_empty() -> None:
    # Arrange
    draws = {0: {"A": 1.0, "B": 2.0}, 1: {"A": 1.0, "C": 2.0}}

    # Act
    pairs = _pairs(draws)

    # Assert
    row = _pair(pairs, "B", "C")
    assert (row["above"], row["gap_q500"], row["share"]) == (None, None, 0.0)


def test_summarize_rank_pairs_pairs_only_eligible_units_in_order() -> None:
    # Act
    pairs = _pairs(_FOUR_DRAWS, eligible={"A", "C"})

    # Assert
    assert pairs.select("unit", "other_unit").rows() == [("A", "C"), ("C", "A")]


def test_summarize_rank_pairs_rates_a_clearly_better_team_above_nearly_always() -> None:
    # Arrange
    game_logs = _noisy_league(np.random.default_rng(1))
    fit = fit_team_ratings(game_logs)
    draws = bootstrap_team_ratings(game_logs, fit, resamples=200, seed=0)

    # Act
    pairs = summarize_rank_pairs(draws, fit.ratings, TEAM_PAIR_COLUMNS)

    # Assert
    row = pairs.filter((pl.col("team") == "A") & (pl.col("other_team") == "H")).row(0, named=True)
    assert row["team_rated_above_probability"] > 0.99
    assert row["team_rating_gap_q025"] > 0.0


@pytest.mark.parametrize("columns", [TEAM_PAIR_COLUMNS, QB_PAIR_COLUMNS])
def test_summarize_rank_pairs_publishes_columns_in_order(columns: PairColumns) -> None:
    # Arrange
    draws = _draws(_FOUR_DRAWS).rename({"unit": columns.id, "rating": columns.rating})
    published = _PUBLISHED.rename({"unit": columns.id, "rating": columns.rating})
    expected = [
        columns.id,
        columns.other,
        columns.above,
        *(f"{columns.gap}{quantile_suffix(level)}" for level in PAIR_QUANTILES),
        columns.share,
    ]

    # Act
    pairs = summarize_rank_pairs(draws, published, columns)

    # Assert
    assert pairs.columns == expected


def test_summarize_unit_rank_ranges_ranks_each_unit_rating_on_its_own() -> None:
    # Arrange
    game_logs = _noisy_league(np.random.default_rng(2))
    fit = fit_team_ratings(game_logs)
    draws = bootstrap_team_ratings(game_logs, fit, resamples=40, seed=0)
    defense = RangeColumns(id="team", rating="defense_rating", rank="defense_rank")
    expected = summarize_rank_ranges(draws, fit.ratings, defense).drop(
        "defense_rank_missing_share",
        "defense_rank_top5_probability",
        "defense_rank_top10_probability",
        "defense_rank_probabilities",
    )

    # Act
    summary = summarize_unit_rank_ranges(draws, fit.ratings)

    # Assert
    defense_part = summary.select(expected.columns).sort("team")
    assert defense_part.equals(expected.sort("team"))


def test_summarize_unit_rank_ranges_has_one_row_per_team_in_published_order() -> None:
    # Arrange
    game_logs = _noisy_league(np.random.default_rng(3))
    fit = fit_team_ratings(game_logs)
    draws = bootstrap_team_ratings(game_logs, fit, resamples=10, seed=0)

    # Act
    summary = summarize_unit_rank_ranges(draws, fit.ratings)

    # Assert
    assert summary.get_column("team").to_list() == fit.ratings.get_column("team").to_list()
    assert [column for column in summary.columns if column.endswith("_rank")] == [
        "offense_rank",
        "defense_rank",
        "special_teams_rank",
    ]
