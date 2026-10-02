"""Tests for the defense-adjusted quarterback EPA rate built from pre-game strength snapshots."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.ml.walk_forward import resolve_feature_group_columns
from nfl_predictor.utils.polars import qb_stats
from nfl_predictor.utils.polars.finalize import build_final_column_order

_GROUP = "qb_def_adj"
_DEFENSE = constants.QB_DEF_ADJ_SOURCE_STAT


def _qb_game(key: tuple[int, int, str, str, str], dropbacks: float, epa: float) -> dict[str, Any]:
    """Return one quarterback-game row with dropbacks and EPA, zero for every other sum.

    ``key`` is ``(season, week, team, quarterback id, defense faced)``.
    """
    season, week, team, qb_id, opponent = key
    row: dict[str, Any] = {
        "season": season,
        "week": week,
        "team_abbr": team,
        "opponent_abbr": opponent,
        "qb_id": qb_id,
        "passer_name": None,
    }
    row.update(dict.fromkeys(qb_stats.QB_GAME_SUM_COLUMNS, 0.0))
    row["dropbacks"] = float(dropbacks)
    row["attempts"] = float(dropbacks)
    row["qb_epa_sum"] = float(epa)
    return row


def _qb_games(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a quarterback-game frame with the module's dtypes."""
    return pl.DataFrame(rows, schema=qb_stats.QB_GAME_SCHEMA)


def _snapshots(rows: list[tuple[int, int, str, float | None]]) -> pl.DataFrame:
    """Return pre-week strength snapshot rows: season, week, team and its pass-defense value."""
    return pl.DataFrame(
        [{"season": s, "week": w, "team_abbr": t, _DEFENSE: v} for s, w, t, v in rows],
        schema={"season": pl.Int64, "week": pl.Int64, "team_abbr": pl.String, _DEFENSE: pl.Float64},
    )


def _games(rows: list[tuple[int, int, str, str]]) -> pl.DataFrame:
    """Return game rows keyed by season, week and the two quarterback names."""
    return pl.DataFrame(
        [
            {"game_id": f"g{i}", "season": s, "week": w, "away_qb": a, "home_qb": h}
            for i, (s, w, a, h) in enumerate(rows)
        ]
    )


def _shrunk(numerator: float, denominator: float, prior: float) -> float:
    """Return ``(numerator + K * prior) / (denominator + K)`` with the configured K."""
    k = constants.QB_PRIOR_DROPBACKS
    return (numerator + k * prior) / (denominator + k)


_IDENTITY = pl.DataFrame(
    {"qb_name": ["Name A", "Name B", "Name New"], "qb_id": ["A", "B", "N"]},
    schema={"qb_name": pl.String, "qb_id": pl.String},
)

# Quarterback A faces NYJ in week 1 and MIA in week 2; B faces MIA in week 1.
_A1 = _qb_game((2023, 1, "NE", "A", "NYJ"), dropbacks=30, epa=6.0)
_A2 = _qb_game((2023, 2, "NE", "A", "MIA"), dropbacks=40, epa=-4.0)
_B1 = _qb_game((2023, 1, "BUF", "B", "MIA"), dropbacks=35, epa=3.5)

# Each defense's pre-game value for the week it was faced, plus values for other weeks that a
# game must never read (NYJ in week 2, MIA in week 3, which is the predicted week).
_SNAPSHOTS = _snapshots(
    [
        (2023, 1, "NYJ", 0.05),
        (2023, 1, "MIA", -0.02),
        (2023, 2, "MIA", 0.03),
        (2023, 2, "NYJ", 0.40),
        (2023, 3, "MIA", -0.50),
        (2023, 3, "NYJ", 0.60),
    ]
)


def test_defense_adjusted_rate_credits_each_dropback_with_the_faced_defense() -> None:
    """Per game ``qb_epa_sum + dropbacks * adj_def_pass_epa_snap``, shrunk like the raw rate."""
    out = qb_stats.attach_qb_features(
        _games([(2023, 3, "Name A", "Name New")]),
        _qb_games([_A1, _A2, _B1]),
        _IDENTITY,
        defense_snapshots=_SNAPSHOTS,
    )
    row = out.row(0, named=True)

    adjusted_a1 = 6.0 + 30 * 0.05
    adjusted_a2 = -4.0 + 40 * 0.03
    adjusted_b1 = 3.5 + 35 * -0.02
    league = (adjusted_a1 + adjusted_a2 + adjusted_b1) / 105
    career = _shrunk(adjusted_a1 + adjusted_a2, 70, league)
    assert row["away_qb_def_adj_epa"] == pytest.approx(career)
    assert row["away_qb_def_adj_epa_recent"] == pytest.approx(
        _shrunk(adjusted_a1 + adjusted_a2, 70, career)
    )
    # A first start has no history: both windows are the league's adjusted rate.
    assert row["home_qb_def_adj_epa"] == pytest.approx(league)
    assert row["home_qb_def_adj_epa_recent"] == pytest.approx(league)
    assert row["qb_def_adj_epa_diff"] == pytest.approx(career - league)
    assert row["qb_def_adj_epa_recent_diff"] == pytest.approx(
        row["away_qb_def_adj_epa_recent"] - row["home_qb_def_adj_epa_recent"]
    )


def test_defense_adjusted_recent_window_keeps_the_last_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recent window sums only the last games and shrinks toward the career rate."""
    monkeypatch.setattr(constants, "QB_RECENT_GAMES", 1)
    out = qb_stats.attach_qb_features(
        _games([(2023, 3, "Name A", "Name B")]),
        _qb_games([_A1, _A2, _B1]),
        _IDENTITY,
        defense_snapshots=_SNAPSHOTS,
    )
    row = out.row(0, named=True)

    adjusted_a2 = -4.0 + 40 * 0.03
    assert row["away_qb_def_adj_epa_recent"] == pytest.approx(
        _shrunk(adjusted_a2, 40, row["away_qb_def_adj_epa"])
    )


def test_week_n_value_reads_only_pre_game_snapshots_and_earlier_games() -> None:
    """A week-N value never sees week-N-or-later snapshots, later games, or other weeks' values.

    Each earlier game reads its defense's snapshot for that game's own week, which is solved
    from games strictly before it; snapshots for the predicted week or later, the same
    defense's value in a different week, and quarterback games from week N on never count.
    """
    games = _games([(2023, 3, "Name A", "Name B")])
    columns = [f"{side}_{stat}" for stat in constants.QB_DEF_ADJ_STATS for side in ("away", "home")]
    base = qb_stats.attach_qb_features(
        games, _qb_games([_A1, _A2, _B1]), _IDENTITY, defense_snapshots=_SNAPSHOTS
    ).select(columns)

    perturbed_snapshots = pl.concat(
        [
            _SNAPSHOTS.filter(pl.col("week") < 3).with_columns(
                # A faced NYJ in week 1, so NYJ's week-2 value must not count.
                pl.when((pl.col("team_abbr") == "NYJ") & (pl.col("week") == 2))
                .then(-9.0)
                .otherwise(pl.col(_DEFENSE))
                .alias(_DEFENSE)
            ),
            _snapshots([(2023, 3, "MIA", 9.0), (2023, 3, "NYJ", -9.0), (2023, 4, "MIA", 7.0)]),
        ]
    )
    later_games = [
        _qb_game((2023, 3, "NE", "A", "NYJ"), dropbacks=50, epa=40.0),
        _qb_game((2023, 3, "BUF", "B", "MIA"), dropbacks=10, epa=-9.0),
        _qb_game((2024, 1, "NE", "A", "MIA"), dropbacks=40, epa=-20.0),
    ]
    perturbed = qb_stats.attach_qb_features(
        games,
        _qb_games([_A1, _A2, _B1, *later_games]),
        _IDENTITY,
        defense_snapshots=perturbed_snapshots,
    ).select(columns)

    assert base.equals(perturbed)

    # The game-week snapshot is what the value reads: moving it moves the value.
    moved = _SNAPSHOTS.with_columns(
        pl.when((pl.col("team_abbr") == "MIA") & (pl.col("week") == 2))
        .then(0.13)
        .otherwise(pl.col(_DEFENSE))
        .alias(_DEFENSE)
    )
    shifted = qb_stats.attach_qb_features(
        games, _qb_games([_A1, _A2, _B1]), _IDENTITY, defense_snapshots=moved
    )
    assert shifted["away_qb_def_adj_epa"][0] != pytest.approx(base["away_qb_def_adj_epa"][0])


def test_a_defense_without_a_snapshot_counts_as_average() -> None:
    """With no snapshot for the faced defenses the adjustment is zero: the raw rate comes back."""
    games = _games([(2023, 3, "Name A", "Name B")])
    out = qb_stats.attach_qb_features(
        games,
        _qb_games([_A1, _A2, _B1]),
        _IDENTITY,
        defense_snapshots=_snapshots([(2023, 1, "BUF", 0.2), (2023, 2, "NYJ", None)]),
    )
    row = out.row(0, named=True)

    assert row["away_qb_def_adj_epa"] == pytest.approx(row["away_qb_dropback_epa"])
    assert row["away_qb_def_adj_epa_recent"] == pytest.approx(row["away_qb_dropback_epa_recent"])
    assert row["home_qb_def_adj_epa"] == pytest.approx(row["home_qb_dropback_epa"])


def test_defense_adjusted_rate_is_null_without_a_quarterback_or_any_history() -> None:
    """An unknown quarterback, and a week with no earlier game in the league, give nulls."""
    out = qb_stats.attach_qb_features(
        _games([(2023, 2, "Nobody Known", "Name A"), (2023, 1, "Name A", "Name B")]),
        _qb_games([_A1, _A2, _B1]),
        _IDENTITY,
        defense_snapshots=_SNAPSHOTS,
    )
    unknown, first_week = out.row(0, named=True), out.row(1, named=True)

    assert unknown["away_qb_def_adj_epa"] is None
    assert unknown["qb_def_adj_epa_diff"] is None
    assert unknown["home_qb_def_adj_epa"] is not None
    assert first_week["away_qb_def_adj_epa"] is None
    assert first_week["home_qb_def_adj_epa_recent"] is None


def test_without_defense_snapshots_the_adjusted_columns_are_not_built() -> None:
    """Callers that pass no snapshots get the raw family only; the final schema fills nulls."""
    out = qb_stats.attach_qb_features(
        _games([(2023, 3, "Name A", "Name B")]), _qb_games([_A1, _A2, _B1]), _IDENTITY
    )

    assert not [column for column in out.columns if "qb_def_adj" in column]
    assert "away_qb_dropback_epa" in out.columns


def test_unadjusted_quarterback_games_are_counted() -> None:
    """Games whose faced defense has no pre-game value are counted as the group's fallback."""
    counted = qb_stats.attach_defense_expectation(
        _qb_games([_A1, _A2, _B1]), _snapshots([(2023, 1, "NYJ", 0.05), (2023, 2, "MIA", None)])
    )

    assert counted.select(pl.col(qb_stats.DEFENSE_UNKNOWN_FLAG).sum()).item() == 2
    by_id = {row["qb_id"]: row for row in counted.filter(pl.col("week") == 1).iter_rows(named=True)}
    assert by_id["A"][qb_stats.QB_DEF_ADJ_SUM] == pytest.approx(6.0 + 30 * 0.05)
    assert by_id["B"][qb_stats.QB_DEF_ADJ_SUM] == pytest.approx(3.5)


def test_defense_adjusted_columns_form_their_own_feature_group() -> None:
    """The final schema publishes the stats per side and as diffs, in one disjoint group."""
    order = build_final_column_order()
    expected = sorted(
        column
        for stat in constants.QB_DEF_ADJ_STATS
        for column in (f"away_{stat}", f"home_{stat}", f"{stat}_diff")
    )

    assert set(expected) <= set(order)
    assert resolve_feature_group_columns(order, [_GROUP]) == expected
    assert sorted(constants.QB_DEF_ADJ_FEATURE_COLUMNS) == expected
    for other in constants.FEATURE_GROUP_COLUMN_MARKERS:
        if other != _GROUP:
            assert not set(resolve_feature_group_columns(order, [other])) & set(expected), other
    # Inside the default feature range, so training and the leakage audit both see them.
    start = order.index("away_rest")
    end = order.index("home_moneyline")
    assert all(start < order.index(column) < end for column in expected)
