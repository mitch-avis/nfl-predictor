"""Tests for building both own Elo families for the dataset."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_quarterbacks, elo_ratings, elo_win_totals, own_elo_build

_TEAMS = (("NE", "BOS00"), ("LAC", "LAX01"), ("KC", "KAN00"), ("DEN", "DEN00"))


def _schedule(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Return two weeks per season of a four-team round, every game played.

    NE hosts LAC and KC hosts DEN in week 1, and the visitors host the rematches in week 2.
    """
    rows: list[dict[str, object]] = []
    for season in seasons:
        for week in (1, 2):
            for game, (first, second) in enumerate(((0, 1), (2, 3))):
                home, away = (first, second) if week == 1 else (second, first)
                (home_team, stadium), (away_team, _stadium) = _TEAMS[home], _TEAMS[away]
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{away_team}_{home_team}",
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "date": date(season, 9, 10) + timedelta(days=7 * (week - 1) + game),
                        "home_abbr": home_team,
                        "away_abbr": away_team,
                        "home_score": 24 + game,
                        "away_score": 17,
                        "neutral": 0,
                        "stadium_id": stadium,
                    }
                )
    return pl.DataFrame(rows)


def _no_box_scores() -> pl.DataFrame:
    return pl.DataFrame(schema=elo_quarterbacks.BOX_SCORE_SCHEMA)


def _no_named_starters() -> pl.DataFrame:
    return pl.DataFrame(schema={"game_id": pl.String, "team_abbr": pl.String, "qb_id": pl.String})


def _win_totals(seasons: tuple[int, ...]) -> elo_win_totals.WinTotalsSnapshot:
    """Return a downloaded-looking snapshot rating every team in ``seasons``."""
    frame = pl.DataFrame(
        [
            {"season": season, "team_abbr": team, "win_total_elo": 1505.0 + 20.0 * index}
            for season in seasons
            for index, (team, _stadium) in enumerate(_TEAMS)
        ],
        schema=elo_win_totals.WIN_TOTALS_SCHEMA,
    )
    return elo_win_totals.WinTotalsSnapshot(
        frame, "https://example.invalid/wt_ratings.csv", "download", "abc123", Path("snap.csv")
    )


def _build(
    seasons: tuple[int, ...], win_total_seasons: tuple[int, ...]
) -> own_elo_build.OwnEloColumns:
    return own_elo_build.own_elo_columns(
        _schedule(seasons),
        box_scores=_no_box_scores(),
        named_starters=_no_named_starters(),
        win_totals=_win_totals(win_total_seasons),
        seasons=seasons,
    )


def test_own_elo_columns_publish_both_families_for_every_game() -> None:
    # Act
    built = _build((2019, 2020), (2019, 2020))

    # Assert
    assert built.frame.columns == [
        "game_id",
        *constants.OWN_ELO_COLUMNS,
        *constants.OWN_WT_ELO_COLUMNS,
    ]
    assert built.frame.height == _schedule((2019, 2020)).height
    assert built.frame.null_count().row(0) == (0,) * built.frame.width


def test_own_elo_columns_equal_each_preset_built_on_its_own() -> None:
    # Arrange
    schedule = _schedule((2019, 2020))
    adjustments = elo_quarterbacks.value_qb_adjustments(schedule, _no_box_scores())
    market_free = elo_ratings.build_own_elo_features(
        schedule,
        variant="market_free",
        inputs=elo_ratings.EloInputs(qb_adjustments=adjustments),
    )

    # Act
    built = _build((2019, 2020), (2019, 2020))

    # Assert
    assert built.frame.select(market_free.frame.columns).equals(market_free.frame)


@pytest.mark.parametrize("win_total_seasons", [(), (2003,)], ids=["unavailable", "elsewhere"])
def test_win_totals_covering_none_of_the_build_seasons_fail_loudly(
    win_total_seasons: tuple[int, ...],
) -> None:
    # Act & Assert
    with pytest.raises(ValueError, match="cover none of the build's seasons"):
        _build((2019, 2020), win_total_seasons)


def test_build_seasons_without_win_totals_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    # Act
    _build((2019, 2020), (2019,))

    # Assert
    assert "no win totals for season(s) [2020]" in caplog.text


def test_the_metadata_records_both_presets_and_the_win_total_file() -> None:
    # Act
    built = _build((2019, 2020), (2019, 2020))

    # Assert
    record = json.loads(json.dumps(built.metadata))
    assert record["market_free"]["variant"] == "market_free"
    assert record["win_totals"]["variant"] == "win_totals"
    assert record["win_total_file"]["origin"] == "download"
    assert record["win_total_file"]["sha256"] == "abc123"
    assert record["seasons"] == [2019, 2020]
