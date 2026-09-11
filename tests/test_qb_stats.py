"""Tests for the quarterback per-dropback feature family built from play-by-play."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nfl_predictor import constants
from nfl_predictor.ml.walk_forward import resolve_feature_group_columns
from nfl_predictor.utils.polars import qb_stats
from nfl_predictor.utils.polars.finalize import build_final_column_order

_PLAY_DEFAULTS: dict[str, Any] = {
    "season": 2024,
    "week": 1,
    "season_type": "REG",
    "posteam": "NE",
    "defteam": "NYJ",
    "qb_dropback": 0,
    "qb_kneel": 0,
    "qb_spike": 0,
    "two_point_attempt": 0,
    "rush": 0,
    "sack": 0,
    "complete_pass": 0,
    "interception": 0,
    "pass_touchdown": 0,
    "yards_gained": 0.0,
    "qb_epa": 0.0,
    "cpoe": None,
    "passer_player_id": None,
    "passer_player_name": None,
}

_PLAY_DTYPES: dict[str, Any] = {
    "yards_gained": pl.Float64,
    "qb_epa": pl.Float64,
    "cpoe": pl.Float64,
    "passer_player_id": pl.String,
    "passer_player_name": pl.String,
}


def _play(**overrides: Any) -> dict[str, Any]:
    """Return one fixture play with the shared defaults applied."""
    return _PLAY_DEFAULTS | overrides


def _starter_pass(**overrides: Any) -> dict[str, Any]:
    """Return a dropback by the fixture starter ``S``."""
    base = {"qb_dropback": 1, "passer_player_id": "S", "passer_player_name": "S.Starter"}
    return _play(**(base | overrides))


def _qb_game(season: int, week: int, team: str, qb_id: str, **sums: float) -> dict[str, Any]:
    """Return one quarterback-game row of sums, zero for every sum not given."""
    row: dict[str, Any] = {
        "season": season,
        "week": week,
        "team_abbr": team,
        "qb_id": qb_id,
        "passer_name": None,
    }
    for column in qb_stats.QB_GAME_SUM_COLUMNS:
        row[column] = float(sums.get(column, 0.0))
    return row


def _qb_games(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Return a quarterback-game frame with the module's dtypes."""
    return pl.DataFrame(rows, schema=qb_stats.QB_GAME_SCHEMA)


# Two quarterbacks with hand-chosen sums. League totals before 2023 week 2 are A1 + B1,
# and before 2023 week 3 they are A1 + B1 + A2.
_A1 = _qb_game(
    2023, 1, "NE", "A",
    dropbacks=30, qb_epa_sum=6.0, attempts=28, completions=18, pass_yards=200,
    pass_tds=2, interceptions=0, sacks=2, sack_yards=12, cpoe_sum=28.0, cpoe_count=28,
)  # fmt: skip
_A2 = _qb_game(
    2023, 2, "NE", "A",
    dropbacks=40, qb_epa_sum=-4.0, attempts=36, completions=20, pass_yards=250,
    pass_tds=1, interceptions=2, sacks=4, sack_yards=30, cpoe_sum=-36.0, cpoe_count=36,
)  # fmt: skip
_B1 = _qb_game(
    2023, 1, "BUF", "B",
    dropbacks=35, qb_epa_sum=3.5, attempts=33, completions=22, pass_yards=240,
    pass_tds=1, interceptions=1, sacks=2, sack_yards=10, cpoe_sum=0.0, cpoe_count=33,
)  # fmt: skip

_IDENTITY = pl.DataFrame(
    {"qb_name": ["Name A", "Name B", "Name New"], "qb_id": ["A", "B", "N"]},
    schema={"qb_name": pl.String, "qb_id": pl.String},
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


def test_build_qb_identity_drops_ambiguous_names_and_adds_aliases() -> None:
    """Keeps one id per unambiguous name, drops null ids, and maps the configured aliases."""
    meta = pl.DataFrame(
        {
            "name_id": ["Tom Brady", "Mark Miller", "Mark Miller", "No Id", "AJ McCarron"],
            "gsis_id": ["00-1", "X1", "X2", None, "00-2"],
        }
    )

    identity = qb_stats.build_qb_identity(meta)
    mapping = dict(zip(identity["qb_name"], identity["qb_id"], strict=True))

    assert mapping["Tom Brady"] == "00-1"
    assert mapping["AJ McCarron"] == "00-2"
    assert mapping["A.J. McCarron"] == "00-2"
    assert "Mark Miller" not in mapping
    assert "No Id" not in mapping
    assert identity["qb_name"].is_unique().all()


def test_aggregate_qb_game_stats_counts_one_quarterback_game() -> None:
    """Sums a starter's and a backup's dropbacks, crediting scrambles to the starter."""
    plays = [
        _starter_pass(complete_pass=1, yards_gained=10.0, pass_touchdown=1, qb_epa=2.0, cpoe=20.0),
        _starter_pass(qb_epa=-0.5, cpoe=-10.0),
        _starter_pass(interception=1, qb_epa=-3.0),
        _starter_pass(sack=1, yards_gained=-7.0, qb_epa=-1.5),
        # Scramble: a dropback with no passer, credited to the team-game's primary passer.
        _play(qb_dropback=1, rush=1, yards_gained=8.0, qb_epa=1.0),
        # Spike and two-point try: excluded, like the team dropback definition.
        _starter_pass(qb_spike=1, qb_epa=-0.2),
        _starter_pass(two_point_attempt=1, qb_epa=0.9),
        # Designed run and a postseason dropback: excluded.
        _play(rush=1, yards_gained=4.0, qb_epa=0.4),
        _starter_pass(season_type="POST", complete_pass=1, yards_gained=30.0, qb_epa=3.0),
        _play(
            qb_dropback=1,
            passer_player_id="B",
            passer_player_name="B.Backup",
            complete_pass=1,
            yards_gained=5.0,
            qb_epa=0.3,
            cpoe=5.0,
        ),
    ]

    games = qb_stats.aggregate_qb_game_stats(pl.DataFrame(plays, schema_overrides=_PLAY_DTYPES))
    rows = {row["qb_id"]: row for row in games.iter_rows(named=True)}

    assert set(rows) == {"S", "B"}
    starter = rows["S"]
    assert (starter["season"], starter["week"], starter["team_abbr"]) == (2024, 1, "NE")
    assert starter["passer_name"] == "S.Starter"
    assert starter["dropbacks"] == 5
    assert starter["attempts"] == 3
    assert starter["completions"] == 1
    assert starter["pass_yards"] == pytest.approx(10.0)
    assert starter["pass_tds"] == 1
    assert starter["interceptions"] == 1
    assert starter["sacks"] == 1
    assert starter["sack_yards"] == pytest.approx(7.0)
    assert starter["qb_epa_sum"] == pytest.approx(-2.0)
    assert starter["cpoe_sum"] == pytest.approx(10.0)
    assert starter["cpoe_count"] == 2
    backup = rows["B"]
    assert (backup["dropbacks"], backup["attempts"], backup["completions"]) == (1, 1, 1)
    assert backup["qb_epa_sum"] == pytest.approx(0.3)
    assert backup["cpoe_count"] == 1


def test_attach_qb_features_uses_only_earlier_weeks_and_shrinks_to_the_league() -> None:
    """Career rates use games strictly before the row's week, shrunk toward the league."""
    qb_games = _qb_games([_A1, _A2, _B1])
    games = _games([(2023, 2, "Name A", "Name B"), (2023, 3, "Name A", "Name New")])

    out = qb_stats.attach_qb_features(games, qb_games, _IDENTITY)
    week2, week3 = out.row(0, named=True), out.row(1, named=True)

    league_epa_w2 = 9.5 / 65
    career_a_w2 = _shrunk(6.0, 30, league_epa_w2)
    assert week2["away_qb_dropback_epa"] == pytest.approx(career_a_w2)
    assert week2["away_qb_dropback_epa_recent"] == pytest.approx(_shrunk(6.0, 30, career_a_w2))
    assert week2["home_qb_dropback_epa"] == pytest.approx(_shrunk(3.5, 35, league_epa_w2))
    assert week2["away_qb_history_dropbacks"] == 30
    assert week2["qb_dropback_epa_diff"] == pytest.approx(
        week2["away_qb_dropback_epa"] - week2["home_qb_dropback_epa"]
    )

    league_epa_w3 = 5.5 / 105
    assert week3["away_qb_dropback_epa"] == pytest.approx(_shrunk(2.0, 70, league_epa_w3))
    assert week3["away_qb_sack_rate"] == pytest.approx(_shrunk(6, 70, 8 / 105))
    assert week3["away_qb_any_a"] == pytest.approx(_shrunk(378.0, 70, 583.0 / 105))
    assert week3["away_qb_cpoe"] == pytest.approx(_shrunk(-8.0, 64, -8.0 / 97))
    assert week3["away_qb_history_dropbacks"] == 70
    # A first start: no history, so every rate is the league prior.
    assert week3["home_qb_history_dropbacks"] == 0
    assert week3["home_qb_dropback_epa"] == pytest.approx(league_epa_w3)
    assert week3["home_qb_dropback_epa_recent"] == pytest.approx(league_epa_w3)


def test_attach_qb_features_recent_window_shrinks_toward_career(monkeypatch) -> None:
    """The recent window keeps the last games only and is shrunk toward the career rate."""
    monkeypatch.setattr(constants, "QB_RECENT_GAMES", 1)
    qb_games = _qb_games([_A1, _A2, _B1])
    games = _games([(2023, 3, "Name A", "Name B")])

    row = qb_stats.attach_qb_features(games, qb_games, _IDENTITY).row(0, named=True)

    career = _shrunk(2.0, 70, 5.5 / 105)
    assert row["away_qb_dropback_epa_recent"] == pytest.approx(_shrunk(-4.0, 40, career))
    career_any_a = _shrunk(378.0, 70, 583.0 / 105)
    recent_any_a = 250.0 + 20 * 1 - 45 * 2 - 30
    assert row["away_qb_any_a_recent"] == pytest.approx(_shrunk(recent_any_a, 40, career_any_a))


def test_attach_qb_features_ignores_the_game_week_and_later() -> None:
    """Games in the row's own week or later never change its features."""
    games = _games([(2023, 2, "Name A", "Name B")])
    base = qb_stats.attach_qb_features(games, _qb_games([_A1, _B1]), _IDENTITY)

    later = [
        _qb_game(2023, 2, "NE", "A", dropbacks=50, qb_epa_sum=40.0, attempts=50, pass_yards=900),
        _qb_game(2023, 2, "BUF", "B", dropbacks=10, qb_epa_sum=-9.0, attempts=10),
        _qb_game(2024, 1, "NE", "A", dropbacks=40, qb_epa_sum=-20.0, attempts=40),
    ]
    perturbed = qb_stats.attach_qb_features(games, _qb_games([_A1, _B1, *later]), _IDENTITY)

    columns = [c for c in base.columns if "qb_" in c and c not in ("away_qb", "home_qb")]
    assert base.select(columns).equals(perturbed.select(columns))


def test_attach_qb_features_leaves_the_first_week_without_a_prior() -> None:
    """With no earlier quarterback games at all there is no league prior, so rates are null."""
    games = _games([(2023, 1, "Name A", "Name B")])

    row = qb_stats.attach_qb_features(games, _qb_games([_A1, _B1]), _IDENTITY).row(0, named=True)

    assert row["away_qb_history_dropbacks"] == 0
    assert row["away_qb_dropback_epa"] is None
    assert row["away_qb_cpoe"] is None


def test_attach_qb_features_matches_abbreviated_names_and_nulls_unknown_quarterbacks() -> None:
    """Unmapped names fall back to the play-by-play passer name; unknown ones stay null."""
    burrow = _qb_game(2023, 1, "CIN", "JB", dropbacks=40, qb_epa_sum=8.0, attempts=38)
    burrow["passer_name"] = "J.Burrow"
    games = _games([(2023, 2, "Joe Burrow", "Nobody Known"), (2023, 2, "Name A", "Name B")])

    out = qb_stats.attach_qb_features(games, _qb_games([_A1, _B1, burrow]), _IDENTITY)

    assert out.height == 2
    assert out["game_id"].to_list() == ["g0", "g1"]
    first = out.row(0, named=True)
    assert first["away_qb_history_dropbacks"] == 40
    assert first["home_qb_history_dropbacks"] is None
    assert first["home_qb_dropback_epa"] is None
    assert first["qb_dropback_epa_diff"] is None


def test_qb_columns_join_the_final_schema_as_their_own_feature_group() -> None:
    """The final schema carries every quarterback stat per side and as a diff, in one group."""
    order = build_final_column_order()
    expected = sorted(
        column
        for stat in constants.QB_PBP_STATS
        for column in (f"away_{stat}", f"home_{stat}", f"{stat}_diff")
    )

    assert set(expected) <= set(order)
    assert resolve_feature_group_columns(order, ["qb"]) == expected
    for other in ("pbp", "strength"):
        assert not set(resolve_feature_group_columns(order, [other])) & set(expected)
