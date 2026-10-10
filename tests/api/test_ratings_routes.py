"""Route tests for the team and quarterback pages' data: tables, sections, schedules, glossary."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from tests.api import ratings_factories as factories

from nfl_predictor.api.readers import cache
from nfl_predictor.utils import clock

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

SEASON = 2025
KC_QB = "00-0033873"
WSH_QB = "00-0039910"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty table cache."""
    cache.clear()


@pytest.fixture(autouse=True)
def _october_2026(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin today to week 6 of the 2026 season, so 2025 is a completed season."""
    monkeypatch.setattr(clock, "local_today", lambda: date(2026, 10, 14))


@pytest.fixture
def ratings_dir(project_root: Path) -> Path:
    """Return the data directory's ratings folder with the 2025 season written."""
    return factories.write_season(project_root / "data" / "ratings", SEASON)


@pytest.mark.parametrize(
    "path",
    [
        "/api/ratings/seasons",
        "/api/ratings/metrics",
        "/api/ratings/schedules",
        f"/api/ratings/{SEASON}/teams",
        f"/api/ratings/{SEASON}/qbs/rank-ranges",
        f"/api/ratings/{SEASON}/teams/KC/game-logs",
        f"/api/ratings/{SEASON}/teams/KC/rating-history",
        f"/api/ratings/{SEASON}/teams/KC/rank-history",
        f"/api/ratings/{SEASON}/teams/KC/head-to-head",
    ],
)
def test_every_ratings_route_requires_a_session(
    ratings_dir: Path, client: TestClient, path: str
) -> None:
    """Like every data route, the ratings routes refuse an anonymous request."""
    # Act
    response = client.get(path)

    # Assert
    assert response.status_code == 401


def test_seasons_lists_seasons_newest_first_with_the_current_season(
    project_root: Path, viewer_client: TestClient
) -> None:
    """The season picker gets every season with tables and the season under way."""
    # Arrange
    factories.write_season(project_root / "data" / "ratings", 2024)
    factories.write_season(project_root / "data" / "ratings", 2025)

    # Act
    payload = viewer_client.get("/api/ratings/seasons").json()

    # Assert
    assert payload == {"seasons": [2025, 2024], "current_season": 2026}


def test_seasons_is_empty_before_the_etl_writes_ratings(viewer_client: TestClient) -> None:
    """Without a ratings folder the list is empty rather than an error."""
    # Act
    response = viewer_client.get("/api/ratings/seasons")

    # Assert
    assert response.status_code == 200
    assert response.json()["seasons"] == []


def test_team_table_returns_the_season_with_its_categories(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """The team index reads the season's combined table and the registry's categories."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams").json()

    # Assert
    assert (payload["kind"], payload["season"], payload["in_progress"]) == ("teams", SEASON, False)
    assert (payload["games_played"], payload["games_per_team"]) == (17, 17)
    table = payload["table"]
    assert [row["team"] for row in table["rows"]] == ["BUF", "KC", "NE", "WSH"]
    assert table["column_groups"]["ratings"][0] == "team_rating"
    assert table["column_metadata"]["team_rating"]["label"] == "Team Rating"
    assert table["column_metadata"]["sos"]["contextual"] is True
    assert payload["categories"][0]["name"] == "Schedule-Adjusted Ratings"
    assert payload["team_names"]["WSH"] == "Washington Commanders"


def test_qb_table_returns_every_passer_with_the_qualifier(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """The quarterback index lists qualifiers and the backup below the qualifier."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/qbs").json()

    # Assert
    rows = payload["table"]["rows"]
    assert len(rows) == 5
    assert [row["qb_is_eligible"] for row in rows].count(False) == 1
    assert payload["table"]["column_groups"]["rating_companions"] == [
        "qb_epa_per_dropback",
        "qb_dropbacks_total",
    ]


def test_current_season_with_games_left_is_in_progress(
    project_root: Path, viewer_client: TestClient
) -> None:
    """Week 6 of the current season reads as in progress, with the games played so far."""
    # Arrange
    factories.write_season(project_root / "data" / "ratings", 2026, games_played=5)

    # Act
    payload = viewer_client.get("/api/ratings/2026/qbs").json()

    # Assert
    assert (payload["in_progress"], payload["games_played"]) == (True, 5)


def test_a_season_without_tables_is_a_clear_404(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """Asking for a season the ETL never wrote names the missing file."""
    # Act
    response = viewer_client.get("/api/ratings/2019/teams")

    # Assert
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "no_ratings_table"
    assert "2019_combined.parquet" in error["message"]


def test_an_unknown_page_kind_is_rejected(ratings_dir: Path, viewer_client: TestClient) -> None:
    """Only the team and quarterback pages exist."""
    # Act
    response = viewer_client.get(f"/api/ratings/{SEASON}/coaches")

    # Assert
    assert response.status_code == 422


def test_team_game_logs_carry_each_opponents_season_ratings(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """A team's weekly log lists its games in order with every opponent's season rating."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams/WSH/game-logs").json()

    # Assert
    assert (payload["kind"], payload["season"], payload["entity_id"]) == ("teams", SEASON, "WSH")
    rows = payload["table"]["rows"]
    assert [row["week"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert {row["opponent_team"]: row["opp_team_rating"] for row in rows}["BUF"] == 6.5


def test_qb_game_logs_list_one_passers_games(ratings_dir: Path, viewer_client: TestClient) -> None:
    """A quarterback's log holds only his games."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/qbs/{WSH_QB}/game-logs").json()

    # Assert
    assert {row["qb_id"] for row in payload["table"]["rows"]} == {WSH_QB}
    assert len(payload["table"]["rows"]) == 6


def test_game_logs_for_an_unknown_team_are_a_404(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """A team that played no games that season has no log."""
    # Act
    response = viewer_client.get(f"/api/ratings/{SEASON}/teams/LV/game-logs")

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ratings_entity"


def test_game_logs_without_a_ratings_table_still_load(
    project_root: Path, viewer_client: TestClient
) -> None:
    """The game log needs only its own table; the opponent ratings are added when present."""
    # Arrange
    factories.write_season(
        project_root / "data" / "ratings",
        SEASON,
        kinds=("combined", "qb_combined", "team_game_logs"),
    )

    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams/KC/game-logs").json()

    # Assert
    assert payload["table"]["column_groups"]["opponent_ratings"] == []


def test_rating_history_lists_a_teams_weeks(ratings_dir: Path, viewer_client: TestClient) -> None:
    """A team's rating after each week of the season."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams/KC/rating-history").json()

    # Assert
    assert [row["week"] for row in payload["table"]["rows"]] == [1, 2, 3, 4, 5, 6]
    assert payload["table"]["column_groups"]["ratings"][0] == "team_rating"


def test_qb_rating_history_lists_a_passers_weeks(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """A quarterback's rating after each week."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/qbs/{KC_QB}/rating-history").json()

    # Assert
    assert payload["table"]["column_groups"]["ratings"] == ["adj_qb_epa_per_dropback"]
    assert len(payload["table"]["rows"]) == 6


def test_rank_ranges_list_every_team_by_published_rank(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """The league's rank ranges, best first, with each team's chance of every rank."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams/rank-ranges").json()

    # Assert
    assert payload["entity_id"] is None
    rows = payload["table"]["rows"]
    assert [row["team"] for row in rows] == ["BUF", "KC", "NE", "WSH"]
    assert len(rows[0]["team_rank_probabilities"]) == 4


@pytest.mark.parametrize("kind", ["teams", "qbs"])
def test_rank_ranges_name_the_teams_they_list(
    ratings_dir: Path, viewer_client: TestClient, kind: str
) -> None:
    """The rank-range chart labels its rows without fetching the whole team table."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/{kind}/rank-ranges").json()

    # Assert
    assert payload["team_names"] == {
        "BUF": "Buffalo Bills",
        "KC": "Kansas City Chiefs",
        "NE": "New England Patriots",
        "WSH": "Washington Commanders",
    }


def test_qb_rank_ranges_list_the_qualifiers(ratings_dir: Path, viewer_client: TestClient) -> None:
    """Only qualifying quarterbacks have rank ranges."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/qbs/rank-ranges").json()

    # Assert
    assert [row["qb_rank"] for row in payload["table"]["rows"]] == [1, 2, 3, 4]


def test_a_season_built_without_rank_ranges_is_a_404(
    project_root: Path, viewer_client: TestClient
) -> None:
    """A page hides its rank ranges when the season has none."""
    # Arrange
    factories.write_season(project_root / "data" / "ratings", SEASON, kinds=factories.INDEX_KINDS)

    # Act
    response = viewer_client.get(f"/api/ratings/{SEASON}/teams/rank-ranges")

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ratings_table"


def test_rank_history_starts_once_every_team_has_three_games(
    project_root: Path, viewer_client: TestClient
) -> None:
    """The weekly rank ranges of a season in progress start at week 3 of the factory season."""
    # Arrange
    factories.write_season(
        project_root / "data" / "ratings",
        2026,
        games_played=6,
        kinds=(*factories.ALL_KINDS, *factories.IN_PROGRESS_KINDS),
    )

    # Act
    payload = viewer_client.get("/api/ratings/2026/teams/KC/rank-history").json()

    # Assert
    assert [row["week"] for row in payload["table"]["rows"]] == [3, 4, 5, 6]


def test_qb_rank_history_uses_the_team_weeks_for_its_start(
    project_root: Path, viewer_client: TestClient
) -> None:
    """Quarterbacks share the teams' starting week."""
    # Arrange
    factories.write_season(
        project_root / "data" / "ratings",
        2026,
        games_played=6,
        kinds=(*factories.ALL_KINDS, *factories.IN_PROGRESS_KINDS),
    )

    # Act
    payload = viewer_client.get(f"/api/ratings/2026/qbs/{KC_QB}/rank-history").json()

    # Assert
    assert [row["week"] for row in payload["table"]["rows"]] == [3, 4, 5, 6]


def test_completed_seasons_have_no_rank_history(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """Weekly rank ranges exist only for the season in progress."""
    # Act
    response = viewer_client.get(f"/api/ratings/{SEASON}/teams/KC/rank-history")

    # Assert
    assert response.status_code == 404


def test_head_to_head_lists_a_teams_chances_against_each_other_team(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """One team's chance of being rated above every other team."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/teams/BUF/head-to-head").json()

    # Assert
    rows = payload["table"]["rows"]
    assert [row["other_team"] for row in rows] == ["KC", "NE", "WSH"]
    assert {row["team_rated_above_probability"] for row in rows} == {0.8}


def test_qb_head_to_head_lists_a_passers_chances(
    ratings_dir: Path, viewer_client: TestClient
) -> None:
    """One quarterback's chance of being rated above every other qualifier."""
    # Act
    payload = viewer_client.get(f"/api/ratings/{SEASON}/qbs/{KC_QB}/head-to-head").json()

    # Assert
    assert len(payload["table"]["rows"]) == 3
    assert payload["table"]["column_groups"]["chance"] == [
        "qb_rated_above_probability",
        "qb_pair_share",
    ]


@pytest.mark.parametrize(
    ("kind", "path"),
    [
        ("rating_pairs", f"/api/ratings/{SEASON}/teams/BUF/head-to-head"),
        ("combined", f"/api/ratings/{SEASON}/teams"),
        ("ratings", f"/api/ratings/{SEASON}/teams/BUF/game-logs"),
        ("ratings", "/api/ratings/schedules"),
    ],
)
def test_a_broken_table_is_a_503_that_names_it(
    ratings_dir: Path, viewer_client: TestClient, kind: str, path: str
) -> None:
    """A table left empty or cut short says which file to rebuild instead of failing blindly."""
    # Arrange
    table = ratings_dir / f"{SEASON}_{kind}.parquet"
    table.write_bytes(table.read_bytes()[:100])

    # Act
    response = viewer_client.get(path)

    # Assert
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "ratings_table_unreadable"
    assert f"{SEASON}_{kind}.parquet" in error["message"]


def test_schedules_rank_every_completed_team_season(
    project_root: Path, viewer_client: TestClient
) -> None:
    """The leaderboard spans every completed season and leaves out the one in progress."""
    # Arrange
    ratings_dir = project_root / "data" / "ratings"
    factories.write_season(ratings_dir, 2024)
    factories.write_season(ratings_dir, 2025)
    factories.write_season(ratings_dir, 2026, games_played=5)

    # Act
    payload = viewer_client.get("/api/ratings/schedules").json()

    # Assert
    assert (payload["count"], payload["seasons"], payload["skipped_seasons"]) == (
        8,
        [2024, 2025],
        [2026],
    )
    assert payload["table"]["rows"][0]["softest_rank"] == 1


def test_metrics_serve_the_glossary_registry(viewer_client: TestClient) -> None:
    """The glossary reads the registry, which needs no season data."""
    # Act
    payload = viewer_client.get("/api/ratings/metrics").json()

    # Assert
    assert payload["metrics"]["sos"]["contextual"] is True
    assert payload["entities"]["qb"]["categories"][0]["name"] == "Schedule-Adjusted Ratings"


def test_ratings_follow_the_configured_data_directory(
    tmp_path: Path, app: FastAPI, viewer_client: TestClient
) -> None:
    """The routes read ``ratings/`` under whatever data directory the settings name."""
    # Arrange
    other_data = tmp_path / "elsewhere"
    factories.write_season(other_data / "ratings", 2023)
    app.state.settings.data_dir = other_data

    # Act
    payload = viewer_client.get("/api/ratings/seasons").json()

    # Assert
    assert payload["seasons"] == [2023]
