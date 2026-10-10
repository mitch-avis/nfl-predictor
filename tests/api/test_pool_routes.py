"""Route tests for the pool helpers and the confidence picks CSV export."""

from __future__ import annotations

import csv
import io
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import polars as pl
import pytest
from tests.api import factories

from nfl_predictor.api.readers import cache

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

START_WEEK = 5


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Start every test with an empty artifact cache."""
    cache.clear()


def _seed(project_root: Path, app: FastAPI, *later_weeks: int) -> Path:
    """Create the active week-5 weekly run, with later weeks predicted into it."""
    run_dir = factories.make_run_dir(
        project_root / "models", "weekly_w5", factories.RunSpec(season=2026, week=START_WEEK)
    )
    for week in later_weeks:
        factories.write_predictions_csv(
            run_dir / f"season_2026_week_{week:02d}_predictions.csv", 2026, week
        )
    app.state.run_index.invalidate()
    return run_dir


def _survivor(client: TestClient, **body: Any) -> Any:
    """POST a survivor request for 2026 from week 5 with ``body`` overriding the defaults."""
    payload = {"season": 2026, "start_week": START_WEEK, **body}
    return client.post("/api/pool/survivor", json=payload)


def test_confidence_sheet_lists_the_weeks_picks(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The sheet reads the active run's week: four games, ten points on offer."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/pool/confidence").json()

    # Assert
    assert (payload["run_id"], payload["season"], payload["week"]) == ("weekly_w5", 2026, 5)
    assert (payload["games"], payload["max_points"]) == (4, 10)
    assert [(row["points"], row["pick"]) for row in payload["rows"]] == [
        (4, "NE"),
        (3, "GB"),
        (2, "BUF"),
        (1, "DEN"),
    ]
    assert payload["rank_problem"] is None
    assert payload["lock_at"] == "2026-09-13T13:00:00-04:00"


def test_confidence_sheet_scores_finals_from_the_dataset(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Finals in the ETL's dataset score the sheet, a tie counting for neither side."""
    # Arrange
    _seed(project_root, app)
    (project_root / "data" / "all_data.csv").write_text(
        "game_id,away_score,home_score\n2026_05_NYJ_NE,10,20\n2026_05_GB_MIN,17,17\n",
        encoding="utf-8",
    )

    # Act
    payload = viewer_client.get("/api/pool/confidence").json()

    # Assert
    results = {row["pick"]: (row["result"], row["points_won"]) for row in payload["rows"]}
    assert results == {"NE": ("win", 4), "GB": ("tie", 0), "BUF": (None, None), "DEN": (None, None)}
    assert (payload["games_final"], payload["points_won"], payload["points_possible"]) == (2, 4, 7)


def _repredict_after_the_lock(run_dir: Path, *, picks_before_lock: bool) -> None:
    """Rewrite week 5's predictions with reversed ranks after the first kickoff (Sept. 13).

    The weekly run's picks file is dated the day before the lock, or the day after it.
    """
    predictions = run_dir / "season_2026_week_05_predictions.csv"
    frame = pl.read_csv(predictions, infer_schema_length=10000)
    frame.with_columns((5 - pl.col("confidence_rank")).alias("confidence_rank")).write_csv(
        predictions
    )
    after = datetime(2026, 9, 15, tzinfo=UTC).timestamp()
    picks_day = datetime(2026, 9, 12 if picks_before_lock else 14, tzinfo=UTC).timestamp()
    os.utime(predictions, (after, after))
    os.utime(run_dir / "season_2026_week_05_confidence_picks.csv", (picks_day, picks_day))


def test_confidence_sheet_after_the_lock_reads_the_picks_made_before_it(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A re-predict after the first kickoff leaves the sheet on the picks entered before it."""
    # Arrange
    run_dir = _seed(project_root, app)
    _repredict_after_the_lock(run_dir, picks_before_lock=True)

    # Act
    payload = viewer_client.get("/api/pool/confidence").json()

    # Assert
    assert [(row["points"], row["pick"], row["newer_points"]) for row in payload["rows"]] == [
        (4, "NE", 1),
        (3, "GB", 2),
        (2, "BUF", 3),
        (1, "DEN", 4),
    ]
    assert payload["picks_file"] == "season_2026_week_05_confidence_picks.csv"
    assert (payload["generated_after_lock"], payload["changed_after_lock"]) == (False, 4)
    assert payload["generated_at"].startswith("2026-09-12")


def test_confidence_sheet_after_the_lock_without_earlier_picks_warns(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """With no picks written before the lock, the sheet shows the newer ranks and warns."""
    # Arrange
    run_dir = _seed(project_root, app)
    _repredict_after_the_lock(run_dir, picks_before_lock=False)

    # Act
    payload = viewer_client.get("/api/pool/confidence").json()

    # Assert
    assert [row["points"] for row in payload["rows"]] == [4, 3, 2, 1]
    assert [row["pick"] for row in payload["rows"]] == ["DEN", "BUF", "GB", "NE"]
    assert (payload["picks_file"], payload["generated_after_lock"]) == (None, True)


def test_confidence_sheet_for_a_week_without_a_picks_file_keeps_its_predictions(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A week predicted ahead has no picks file, so after the lock it keeps its ranks and warns."""
    # Arrange
    _seed(project_root, app, 6)

    # Act
    payload = viewer_client.get("/api/pool/confidence", params={"season": 2026, "week": 6}).json()

    # Assert
    assert payload["week"] == 6
    assert {row["game_id"][:7] for row in payload["rows"]} == {"2026_06"}
    assert (payload["picks_file"], payload["generated_after_lock"]) == (None, True)


def test_picks_after_the_lock_follow_the_picks_made_before_it(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The Predictions page's picks and the CSV download keep the pre-lock picks too."""
    # Arrange
    run_dir = _seed(project_root, app)
    _repredict_after_the_lock(run_dir, picks_before_lock=True)

    # Act
    payload = viewer_client.get("/api/predictions/picks").json()

    # Assert
    picks = [(row["confidence_rank"], row["predicted_winner"]) for row in payload["table"]["rows"]]
    assert picks == [(4, "NE"), (3, "GB"), (2, "BUF"), (1, "DEN")]


def test_confidence_sheet_without_predictions_is_a_404(viewer_client: TestClient) -> None:
    """With nothing predicted the sheet says so, like the predictions page."""
    # Act
    response = viewer_client.get("/api/pool/confidence")

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_pool_routes_need_a_session(client: TestClient) -> None:
    """The pool helpers are for signed-in users only."""
    # Act
    response = client.get("/api/pool/confidence")

    # Assert
    assert response.status_code == 401


def test_picks_export_as_csv(project_root: Path, viewer_client: TestClient, app: FastAPI) -> None:
    """``format=csv`` downloads the picks table, most confident first."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = viewer_client.get("/api/predictions/picks", params={"format": "csv"})

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == (
        'attachment; filename="season_2026_week_05_confidence_picks.csv"'
    )
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert [row["confidence_rank"] for row in rows] == ["4", "3", "2", "1"]
    assert [row["predicted_winner"] for row in rows] == ["NE", "GB", "BUF", "DEN"]


def test_picks_export_follows_an_unattached_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The download takes the same week the sheet shows, an unattached file included."""
    # Arrange
    _seed(project_root, app)
    factories.write_predictions_csv(
        project_root / "data" / "predict" / "week_05_predictions.csv", 2026, START_WEEK
    )
    path = project_root / "data" / "predict" / "week_05_predictions.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("2026_05_", "loose_"), "utf-8")

    # Act
    response = viewer_client.get(
        "/api/predictions/picks",
        params={"season": 2026, "week": START_WEEK, "source": "unattached", "format": "csv"},
    )

    # Assert
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert {row["game_id"][:6] for row in rows} == {"loose_"}


def test_picks_export_rejects_an_unknown_format(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Only JSON and CSV are offered."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = viewer_client.get("/api/predictions/picks", params={"format": "xlsx"})

    # Assert
    assert response.status_code == 422


def test_tiebreakers_suggest_the_sunday_and_monday_games(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The card lists the week's games and suggests the latest Sunday and Monday kickoffs."""
    # Arrange
    _seed(project_root, app)

    # Act
    payload = viewer_client.get("/api/pool/tiebreakers").json()

    # Assert
    assert payload["week"] == START_WEEK
    assert len(payload["games"]) == 4
    assert payload["default_game_ids"] == ["2026_05_NYJ_NE", "2026_05_DEN_KC"]
    assert payload["highest"]["points"] > payload["lowest"]["points"]


def test_survivor_plans_the_weeks_with_predictions(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Weeks 5 to 7 have predictions: week 5 is planned, 6 and 7 are covered by the lookahead."""
    # Arrange
    _seed(project_root, app, 6, 7)

    # Act
    response = _survivor(viewer_client, horizon_weeks=1)

    # Assert
    assert response.status_code == 200
    payload = response.json()
    assert (payload["planned_weeks"], payload["later_weeks"]) == ([5], [6, 7])
    assert payload["missing_weeks"] == list(range(8, 19))
    best = payload["plans"][0]
    assert best["rank"] == 1
    assert [(pick["week"], pick["team"], pick["later"]) for pick in best["picks"]] == [
        (5, "NE", False),
        (6, "BUF", True),
        (7, "GB", True),
    ]
    assert best["total_survival"] == pytest.approx(0.71 * 0.6678 * 0.63)
    assert len(payload["teams"]) == 32


def test_survivor_leaves_out_used_teams(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Teams already used are normalized and never planned again."""
    # Arrange
    _seed(project_root, app, 6)

    # Act
    payload = _survivor(viewer_client, used_teams=["ne", "BUF"]).json()

    # Assert
    assert payload["used_teams"] == ["BUF", "NE"]
    planned = {pick["team"] for plan in payload["plans"] for pick in plan["picks"]}
    assert planned.isdisjoint({"NE", "BUF"})


def test_survivor_rejects_an_unknown_team(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A used team that is not an NFL team is a request error."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = _survivor(viewer_client, used_teams=["XYZ"])

    # Assert
    assert response.status_code == 422
    assert "XYZ" in response.json()["error"]["message"]


def test_survivor_without_predictions_for_the_start_week_is_a_404(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The week to pick now must have predictions."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = _survivor(viewer_client, start_week=9)

    # Assert
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_predictions"


def test_survivor_stops_at_the_end_of_the_regular_season(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """Survivor pools run through the regular season, so a playoff week is a request error."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = _survivor(viewer_client, start_week=19)

    # Assert
    assert response.status_code == 422


def test_survivor_rejects_more_used_teams_than_the_league_has(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """A list of used teams longer than the league's 32 teams is a request error."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = _survivor(viewer_client, used_teams=["KC"] * 33)

    # Assert
    assert response.status_code == 422


def test_survivor_needs_at_least_one_planned_week(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The week to pick now is always planned, so zero planned weeks is a request error."""
    # Arrange
    _seed(project_root, app)

    # Act
    response = _survivor(viewer_client, horizon_weeks=0)

    # Assert
    assert response.status_code == 422


def test_survivor_shows_every_week_as_planned_with_a_long_horizon(
    project_root: Path, viewer_client: TestClient, app: FastAPI
) -> None:
    """The planned weeks are only a display split, so any number of them is accepted."""
    # Arrange
    _seed(project_root, app, 6, 7)

    # Act
    payload = _survivor(viewer_client, horizon_weeks=18).json()

    # Assert
    assert (payload["planned_weeks"], payload["later_weeks"]) == ([5, 6, 7], [])
    assert all(not pick["later"] for pick in payload["plans"][0]["picks"])
