"""Tests for scraping utilities (HTML parsing and SurvivorGrid spreads)."""

import datetime as dt
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from bs4 import BeautifulSoup, Tag

from nfl_predictor import constants
from nfl_predictor.utils import clock, scraping_utils


class DummyResponse:
    """Minimal requests-like response for monkeypatched HTML downloads."""

    def __init__(self, text: str) -> None:
        """Store HTML payload text for the response double."""
        self.text = text
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        """No-op for dummy response."""
        return


def test_parse_tr_rating_table() -> None:
    """TeamRankings rating table parsing returns canonical team abbr + float rating."""
    # Arrange
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
        <tr><td>1</td><td>Buffalo Bills (10-4)</td><td>6.7</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    assert isinstance(table, Tag)

    # Act
    teams, ratings = scraping_utils._parse_tr_rating_table(table)

    # Assert
    assert teams == ["BUF"]
    assert ratings == [6.7]


def test_parse_tr_stat_table() -> None:
    """TeamRankings stat table parsing returns canonical team abbr + numeric stat."""
    # Arrange
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Kansas City Chiefs (11-3)</td><td>52.5%</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    assert isinstance(table, Tag)

    # Act
    teams, stats = scraping_utils._parse_tr_stat_table(table)

    # Assert
    assert teams == ["KC"]
    assert stats == [52.5]


def test_parse_tr_tables_default_invalid_values_to_zero() -> None:
    """Invalid numeric rating cells fall back to `0.0` during table parsing."""
    # Arrange
    rating_html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>bad</td></tr>
    </table>
    """
    rating_table = BeautifulSoup(rating_html, "html.parser").find("table")
    assert isinstance(rating_table, Tag)

    # Act
    _teams, ratings = scraping_utils._parse_tr_rating_table(rating_table)

    # Assert
    assert ratings == [0.0]


def test_parse_tr_stat_table_defaults_invalid_values_to_zero() -> None:
    """Invalid numeric stat cells fall back to `0.0` during table parsing."""
    # Arrange
    stat_html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Kansas City Chiefs</td><td>bad%</td></tr>
    </table>
    """
    stat_table = BeautifulSoup(stat_html, "html.parser").find("table")
    assert isinstance(stat_table, Tag)

    # Act
    _teams, stats = scraping_utils._parse_tr_stat_table(stat_table)

    # Assert
    assert stats == [0.0]


def test_scrape_survivor_grid_spreads_parses(monkeypatch) -> None:
    """SurvivorGrid spread table parsing extracts per-week spreads."""
    # Arrange
    rows = [
        """
        <tr>
            <td>BUF(10-4)</td>
            <td>@KC-3.5</td>
            <td>MIA+2</td>
        </tr>
        """
    ]
    rows.extend(f"<tr><td>XXX{i}</td><td>BYE</td><td>BYE</td></tr>" for i in range(30))
    html = f"""
    <html>
        <body>
            <table>
                <tr><th>Team</th><th>16</th><th>17</th></tr>
                {"".join(rows)}
            </table>
        </body>
    </html>
    """

    def fake_get(*_args, **_kwargs):
        return DummyResponse(html)

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads["BUF"][16] == -3.5
    assert spreads["BUF"][17] == 2.0


def test_get_season_start_and_week_date() -> None:
    """Season start calculations are correct."""
    # Act
    start = scraping_utils.get_season_start(2023)

    # Assert
    assert start == date(2023, 9, 7)


def test_get_week_date() -> None:
    """Week date calculations are correct."""
    # Act
    week_date = scraping_utils.get_week_date(2023, 1)

    # Assert
    assert week_date == date(2023, 9, 6)


def test_get_current_nfl_week_handles_preseason_and_clamp(monkeypatch) -> None:
    """Pre-kickoff dates resolve to week 1."""
    # Arrange
    # Today is an August preseason date before kickoff.
    monkeypatch.setattr(clock, "local_today", lambda: dt.date(2025, 8, 15))

    # Act
    preseason_season, preseason_week = scraping_utils.get_current_nfl_week()

    # Assert
    assert preseason_season == 2025
    assert preseason_week == 1


def test_get_current_nfl_week_clamps_late_february_to_the_max_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late-February dates clamp to max week."""
    # Arrange
    # Today is a date far enough after kickoff to require clamping.
    monkeypatch.setattr(clock, "local_today", lambda: dt.date(2026, 2, 28))

    # Act
    late_season, late_week = scraping_utils.get_current_nfl_week()

    # Assert
    assert late_season == 2025
    assert late_week == constants.get_regular_season_weeks(2025) + 4


def test_normalize_team_column() -> None:
    """Team column normalization maps known variants to canonical abbreviations."""
    # Arrange
    df = pl.DataFrame({"team": ["JAC"]})

    # Act
    out = scraping_utils.normalize_team_column(df, "team")

    # Assert
    assert out["team"][0] == "JAX"


def test_get_missing_tr_columns_and_merge() -> None:
    """Missing TR columns are identified correctly."""
    # Arrange
    existing = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})

    # Act
    missing_ratings, missing_stats = scraping_utils.get_missing_tr_columns(existing)

    # Assert
    assert missing_ratings
    assert missing_stats


def test_merge_tr_data_adds_the_new_columns() -> None:
    """New TR columns are merged into the existing data."""
    # Arrange
    existing = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})
    new = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "third_down_pct": [0.5]})

    # Act
    merged = scraping_utils.merge_tr_data(existing, new)

    # Assert
    assert "third_down_pct" in merged.columns


_EXISTING_TR = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})
_NEW_TR = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [2.0]})


@pytest.mark.parametrize(
    ("existing", "new", "expected"),
    [
        pytest.param(pl.DataFrame(), _NEW_TR, _NEW_TR, id="empty_existing"),
        pytest.param(_EXISTING_TR, pl.DataFrame(), _EXISTING_TR, id="empty_new"),
        pytest.param(_EXISTING_TR, _NEW_TR, _EXISTING_TR, id="no_new_columns"),
    ],
)
def test_merge_tr_data_passthrough_cases(
    existing: pl.DataFrame, new: pl.DataFrame, expected: pl.DataFrame
) -> None:
    """TR merge returns the non-empty or unchanged side when no new columns exist."""
    # Act
    merged = scraping_utils.merge_tr_data(existing, new)

    # Assert
    assert merged.to_dicts() == expected.to_dicts()


def test_save_and_update_team_rankings(tmp_path, monkeypatch) -> None:
    """TeamRankings weekly data is saved and season update combines correctly."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    df = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})

    scraping_utils.save_team_rankings_week(df, season=2023, week=1)

    # Act
    scraping_utils.update_season_team_rankings(2023)

    # Assert
    combined_path = tmp_path / "2023" / "2023_team_rankings.csv"
    assert combined_path.exists()


def test_scrape_team_rankings_for_week(monkeypatch) -> None:
    """TeamRankings ratings and stats are scraped correctly from HTML."""
    # Arrange
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>6.7</td></tr>
        <tr><td>2</td><td>Miami Dolphins</td><td>5.1</td></tr>
    </table>
    """

    def fake_get(*_args, **_kwargs):
        """Return dummy HTML response."""
        return DummyResponse(html)

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={"stat": "third_down_pct"},
    )

    # Assert
    assert df.height == 2
    assert "predictive_rating" in df.columns
    assert "third_down_pct" in df.columns


def test_scrape_team_rankings_for_week_uses_default_sources(monkeypatch) -> None:
    """Omitted rating/stat maps fall back to the default TeamRankings source lists."""
    # Arrange
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>6.7</td></tr>
    </table>
    """

    monkeypatch.setattr(constants, "TEAM_RANKINGS_RATINGS", {"rating": "predictive_rating"})
    monkeypatch.setattr(constants, "TEAM_RANKINGS_STATS", {"stat": "third_down_pct"})
    monkeypatch.setattr(
        scraping_utils.requests,
        "get",
        lambda *_args, **_kwargs: DummyResponse(html),
    )
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(1, date(2023, 9, 6))

    # Assert
    assert df.height == 1
    assert "predictive_rating" in df.columns
    assert "third_down_pct" in df.columns


def test_scrape_team_rankings_for_week_handles_request_errors(monkeypatch) -> None:
    """Request failures produce an empty scrape result instead of raising."""

    # Arrange
    def fake_get(*_args, **_kwargs):
        """Raise a requests-layer failure for every scrape attempt."""
        msg = "boom"
        raise scraping_utils.requests.RequestException(msg)

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={},
    )

    # Assert
    assert df.height == 0


@pytest.mark.parametrize(
    ("parser_name", "ratings_to_scrape", "stats_to_scrape"),
    [
        ("_parse_tr_rating_table", {"rating": "predictive_rating"}, {}),
        ("_parse_tr_stat_table", {}, {"stat": "third_down_pct"}),
    ],
)
def test_scrape_team_rankings_for_week_handles_parse_errors(
    monkeypatch,
    parser_name: str,
    ratings_to_scrape: dict[str, str],
    stats_to_scrape: dict[str, str],
) -> None:
    """Parser exceptions are swallowed and converted into an empty scrape result."""
    # Arrange
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>6.7</td></tr>
    </table>
    """

    monkeypatch.setattr(
        scraping_utils.requests,
        "get",
        lambda *_args, **_kwargs: DummyResponse(html),
    )
    monkeypatch.setattr(
        scraping_utils, parser_name, lambda _table: (_ for _ in ()).throw(ValueError("bad parse"))
    )
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape=ratings_to_scrape,
        stats_to_scrape=stats_to_scrape,
    )

    # Assert
    assert df.height == 0


@pytest.mark.parametrize(
    ("ratings_to_scrape", "stats_to_scrape", "parser"),
    [
        ({"rating": "predictive_rating"}, {}, "_parse_tr_rating_table"),
        ({}, {"stat": "third_down_pct"}, "_parse_tr_stat_table"),
    ],
)
def test_a_parse_length_mismatch_truncates_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    ratings_to_scrape: dict[str, str],
    stats_to_scrape: dict[str, str],
    parser: str,
) -> None:
    """More teams than values keeps the matched pairs and says so, not "failed to parse"."""
    # Arrange
    warnings: list[str] = []
    monkeypatch.setattr(
        scraping_utils.log, "warning", lambda message, *args: warnings.append(message % args)
    )
    monkeypatch.setattr(
        scraping_utils.requests,
        "get",
        lambda *_args, **_kwargs: DummyResponse("<table><tr><td>x</td></tr></table>"),
    )
    monkeypatch.setattr(scraping_utils, parser, lambda _table: (["BUF", "KC"], [6.7]))
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape=ratings_to_scrape,
        stats_to_scrape=stats_to_scrape,
    )

    # Assert
    assert df["team_abbr"].to_list() == ["BUF"]
    assert any("length mismatch" in warning for warning in warnings)
    assert not any("Failed to parse" in warning for warning in warnings)


def test_scrape_team_rankings_for_week_handles_missing_tables(monkeypatch) -> None:
    """Missing HTML tables produce an empty TeamRankings result."""

    # Arrange
    def fake_get(*_args, **_kwargs):
        """Return HTML without any table content."""
        return DummyResponse("<html><body><div>No table</div></body></html>")

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    # Act
    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={},
    )

    # Assert
    assert df.height == 0


def test_merge_tr_data_without_week_column() -> None:
    """TR merge falls back to team-only joins when week is absent."""
    # Arrange
    existing = pl.DataFrame({"team_abbr": ["AAA"], "predictive_rating": [1.0]})
    new = pl.DataFrame({"team_abbr": ["AAA"], "third_down_pct": [0.5]})

    # Act
    merged = scraping_utils.merge_tr_data(existing, new)

    # Assert
    assert merged.to_dicts() == [
        {"team_abbr": "AAA", "predictive_rating": 1.0, "third_down_pct": 0.5}
    ]


def test_update_season_team_rankings_skips_empty_week_files(tmp_path, monkeypatch) -> None:
    """Empty week CSVs do not produce a consolidated season file."""
    # Arrange
    monkeypatch.setattr(constants, "DATA_PATH", tmp_path)
    season_dir = Path(tmp_path) / "2023"
    season_dir.mkdir(parents=True, exist_ok=True)
    (season_dir / "2023_week_01_team_rankings.csv").write_text("team_abbr,week\n", encoding="utf-8")

    # Act
    scraping_utils.update_season_team_rankings(2023)

    # Assert
    assert not (season_dir / "2023_team_rankings.csv").exists()


def test_scrape_survivor_grid_spreads_handles_request_and_header_failures(monkeypatch) -> None:
    """SurvivorGrid scraping returns empty data on request failures."""

    # Arrange
    def failing_get(*_args, **_kwargs):
        """Raise a request-layer failure."""
        msg = "boom"
        raise scraping_utils.requests.RequestException(msg)

    monkeypatch.setattr(scraping_utils.requests, "get", failing_get)

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads == {}


def test_scrape_survivor_grid_spreads_handles_header_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SurvivorGrid scraping returns empty data on header-parse failures."""

    # Arrange
    def headerless_get(*_args, **_kwargs):
        """Return a table that lacks the required Team/week headers."""
        rows = "".join("<tr><td>BUF</td><td>PK</td></tr>" for _ in range(32))
        return DummyResponse(f"<table><tr><th>Club</th><th>Spread</th></tr>{rows}</table>")

    monkeypatch.setattr(scraping_utils.requests, "get", headerless_get)

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads == {}


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><div>No table</div></body></html>",
        "<table><tr><th>Team</th><th>16</th></tr><tr><td>BUF</td><td>PK</td></tr></table>",
    ],
)
def test_scrape_survivor_grid_spreads_handles_missing_data_table(monkeypatch, html: str) -> None:
    """Pages with no usable SurvivorGrid table return an empty spread mapping."""
    # Arrange
    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads == {}


def test_scrape_survivor_grid_spreads_requires_team_column(monkeypatch) -> None:
    """A SurvivorGrid table without the Team header is rejected."""
    # Arrange
    rows = "".join("<tr><td>BUF</td><td>PK</td></tr>" for _ in range(32))
    html = f"<table><tr><th>Club</th><th>16</th></tr>{rows}</table>"

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads == {}


def test_scrape_survivor_grid_spreads_parses_pickem_cells(monkeypatch) -> None:
    """Pick'em cells are normalized to a zero spread."""
    # Arrange
    rows = ["<tr><td>BUF</td><td>PK</td></tr>"]
    rows.extend(f"<tr><td>XXX{i}</td><td>BYE</td></tr>" for i in range(31))
    html = f"<table><tr><th>Team</th><th>16</th></tr>{''.join(rows)}</table>"

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads["BUF"][16] == pytest.approx(0.0)


def test_scrape_survivor_grid_spreads_handles_row_edge_cases(monkeypatch) -> None:
    """Short rows, invalid teams, and unmatched spread cells are skipped safely."""
    # Arrange
    rows = [
        "<tr><td>SHORT</td></tr>",
        "<tr><td>XXX</td><td>@KC-3.5</td><td>@KC-3.5</td></tr>",
        "<tr><td>BUF</td><td>TBD</td><td>@KC-3.5</td></tr>",
    ]
    rows.extend(f"<tr><td>YYY{i}</td><td>BYE</td><td>BYE</td></tr>" for i in range(29))
    html = f"<table><tr><th>Team</th><th>16</th><th>17</th></tr>{''.join(rows)}</table>"

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert 16 not in spreads.get("BUF", {})
    assert spreads["BUF"][17] == pytest.approx(-3.5)


def test_scrape_survivor_grid_spreads_handles_float_conversion_failure(monkeypatch) -> None:
    """Numeric spread conversion failures are skipped instead of aborting the row."""
    # Arrange
    rows = ["<tr><td>BUF</td><td>@KC-3.5</td></tr>"]
    rows.extend(f"<tr><td>ZZZ{i}</td><td>BYE</td></tr>" for i in range(31))
    html = f"<table><tr><th>Team</th><th>16</th></tr>{''.join(rows)}</table>"

    def fake_float(_text: str) -> float:
        """Force the spread parser down its ValueError handling branch."""
        msg = "bad float"
        raise ValueError(msg)

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )
    monkeypatch.setattr(scraping_utils, "float", fake_float, raising=False)

    # Act
    spreads = scraping_utils.scrape_survivor_grid_spreads()

    # Assert
    assert spreads == {}
