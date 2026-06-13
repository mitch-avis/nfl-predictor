"""Tests for scraping utilities (HTML parsing and SurvivorGrid spreads)."""

import builtins
import datetime
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from bs4 import BeautifulSoup

from nfl_predictor import constants
from nfl_predictor.utils import scraping_utils


class DummyResponse:
    """Minimal requests-like response for monkeypatched HTML downloads."""

    def __init__(self, text: str) -> None:
        """Store HTML payload text for the response double."""
        self.text = text
        self.content = text.encode("utf-8")

    def raise_for_status(self) -> None:
        """No-op for dummy response."""
        return None


def test_parse_tr_rating_table() -> None:
    """TeamRankings rating table parsing returns canonical team abbr + float rating."""
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
        <tr><td>1</td><td>Buffalo Bills (10-4)</td><td>6.7</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    teams, ratings = scraping_utils._parse_tr_rating_table(table)

    assert teams == ["BUF"]
    assert ratings == [6.7]


def test_parse_tr_stat_table() -> None:
    """TeamRankings stat table parsing returns canonical team abbr + numeric stat."""
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Kansas City Chiefs (11-3)</td><td>52.5%</td></tr>
    </table>
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    teams, stats = scraping_utils._parse_tr_stat_table(table)

    assert teams == ["KC"]
    assert stats == [52.5]


def test_parse_tr_tables_default_invalid_values_to_zero() -> None:
    """Invalid numeric cells fall back to `0.0` during table parsing."""
    rating_html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Rating</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>bad</td></tr>
    </table>
    """
    stat_html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Kansas City Chiefs</td><td>bad%</td></tr>
    </table>
    """

    rating_table = BeautifulSoup(rating_html, "html.parser").find("table")
    stat_table = BeautifulSoup(stat_html, "html.parser").find("table")

    assert scraping_utils._parse_tr_rating_table(rating_table)[1] == [0.0]
    assert scraping_utils._parse_tr_stat_table(stat_table)[1] == [0.0]


def test_scrape_survivor_grid_spreads_parses(monkeypatch) -> None:
    """SurvivorGrid spread table parsing extracts per-week spreads."""
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

    spreads = scraping_utils.scrape_survivor_grid_spreads()
    assert spreads["BUF"][16] == -3.5
    assert spreads["BUF"][17] == 2.0


def test_get_season_start_and_week_date() -> None:
    """Season start and week date calculations are correct."""
    start = scraping_utils.get_season_start(2023)
    assert start == date(2023, 9, 7)

    week_date = scraping_utils.get_week_date(2023, 1)
    assert week_date == date(2023, 9, 6)


def test_get_current_nfl_week_handles_preseason_and_clamp(monkeypatch) -> None:
    """Preseason dates return the prior season; late-February dates clamp to max week."""

    class _PreseasonDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            """Return an August preseason date before kickoff."""
            return cls(2025, 8, 15)

    monkeypatch.setattr(scraping_utils, "date", _PreseasonDate)
    preseason_season, preseason_week = scraping_utils.get_current_nfl_week()

    assert preseason_season == 2024
    assert preseason_week == constants.get_regular_season_weeks(2024) + 4

    class _LateFebruaryDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            """Return a date far enough after kickoff to require clamping."""
            return cls(2026, 2, 28)

    monkeypatch.setattr(scraping_utils, "date", _LateFebruaryDate)
    late_season, late_week = scraping_utils.get_current_nfl_week()

    assert late_season == 2025
    assert late_week == constants.get_regular_season_weeks(2025) + 4


def test_normalize_team_column() -> None:
    """Team column normalization maps known variants to canonical abbreviations."""
    df = pl.DataFrame({"team": ["JAC"]})
    out = scraping_utils.normalize_team_column(df, "team")
    assert out["team"][0] == "JAX"


def test_get_missing_tr_columns_and_merge() -> None:
    """Missing TR columns are identified and merged correctly."""
    existing = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})
    missing_ratings, missing_stats = scraping_utils.get_missing_tr_columns(existing)
    assert missing_ratings
    assert missing_stats

    new = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "third_down_pct": [0.5]})
    merged = scraping_utils.merge_tr_data(existing, new)
    assert "third_down_pct" in merged.columns


def test_merge_tr_data_passthrough_cases() -> None:
    """TR merge returns the non-empty or unchanged side when no new columns exist."""
    existing = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})
    new = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [2.0]})

    assert scraping_utils.merge_tr_data(pl.DataFrame(), new).to_dicts() == new.to_dicts()
    assert scraping_utils.merge_tr_data(existing, pl.DataFrame()).to_dicts() == existing.to_dicts()
    assert scraping_utils.merge_tr_data(existing, new).to_dicts() == existing.to_dicts()


def test_save_and_update_team_rankings(tmp_path, monkeypatch) -> None:
    """TeamRankings weekly data is saved and season update combines correctly."""
    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    df = pl.DataFrame({"team_abbr": ["AAA"], "week": [1], "predictive_rating": [1.0]})

    scraping_utils.save_team_rankings_week(df, season=2023, week=1)
    scraping_utils.update_season_team_rankings(2023)

    combined_path = tmp_path / "2023" / "2023_team_rankings.csv"
    assert combined_path.exists()


def test_scrape_team_rankings_for_week(monkeypatch) -> None:
    """TeamRankings ratings and stats are scraped correctly from HTML."""
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

    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={"stat": "third_down_pct"},
    )

    assert df.height == 2
    assert "predictive_rating" in df.columns
    assert "third_down_pct" in df.columns


def test_scrape_team_rankings_for_week_uses_default_sources(monkeypatch) -> None:
    """Omitted rating/stat maps fall back to the default TeamRankings source lists."""
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

    df = scraping_utils.scrape_team_rankings_for_week(1, date(2023, 9, 6))

    assert df.height == 1
    assert "predictive_rating" in df.columns
    assert "third_down_pct" in df.columns


def test_scrape_team_rankings_for_week_handles_request_errors(monkeypatch) -> None:
    """Request failures produce an empty scrape result instead of raising."""

    def fake_get(*_args, **_kwargs):
        """Raise a requests-layer failure for every scrape attempt."""
        raise scraping_utils.requests.RequestException("boom")

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={},
    )

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

    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape=ratings_to_scrape,
        stats_to_scrape=stats_to_scrape,
    )

    assert df.height == 0


@pytest.mark.parametrize(
    ("ratings_to_scrape", "stats_to_scrape"),
    [({"rating": "predictive_rating"}, {}), ({}, {"stat": "third_down_pct"})],
)
def test_scrape_team_rankings_for_week_uses_zip_truncation_fallback(
    monkeypatch,
    ratings_to_scrape: dict[str, str],
    stats_to_scrape: dict[str, str],
) -> None:
    """Zip strictness mismatches fall back to truncation instead of aborting the scrape."""
    html = """
    <table>
        <tr><th>Rank</th><th>Team</th><th>Value</th></tr>
        <tr><td>1</td><td>Buffalo Bills</td><td>6.7</td></tr>
    </table>
    """

    def fake_zip(*args, **kwargs):
        """Raise only for strict zip calls so the truncation fallback executes."""
        if kwargs.get("strict"):
            raise ValueError("length mismatch")
        return builtins.zip(*args, strict=bool(kwargs.get("strict", False)))

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )
    monkeypatch.setattr(scraping_utils, "zip", fake_zip, raising=False)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape=ratings_to_scrape,
        stats_to_scrape=stats_to_scrape,
    )

    assert df.height == 1


def test_scrape_team_rankings_for_week_handles_missing_tables(monkeypatch) -> None:
    """Missing HTML tables produce an empty TeamRankings result."""

    def fake_get(*_args, **_kwargs):
        """Return HTML without any table content."""
        return DummyResponse("<html><body><div>No table</div></body></html>")

    monkeypatch.setattr(scraping_utils.requests, "get", fake_get)
    monkeypatch.setattr(scraping_utils, "sleep", lambda *_args, **_kwargs: None)

    df = scraping_utils.scrape_team_rankings_for_week(
        1,
        date(2023, 9, 6),
        ratings_to_scrape={"rating": "predictive_rating"},
        stats_to_scrape={},
    )

    assert df.height == 0


def test_merge_tr_data_without_week_column() -> None:
    """TR merge falls back to team-only joins when week is absent."""
    existing = pl.DataFrame({"team_abbr": ["AAA"], "predictive_rating": [1.0]})
    new = pl.DataFrame({"team_abbr": ["AAA"], "third_down_pct": [0.5]})

    merged = scraping_utils.merge_tr_data(existing, new)

    assert merged.to_dicts() == [
        {"team_abbr": "AAA", "predictive_rating": 1.0, "third_down_pct": 0.5}
    ]


def test_update_season_team_rankings_skips_empty_week_files(tmp_path, monkeypatch) -> None:
    """Empty week CSVs do not produce a consolidated season file."""
    monkeypatch.setattr(constants, "DATA_PATH", str(tmp_path))
    season_dir = Path(tmp_path) / "2023"
    season_dir.mkdir(parents=True, exist_ok=True)
    (season_dir / "2023_week_01_team_rankings.csv").write_text("team_abbr,week\n", encoding="utf-8")

    scraping_utils.update_season_team_rankings(2023)

    assert not (season_dir / "2023_team_rankings.csv").exists()


def test_scrape_survivor_grid_spreads_handles_request_and_header_failures(monkeypatch) -> None:
    """SurvivorGrid scraping returns empty data on request or header-parse failures."""

    def failing_get(*_args, **_kwargs):
        """Raise a request-layer failure."""
        raise scraping_utils.requests.RequestException("boom")

    monkeypatch.setattr(scraping_utils.requests, "get", failing_get)
    assert scraping_utils.scrape_survivor_grid_spreads() == {}

    def headerless_get(*_args, **_kwargs):
        """Return a table that lacks the required Team/week headers."""
        rows = "".join("<tr><td>BUF</td><td>PK</td></tr>" for _ in range(32))
        return DummyResponse(f"<table><tr><th>Club</th><th>Spread</th></tr>{rows}</table>")

    monkeypatch.setattr(scraping_utils.requests, "get", headerless_get)
    assert scraping_utils.scrape_survivor_grid_spreads() == {}


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><div>No table</div></body></html>",
        "<table><tr><th>Team</th><th>16</th></tr><tr><td>BUF</td><td>PK</td></tr></table>",
    ],
)
def test_scrape_survivor_grid_spreads_handles_missing_data_table(monkeypatch, html: str) -> None:
    """Pages with no usable SurvivorGrid table return an empty spread mapping."""
    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    assert scraping_utils.scrape_survivor_grid_spreads() == {}


def test_scrape_survivor_grid_spreads_requires_team_column(monkeypatch) -> None:
    """A SurvivorGrid table without the Team header is rejected."""
    rows = "".join("<tr><td>BUF</td><td>PK</td></tr>" for _ in range(32))
    html = f"<table><tr><th>Club</th><th>16</th></tr>{rows}</table>"

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    assert scraping_utils.scrape_survivor_grid_spreads() == {}


def test_scrape_survivor_grid_spreads_parses_pickem_cells(monkeypatch) -> None:
    """Pick'em cells are normalized to a zero spread."""
    rows = ["<tr><td>BUF</td><td>PK</td></tr>"]
    rows.extend(f"<tr><td>XXX{i}</td><td>BYE</td></tr>" for i in range(31))
    html = f"<table><tr><th>Team</th><th>16</th></tr>{''.join(rows)}</table>"

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )

    spreads = scraping_utils.scrape_survivor_grid_spreads()

    assert spreads["BUF"][16] == pytest.approx(0.0)


def test_scrape_survivor_grid_spreads_handles_row_edge_cases(monkeypatch) -> None:
    """Short rows, invalid teams, and unmatched spread cells are skipped safely."""
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

    spreads = scraping_utils.scrape_survivor_grid_spreads()

    assert 16 not in spreads.get("BUF", {})
    assert spreads["BUF"][17] == pytest.approx(-3.5)


def test_scrape_survivor_grid_spreads_handles_float_conversion_failure(monkeypatch) -> None:
    """Numeric spread conversion failures are skipped instead of aborting the row."""
    rows = ["<tr><td>BUF</td><td>@KC-3.5</td></tr>"]
    rows.extend(f"<tr><td>ZZZ{i}</td><td>BYE</td></tr>" for i in range(31))
    html = f"<table><tr><th>Team</th><th>16</th></tr>{''.join(rows)}</table>"

    def fake_float(_text: str) -> float:
        """Force the spread parser down its ValueError handling branch."""
        raise ValueError("bad float")

    monkeypatch.setattr(
        scraping_utils.requests, "get", lambda *_args, **_kwargs: DummyResponse(html)
    )
    monkeypatch.setattr(scraping_utils, "float", fake_float, raising=False)

    assert scraping_utils.scrape_survivor_grid_spreads() == {}
