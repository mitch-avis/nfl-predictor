"""Tests for scraping utilities (HTML parsing and SurvivorGrid spreads)."""

from datetime import date

import polars as pl
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
