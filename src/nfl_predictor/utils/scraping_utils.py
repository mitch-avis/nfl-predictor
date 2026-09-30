"""Web scraping utilities for NFL data collection.

This module provides functions for scraping data from external sources:
    - TeamRankings.com: Team ratings and statistics
    - SurvivorGrid.com: Weekly game spreads

All scraping functions include appropriate rate limiting and error handling.
"""

import re
from datetime import date, timedelta
from time import sleep
from typing import TYPE_CHECKING

import polars as pl
import requests
from bs4 import BeautifulSoup, Tag

from nfl_predictor import constants
from nfl_predictor.utils import clock
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Callable

# A TeamRankings table row holds the rank, the team and the value.
MIN_TEAM_ROW_CELLS = 3
NFL_TEAM_COUNT = 32


def get_season_start(year: int) -> date:
    """Calculate NFL season start date for a given year.

    The NFL season traditionally starts on the Thursday following the first Monday of September.

    Args:
        year: The year for which to calculate the season start.

    Returns:
        The season start date.

    """
    sept_first = date(year, 9, 1)
    first_monday = sept_first + timedelta((7 - sept_first.weekday()) % 7)
    return first_monday + timedelta(days=3)


def get_current_nfl_week() -> tuple[int, int]:
    """Get the current NFL season and week number.

    Returns:
        Tuple of (season, week)

    """
    today = clock.local_today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1

    def adjust_to_tuesday(start_date: date) -> date:
        return start_date - timedelta(days=(start_date.weekday() - 1) % 7)

    season_start = adjust_to_tuesday(get_season_start(current_season))
    max_week = constants.get_regular_season_weeks(current_season) + 4

    if today < season_start:
        return current_season, 1

    week_number = ((today - season_start).days // 7) + 1
    return current_season, min(max(1, week_number), max_week)


def get_week_date(season: int, week: int) -> date:
    """Get the date representing the start of a specific week for TeamRankings scraping.

    This is typically the Wednesday of that week when TR data is most up-to-date.

    Args:
        season: NFL season year
        week: Week number

    Returns:
        Date to use for TR scraping

    """
    season_start = get_season_start(season)
    # Go back 8 days from season start to get the base date
    base_date = season_start - timedelta(days=8)
    # Add weeks to get to the target week
    return base_date + timedelta(weeks=week)


def normalize_team_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Normalize team abbreviations in a column to canonical form.

    Args:
        df: Polars DataFrame
        column: Name of the column containing team abbreviations

    Returns:
        DataFrame with normalized team abbreviations

    """
    return df.with_columns(pl.col(column).replace(constants.ALIAS_TO_CANONICAL).alias(column))


def scrape_team_rankings_for_week(
    week_number: int,
    week_date: date,
    ratings_to_scrape: dict[str, str] | None = None,
    stats_to_scrape: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Scrape team rankings for a specific week from TeamRankings.com.

    This function iterates over each team ranking type defined in constants, constructs URLs,
    parses HTML to extract ranking information, and compiles it into a DataFrame.

    Each table on TeamRankings is sorted by that metric's value, so teams appear in different
    orders across tables. We use a team-keyed dictionary to ensure proper matching.

    Args:
        week_number: The week number for which to scrape rankings.
        week_date: The date corresponding to the week of interest.
        ratings_to_scrape: Optional dict of {url_path: column_name} for ratings.
                           If None, scrapes all ratings from constants.TEAM_RANKINGS_RATINGS.
        stats_to_scrape: Optional dict of {url_path: column_name} for stats.
                         If None, scrapes all stats from constants.TEAM_RANKINGS_STATS.

    Returns:
        A Polars DataFrame containing team rankings for the specified week.

    """
    # Use defaults if not specified
    if ratings_to_scrape is None:
        ratings_to_scrape = constants.TEAM_RANKINGS_RATINGS
    if stats_to_scrape is None:
        stats_to_scrape = constants.TEAM_RANKINGS_STATS

    total_items = len(ratings_to_scrape) + len(stats_to_scrape)
    log.info(
        "Scraping %d items for Week %d (date: %s)...",
        total_items,
        week_number,
        week_date,
    )

    # Use a team-keyed dictionary to properly match data across tables
    # Each team maps to a dict of {column_name: value}
    team_data: dict[str, dict[str, float]] = {}
    tables = [
        *(
            (f"ranking/{path}", name, _parse_tr_rating_table)
            for path, name in ratings_to_scrape.items()
        ),
        *((f"stat/{path}", name, _parse_tr_stat_table) for path, name in stats_to_scrape.items()),
    ]
    for path, column_name, parse_table in tables:
        log.debug("Scraping %s", column_name)
        url = f"{constants.TEAM_RANKINGS_URL}/{path}?date={week_date}"
        for team_abbr, value in _scrape_tr_table(url, column_name, week_date, parse_table):
            team_data.setdefault(team_abbr, {})[column_name] = value
        sleep(constants.TEAM_RANKINGS_SLEEP)

    if not team_data:
        log.warning("No TR data scraped for week %d", week_number)
        return pl.DataFrame()

    # Build DataFrame from team-keyed dictionary
    tr_df = pl.DataFrame(
        [{"team_abbr": team_abbr, **metrics} for team_abbr, metrics in team_data.items()]
    )
    tr_df = tr_df.with_columns(pl.lit(week_number).alias("week"))

    # Normalize team abbreviations
    tr_df = normalize_team_column(tr_df, "team_abbr")

    log.info("Scraped TR data for %d teams", tr_df.height)
    return tr_df


def _scrape_tr_table(
    url: str,
    column_name: str,
    week_date: date,
    parse_table: Callable[[Tag], tuple[list[str], list[float]]],
) -> list[tuple[str, float]]:
    """Fetch one TeamRankings table and return its ``(team, value)`` pairs.

    A request or parse failure, or a page without a table, is logged and gives no pairs. When
    the parser finds more teams than values (or the reverse) the matched pairs are kept.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        table = BeautifulSoup(response.content, "html.parser").find("table")
        if not isinstance(table, Tag):
            log.warning("No data found for %s on %s", column_name, week_date)
            return []
        teams, values = parse_table(table)
    except requests.RequestException as e:
        log.warning("Failed to scrape %s: %s", column_name, e)
        return []
    except (ValueError, AttributeError) as e:
        log.warning("Failed to parse %s: %s", column_name, e)
        return []
    if len(teams) != len(values):
        log.warning(
            "TeamRankings parse length mismatch for %s on %s (teams=%d, values=%d); truncating",
            column_name,
            week_date,
            len(teams),
            len(values),
        )
    return list(zip(teams, values, strict=False))


def _parse_tr_rating_table(table: Tag) -> tuple[list[str], list[float]]:
    """Parse a TeamRankings rating table using BeautifulSoup.

    Args:
        table: BeautifulSoup table element

    Returns:
        Tuple of (team_abbreviations, ratings)

    """
    teams = []
    ratings = []

    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip header row
        cells = row.find_all("td")
        if len(cells) >= MIN_TEAM_ROW_CELLS:
            # Column 1 is team name, column 2 is rating
            team_text = cells[1].get_text(strip=True)
            rating_text = cells[2].get_text(strip=True)

            # Strip win-loss records from team names (with or without space)
            team_str = re.sub(r"\s*\(\d+-\d+(-\d+)*\)$", "", team_text)
            abbr = constants.TEAMS_TO_ABBR.get(team_str, team_str)
            teams.append(abbr)

            # Parse rating value
            try:
                ratings.append(float(rating_text))
            except ValueError:
                ratings.append(0.0)

    return teams, ratings


def _parse_tr_stat_table(table: Tag) -> tuple[list[str], list[float]]:
    """Parse a TeamRankings statistics table using BeautifulSoup.

    Args:
        table: BeautifulSoup table element

    Returns:
        Tuple of (team_abbreviations, stat_values)

    """
    teams = []
    stats = []

    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip header row
        cells = row.find_all("td")
        if len(cells) >= MIN_TEAM_ROW_CELLS:
            # Column 1 is team name, column 2 is stat value
            team_text = cells[1].get_text(strip=True)
            stat_text = cells[2].get_text(strip=True)

            # Strip win-loss records from team names (with or without space)
            team_str = re.sub(r"\s*\(\d+-\d+(-\d+)*\)$", "", team_text)
            abbr = constants.TEAMS_TO_ABBR.get(team_str, team_str)
            teams.append(abbr)

            # Parse stat value (handle percentages, remove % sign)
            stat_text = stat_text.replace("%", "")
            try:
                stats.append(float(stat_text))
            except ValueError:
                stats.append(0.0)

    return teams, stats


def get_missing_tr_columns(
    existing_df: pl.DataFrame,
) -> tuple[dict[str, str], dict[str, str]]:
    """Determine which TR ratings and stats are missing from an existing DataFrame.

    Compares the columns in the existing DataFrame against the required columns
    defined in constants.TEAM_RANKINGS_RATINGS and constants.TEAM_RANKINGS_STATS.

    Args:
        existing_df: Existing TeamRankings DataFrame to check

    Returns:
        Tuple of (missing_ratings, missing_stats) where each is a dict of
        {url_path: column_name} for items that need to be scraped.

    """
    existing_cols = set(existing_df.columns)

    missing_ratings = {
        url_path: col_name
        for url_path, col_name in constants.TEAM_RANKINGS_RATINGS.items()
        if col_name not in existing_cols
    }
    missing_stats = {
        url_path: col_name
        for url_path, col_name in constants.TEAM_RANKINGS_STATS.items()
        if col_name not in existing_cols
    }

    return missing_ratings, missing_stats


def merge_tr_data(
    existing_df: pl.DataFrame,
    new_df: pl.DataFrame,
) -> pl.DataFrame:
    """Merge newly scraped TR data into an existing DataFrame.

    Joins the new columns to the existing data on team_abbr and week columns.

    Args:
        existing_df: Existing TR DataFrame
        new_df: Newly scraped TR DataFrame with additional columns

    Returns:
        Combined DataFrame with all columns

    """
    if existing_df.height == 0:
        return new_df
    if new_df.height == 0:
        return existing_df

    # Get columns to add (excluding team_abbr and week)
    existing_cols = set(existing_df.columns)
    new_cols = [c for c in new_df.columns if c not in existing_cols]

    if not new_cols:
        return existing_df

    # Select only the columns we need to add, plus join keys
    join_cols = ["team_abbr"]
    if "week" in new_df.columns and "week" in existing_df.columns:
        join_cols.append("week")

    new_df_subset = new_df.select(join_cols + new_cols)

    # Join the new data
    return existing_df.join(new_df_subset, on=join_cols, how="left")


def save_team_rankings_week(tr_df: pl.DataFrame, season: int, week: int) -> None:
    """Save scraped team rankings to a CSV file for the specific week.

    Args:
        tr_df: TeamRankings DataFrame to save
        season: Season year
        week: Week number

    """
    season_dir = constants.DATA_PATH / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    file_path = season_dir / f"{season}_week_{week:02d}_team_rankings.csv"
    tr_df.write_csv(file_path)
    log.debug("Saved TR data to %s", file_path)


def update_season_team_rankings(season: int) -> None:
    """Update the consolidated season team rankings file from individual week files.

    Args:
        season: Season year

    """
    season_dir = constants.DATA_PATH / str(season)
    all_weeks_data = []

    # Find and load all week files
    for week in range(1, 23):  # Up to playoff weeks
        week_file = season_dir / f"{season}_week_{week:02d}_team_rankings.csv"
        if week_file.exists():
            week_df = pl.read_csv(week_file)
            if week_df.height > 0:
                all_weeks_data.append(week_df)

    if all_weeks_data:
        combined_df = pl.concat(all_weeks_data, how="diagonal")
        combined_path = season_dir / f"{season}_team_rankings.csv"
        combined_df.write_csv(combined_path)
        log.debug("Updated season TR file: %s", combined_path)


def scrape_survivor_grid_spreads() -> dict[str, dict[int, float]]:
    """Scrape weekly spreads from SurvivorGrid.com for future games.

    The site provides spreads for remaining weeks in the current NFL season.
    Each team row shows the team name and spreads for upcoming weeks.

    Returns:
        Dictionary mapping team abbreviation to dict of week -> spread.
        Example: {"BUF": {16: -10.5, 17: -3.0, 18: -14.0}, ...}
        Returns empty dict if scraping fails.

    """
    try:
        response = requests.get(constants.SURVIVOR_GRID_URL, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch SurvivorGrid data: %s", e)
        return {}

    data_table = _survivor_grid_table(BeautifulSoup(response.text, "lxml"))
    if data_table is None:
        return {}
    grid_columns = _survivor_grid_columns(data_table)
    if grid_columns is None:
        return {}
    team_col_idx, week_columns = grid_columns
    log.debug("Found SurvivorGrid week columns: %s", list(week_columns.values()))

    # Parse team rows
    spreads = {}
    for row in data_table.find_all("tr")[1:]:  # Skip header
        cells = row.find_all(["th", "td"])
        if len(cells) <= team_col_idx:
            continue

        # Get team abbreviation from the Team column
        team_cell = cells[team_col_idx].get_text(strip=True)
        # Extract team abbr (may have record in parentheses like "BUF(10-4)")
        team_abbr_raw = team_cell.split("(")[0].strip() if "(" in team_cell else team_cell.strip()
        # Normalize to canonical abbreviation
        team_abbr = constants.normalize_team_abbr(team_abbr_raw)
        if team_abbr not in constants.TEAM_ABBR:
            continue  # Not a valid team

        team_spreads = {
            week: spread
            for col_idx, week in week_columns.items()
            if col_idx < len(cells)
            and (spread := _parse_grid_spread(cells[col_idx].get_text(strip=True))) is not None
        }
        if team_spreads:
            spreads[team_abbr] = team_spreads

    log.info("Scraped SurvivorGrid spreads for %d teams", len(spreads))
    return spreads


def _survivor_grid_table(soup: BeautifulSoup) -> Tag | None:
    """Return the page's team grid: the first table with a row for every team."""
    tables = soup.find_all("table")
    if not tables:
        log.warning("No tables found on SurvivorGrid page")
        return None
    # The main grid table is typically the first/largest table
    for table in tables:
        if len(table.find_all("tr")) >= NFL_TEAM_COUNT:  # Should have a row for every team
            return table
    log.warning("Could not find SurvivorGrid data table")
    return None


def _survivor_grid_columns(data_table: Tag) -> tuple[int, dict[int, int]] | None:
    """Return the Team column index and each week column's index and week number."""
    header_row = data_table.find("tr")
    if not isinstance(header_row, Tag):
        return None
    headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]

    # Find which columns contain week numbers and which has Team
    week_columns = {}  # index -> week number
    team_col_idx = None
    for i, header in enumerate(headers):
        if header.isdigit():
            week_columns[i] = int(header)
        elif header == "Team":
            team_col_idx = i

    if not week_columns:
        log.warning("No week columns found in SurvivorGrid table")
        return None
    if team_col_idx is None:
        log.warning("No Team column found in SurvivorGrid table")
        return None
    return team_col_idx, week_columns


def _parse_grid_spread(cell_text: str) -> float | None:
    """Return a grid cell's spread ("@CLE-10.5", "LV+3", "PK"), or None for a bye or blank.

    The spread is at the end, after the opponent indicator: an optional sign and a number, or
    PK for a pick'em.
    """
    # Skip bye weeks and empty cells
    if not cell_text or cell_text == "BYE":
        return None
    match = re.search(r"([+-]?\d+\.?\d*|PK)$", cell_text)
    if not match:
        return None
    spread_text = match.group(1)
    if spread_text == "PK":
        return 0.0
    try:
        return float(spread_text)
    except ValueError:
        return None
