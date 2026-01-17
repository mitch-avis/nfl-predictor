"""
Web scraping utilities for NFL data collection.

This module provides functions for scraping data from external sources:
    - TeamRankings.com: Team ratings and statistics
    - SurvivorGrid.com: Weekly game spreads

All scraping functions include appropriate rate limiting and error handling.
"""

import os
import re
from datetime import date, timedelta
from time import sleep
from typing import Optional

import polars as pl
import requests
from bs4 import BeautifulSoup

from nfl_predictor import constants
from nfl_predictor.utils.logger import log


def get_season_start(year: int) -> date:
    """
    Calculate NFL season start date for a given year.

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
    """
    Get the current NFL season and week number.

    Returns:
        Tuple of (season, week)
    """

    today = date.today()
    current_season = today.year if today.month > constants.SEASON_END_MONTH else today.year - 1

    def adjust_to_tuesday(start_date: date) -> date:
        return start_date - timedelta(days=(start_date.weekday() - 1) % 7)

    season_start = adjust_to_tuesday(get_season_start(current_season))
    max_week = constants.get_regular_season_weeks(current_season) + 4

    if today < season_start:
        # Before current year's season starts, we're in offseason
        # Return last week of previous season (or playoffs)
        prev_season = current_season - 1
        prev_max_week = constants.get_regular_season_weeks(prev_season) + 4
        return prev_season, prev_max_week

    week_number = ((today - season_start).days // 7) + 1
    return current_season, min(max(1, week_number), max_week)


def get_week_date(season: int, week: int) -> date:
    """
    Get the date representing the start of a specific week for TeamRankings scraping.

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
    week_date = base_date + timedelta(weeks=week)
    return week_date


def normalize_team_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Normalize team abbreviations in a column to canonical form.

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
    ratings_to_scrape: Optional[dict[str, str]] = None,
    stats_to_scrape: Optional[dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Scrape team rankings for a specific week from TeamRankings.com.

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

    # Scrape ratings
    for rating, rating_name in ratings_to_scrape.items():
        log.debug("Scraping rating: %s", rating_name)
        url = f"{constants.TEAM_RANKINGS_URL}/ranking/{rating}?date={week_date}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if table:
                teams, ratings = _parse_tr_rating_table(table)

                # Store each team's rating, keyed by team abbreviation
                try:
                    pairs = zip(teams, ratings, strict=True)
                except ValueError:
                    log.warning(
                        "TeamRankings parse length mismatch for %s on %s "
                        "(teams=%d, values=%d); truncating",
                        rating_name,
                        week_date,
                        len(teams),
                        len(ratings),
                    )
                    pairs = zip(teams, ratings, strict=False)

                for team_abbr, rating_value in pairs:
                    if team_abbr not in team_data:
                        team_data[team_abbr] = {}
                    team_data[team_abbr][rating_name] = rating_value
            else:
                log.warning("No data found for %s on %s", rating_name, week_date)

        except requests.RequestException as e:
            log.warning("Failed to scrape %s: %s", rating_name, e)
        except (ValueError, AttributeError) as e:
            log.warning("Failed to parse %s: %s", rating_name, e)

        sleep(constants.TEAM_RANKINGS_SLEEP)

    # Scrape statistics
    for statistic, stat_name in stats_to_scrape.items():
        log.debug("Scraping statistic: %s", stat_name)
        url = f"{constants.TEAM_RANKINGS_URL}/stat/{statistic}?date={week_date}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if table:
                teams, stats = _parse_tr_stat_table(table)

                # Store each team's stat, keyed by team abbreviation
                try:
                    pairs = zip(teams, stats, strict=True)
                except ValueError:
                    log.warning(
                        "TeamRankings parse length mismatch for %s on %s "
                        "(teams=%d, values=%d); truncating",
                        stat_name,
                        week_date,
                        len(teams),
                        len(stats),
                    )
                    pairs = zip(teams, stats, strict=False)

                for team_abbr, stat_value in pairs:
                    if team_abbr not in team_data:
                        team_data[team_abbr] = {}
                    team_data[team_abbr][stat_name] = stat_value
            else:
                log.warning("No data found for %s on %s", stat_name, week_date)

        except requests.RequestException as e:
            log.warning("Failed to scrape %s: %s", stat_name, e)
        except (ValueError, AttributeError) as e:
            log.warning("Failed to parse %s: %s", stat_name, e)

        sleep(constants.TEAM_RANKINGS_SLEEP)

    # Build DataFrame from team-keyed dictionary
    if team_data:
        # Convert team_data dict to columnar format
        rows = []
        for team_abbr, metrics in team_data.items():
            row = {"team_abbr": team_abbr, **metrics}
            rows.append(row)

        tr_df = pl.DataFrame(rows)
        tr_df = tr_df.with_columns(pl.lit(week_number).alias("week"))

        # Normalize team abbreviations
        tr_df = normalize_team_column(tr_df, "team_abbr")

        log.info("Scraped TR data for %d teams", tr_df.height)
        return tr_df

    log.warning("No TR data scraped for week %d", week_number)
    return pl.DataFrame()


def _parse_tr_rating_table(table) -> tuple[list[str], list[float]]:
    """
    Parse a TeamRankings rating table using BeautifulSoup.

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
        if len(cells) >= 3:
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


def _parse_tr_stat_table(table) -> tuple[list[str], list[float]]:
    """
    Parse a TeamRankings statistics table using BeautifulSoup.

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
        if len(cells) >= 3:
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
    """
    Determine which TR ratings and stats are missing from an existing DataFrame.

    Compares the columns in the existing DataFrame against the required columns
    defined in constants.TEAM_RANKINGS_RATINGS and constants.TEAM_RANKINGS_STATS.

    Args:
        existing_df: Existing TeamRankings DataFrame to check

    Returns:
        Tuple of (missing_ratings, missing_stats) where each is a dict of
        {url_path: column_name} for items that need to be scraped.
    """

    existing_cols = set(existing_df.columns)

    missing_ratings = {}
    for url_path, col_name in constants.TEAM_RANKINGS_RATINGS.items():
        if col_name not in existing_cols:
            missing_ratings[url_path] = col_name

    missing_stats = {}
    for url_path, col_name in constants.TEAM_RANKINGS_STATS.items():
        if col_name not in existing_cols:
            missing_stats[url_path] = col_name

    return missing_ratings, missing_stats


def merge_tr_data(
    existing_df: pl.DataFrame,
    new_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge newly scraped TR data into an existing DataFrame.

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
    merged = existing_df.join(new_df_subset, on=join_cols, how="left")

    return merged


def save_team_rankings_week(tr_df: pl.DataFrame, season: int, week: int) -> None:
    """
    Save scraped team rankings to a CSV file for the specific week.

    Args:
        tr_df: TeamRankings DataFrame to save
        season: Season year
        week: Week number
    """

    season_dir = os.path.join(constants.DATA_PATH, str(season))
    os.makedirs(season_dir, exist_ok=True)

    file_path = os.path.join(season_dir, f"{season}_week_{week:02d}_team_rankings.csv")
    tr_df.write_csv(file_path)
    log.debug("Saved TR data to %s", file_path)


def update_season_team_rankings(season: int) -> None:
    """
    Update the consolidated season team rankings file from individual week files.

    Args:
        season: Season year
    """

    season_dir = os.path.join(constants.DATA_PATH, str(season))
    all_weeks_data = []

    # Find and load all week files
    for week in range(1, 23):  # Up to playoff weeks
        week_file = os.path.join(season_dir, f"{season}_week_{week:02d}_team_rankings.csv")
        if os.path.exists(week_file):
            week_df = pl.read_csv(week_file)
            if week_df.height > 0:
                all_weeks_data.append(week_df)

    if all_weeks_data:
        combined_df = pl.concat(all_weeks_data, how="diagonal")
        combined_path = os.path.join(season_dir, f"{season}_team_rankings.csv")
        combined_df.write_csv(combined_path)
        log.debug("Updated season TR file: %s", combined_path)


def scrape_survivor_grid_spreads() -> dict[str, dict[int, float]]:
    """
    Scrape weekly spreads from SurvivorGrid.com for future games.

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

    soup = BeautifulSoup(response.text, "lxml")

    # Find the main data table
    tables = soup.find_all("table")
    if not tables:
        log.warning("No tables found on SurvivorGrid page")
        return {}

    # The main grid table is typically the first/largest table
    data_table = None
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) >= 32:  # Should have all 32 teams
            data_table = table
            break

    if not data_table:
        log.warning("Could not find SurvivorGrid data table")
        return {}

    # Parse header row to get week numbers
    header_row = data_table.find("tr")
    if not header_row:
        return {}

    headers = []
    for th in header_row.find_all(["th", "td"]):
        text = th.get_text(strip=True)
        headers.append(text)

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
        return {}

    if team_col_idx is None:
        log.warning("No Team column found in SurvivorGrid table")
        return {}

    log.debug("Found SurvivorGrid week columns: %s", list(week_columns.values()))

    # Parse team rows
    spreads = {}
    rows = data_table.find_all("tr")[1:]  # Skip header

    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) <= team_col_idx:
            continue

        # Get team abbreviation from the Team column
        team_cell = cells[team_col_idx].get_text(strip=True)

        # Extract team abbr (may have record in parentheses like "BUF(10-4)")
        if "(" in team_cell:
            team_abbr_raw = team_cell.split("(")[0].strip()
        else:
            team_abbr_raw = team_cell.strip()

        # Normalize to canonical abbreviation
        team_abbr = constants.normalize_team_abbr(team_abbr_raw)
        if team_abbr not in constants.TEAM_ABBR:
            continue  # Not a valid team

        team_spreads = {}

        for col_idx, week in week_columns.items():
            if col_idx >= len(cells):
                continue

            cell = cells[col_idx]
            # Get cell text - format is like "@CLE-10.5" or "LV-14" or "@KC+3.5"
            cell_text = cell.get_text(strip=True)

            # Skip bye weeks and empty cells
            if not cell_text or cell_text == "BYE":
                continue

            # Extract spread from cell text
            # The spread is at the end, after the opponent indicator
            # Patterns: "@CLE-10.5", "LV+3", "@KC-7", "PK"
            # Look for spread pattern: optional sign followed by number or PK
            match = re.search(r"([+-]?\d+\.?\d*|PK)$", cell_text)
            if not match:
                continue

            spread_text = match.group(1)
            if spread_text == "PK":
                team_spreads[week] = 0.0
            else:
                try:
                    spread_val = float(spread_text)
                    team_spreads[week] = spread_val
                except ValueError:
                    continue

        if team_spreads:
            spreads[team_abbr] = team_spreads

    log.info("Scraped SurvivorGrid spreads for %d teams", len(spreads))
    return spreads
