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

    season_start = adjust_to_tuesday(get_season_start(today.year))

    if today < season_start:
        # Before current year's season starts, we're in offseason
        # Return last week of previous season (or playoffs)
        prev_season = today.year - 1
        return prev_season, 22  # Allow playoff weeks

    week_number = ((today - season_start).days // 7) + 1
    return current_season, min(week_number, 22)  # Allow playoff weeks


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


def scrape_team_rankings_for_week(week_number: int, week_date: date) -> pl.DataFrame:
    """
    Scrape team rankings for a specific week from TeamRankings.com.

    This function iterates over each team ranking type defined in constants, constructs URLs,
    parses HTML to extract ranking information, and compiles it into a DataFrame.

    Args:
        week_number: The week number for which to scrape rankings.
        week_date: The date corresponding to the week of interest.

    Returns:
        A Polars DataFrame containing team rankings for the specified week.
    """
    log.info("Scraping TeamRankings for Week %d (date: %s)...", week_number, week_date)

    # Initialize with team abbreviations column
    all_ratings: dict[str, list] = {"team_abbr": []}

    # Scrape ratings
    for rating, rating_name in constants.TEAM_RANKINGS_RATINGS.items():
        log.debug("Scraping rating: %s", rating_name)
        url = f"{constants.TEAM_RANKINGS_URL}/ranking/{rating}?date={week_date}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if table:
                teams, ratings = _parse_tr_rating_table(table)

                # If this is the first iteration, populate team_abbr
                if not all_ratings["team_abbr"]:
                    all_ratings["team_abbr"] = teams

                all_ratings[rating_name] = ratings
            else:
                log.warning("No data found for %s on %s", rating_name, week_date)

        except requests.RequestException as e:
            log.warning("Failed to scrape %s: %s", rating_name, e)
        except (ValueError, AttributeError) as e:
            log.warning("Failed to parse %s: %s", rating_name, e)

        sleep(constants.TEAM_RANKINGS_SLEEP)

    # Scrape statistics
    for statistic, stat_name in constants.TEAM_RANKINGS_STATS.items():
        log.debug("Scraping statistic: %s", stat_name)
        url = f"{constants.TEAM_RANKINGS_URL}/stat/{statistic}?date={week_date}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            table = soup.find("table")

            if table:
                stats = _parse_tr_stat_table(table)
                all_ratings[stat_name] = stats
            else:
                log.warning("No data found for %s on %s", stat_name, week_date)

        except requests.RequestException as e:
            log.warning("Failed to scrape %s: %s", stat_name, e)
        except (ValueError, AttributeError) as e:
            log.warning("Failed to parse %s: %s", stat_name, e)

        sleep(constants.TEAM_RANKINGS_SLEEP)

    # Build DataFrame
    if all_ratings["team_abbr"]:
        tr_df = pl.DataFrame(all_ratings)
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

            # Strip win-loss records from team names
            team_str = re.sub(r"\s+\(\d+-\d+(-\d+)*\)$", "", team_text)
            abbr = constants.TEAMS_TO_ABBR.get(team_str, team_str)
            teams.append(abbr)

            # Parse rating value
            try:
                ratings.append(float(rating_text))
            except ValueError:
                ratings.append(0.0)

    return teams, ratings


def _parse_tr_stat_table(table) -> list[float]:
    """
    Parse a TeamRankings statistics table using BeautifulSoup.

    Args:
        table: BeautifulSoup table element

    Returns:
        List of statistic values
    """
    stats = []

    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip header row
        cells = row.find_all("td")
        if len(cells) >= 3:
            # Column 2 is the stat value
            stat_text = cells[2].get_text(strip=True)

            # Parse stat value (handle percentages, remove % sign)
            stat_text = stat_text.replace("%", "")
            try:
                stats.append(float(stat_text))
            except ValueError:
                stats.append(0.0)

    return stats


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
