"""
This module defines constants for the NFL predictor project, including configurations for paths,
NFL season details, team abbreviations, and statistics for performance analysis. Centralizing these
configurations helps maintain consistency and integrity across the project, simplifying management
and updates.
"""

import os
from pathlib import Path

# Project directory configurations
ROOT_DIR = Path(__file__).parent.parent
DATA_PATH = os.path.join(ROOT_DIR, "data")

# NFL season configurations
SEASON_END_MONTH = 2  # NFL season typically ends in February
WEEKS_BEFORE_2021 = 17  # Number of weeks in NFL seasons before 2021
WEEKS_FROM_2021_ONWARDS = 18  # Number of weeks in NFL seasons from 2021 onwards


def get_regular_season_weeks(season: int) -> int:
    """Return the number of regular season weeks for a given season.

    Before 2021, the NFL had 17-week regular seasons.
    From 2021 onwards, the NFL has 18-week regular seasons.

    Args:
        season: The NFL season year

    Returns:
        Number of regular season weeks (17 before 2021, 18 from 2021 onwards)
    """
    return WEEKS_FROM_2021_ONWARDS if season >= 2021 else WEEKS_BEFORE_2021


# Median values for key statistics
MEDIAN_THIRD_DOWN = 0.4  # Median third down conversion rate for analysis
MEDIAN_FOURTH_DOWN = 0.5  # Median fourth down conversion rate for analysis
MEDIAN_WIN_PERCENTAGE = 0.5  # Median win percentage for analysis

# Standard deviation of score differences
SCORE_DIFF_STD_DEV = 14.21377923  # Standard deviation of score differences for analysis

# URL for ELO ratings data
ELO_DATA_URL = "https://github.com/greerreNFL/nfeloqb/raw/main/qb_elos.csv"
ELO_LINES_URL = "https://github.com/greerreNFL/nfelomarket_data/raw/main/Data/lines.csv"

# URL for SurvivorGrid spreads (future games)
SURVIVOR_GRID_URL = "https://www.survivorgrid.com/"
DEFAULT_TOTAL_LINE = 45.6  # Average total score across 20+ seasons

# Minimum season for data collection (CPOE data starts Week 2 of 2006)
MIN_SEASON = 2006

# URL for Team Rankings data
TEAM_RANKINGS_URL = "https://www.teamrankings.com/nfl"
TEAM_RANKINGS_SLEEP = 1  # Sleep time in seconds for web scraping

# Team Rankings subpaths for different ratings
TEAM_RANKINGS_RATINGS = {
    "predictive-by-other": "predictive_rating",
    "home-by-other": "home_rating",
    "away-by-other": "away_rating",
    "schedule-strength-by-other": "strength_of_schedule_rating",
    "future-sos-by-other": "future_sos_rating",
    "last-5-games-by-other": "last_5_games_rating",
    "last-10-games-by-other": "last_10_games_rating",
    "in-division-by-other": "in_division_rating",
    "non-division-by-other": "non_division_rating",
    "luck-by-other": "luck_rating",
}

# Team Rankings subpaths for different statistics
# NOTE: We only scrape stats that can't be calculated from nflreadpy data.
# Third/fourth down and red zone percentages are not in nflreadpy team_stats.
# Other stats (points per game, yards per point, etc.) are calculated from nflreadpy.
TEAM_RANKINGS_STATS = {
    "third-down-conversion-pct": "third_down_pct",
    "opponent-third-down-conversion-pct": "opponent_third_down_pct",
    "fourth-down-conversion-pct": "fourth_down_pct",
    "opponent-fourth-down-conversion-pct": "opponent_fourth_down_pct",
    "red-zone-scoring-pct": "red_zone_td_pct",
    "opponent-red-zone-scoring-pct": "opponent_red_zone_td_pct",
}

# Unified team mapping table: canonical abbreviation -> all known aliases
# The canonical abbreviation is the standardized form used throughout this project
TEAM_MAPPING = {
    "ARI": {
        "canonical": "ARI",
        "name": "Arizona Cardinals",
        "city": "Arizona",
        "aliases": ["ARI", "crd", "CRD"],
    },
    "ATL": {
        "canonical": "ATL",
        "name": "Atlanta Falcons",
        "city": "Atlanta",
        "aliases": ["ATL", "atl"],
    },
    "BAL": {
        "canonical": "BAL",
        "name": "Baltimore Ravens",
        "city": "Baltimore",
        "aliases": ["BAL", "rav", "RAV"],
    },
    "BUF": {
        "canonical": "BUF",
        "name": "Buffalo Bills",
        "city": "Buffalo",
        "aliases": ["BUF", "buf"],
    },
    "CAR": {
        "canonical": "CAR",
        "name": "Carolina Panthers",
        "city": "Carolina",
        "aliases": ["CAR", "car"],
    },
    "CHI": {
        "canonical": "CHI",
        "name": "Chicago Bears",
        "city": "Chicago",
        "aliases": ["CHI", "chi"],
    },
    "CIN": {
        "canonical": "CIN",
        "name": "Cincinnati Bengals",
        "city": "Cincinnati",
        "aliases": ["CIN", "cin"],
    },
    "CLE": {
        "canonical": "CLE",
        "name": "Cleveland Browns",
        "city": "Cleveland",
        "aliases": ["CLE", "cle"],
    },
    "DAL": {
        "canonical": "DAL",
        "name": "Dallas Cowboys",
        "city": "Dallas",
        "aliases": ["DAL", "dal"],
    },
    "DEN": {
        "canonical": "DEN",
        "name": "Denver Broncos",
        "city": "Denver",
        "aliases": ["DEN", "den"],
    },
    "DET": {
        "canonical": "DET",
        "name": "Detroit Lions",
        "city": "Detroit",
        "aliases": ["DET", "det"],
    },
    "GB": {
        "canonical": "GB",
        "name": "Green Bay Packers",
        "city": "Green Bay",
        "aliases": ["GB", "gnb", "GNB", "GBP"],
    },
    "HOU": {
        "canonical": "HOU",
        "name": "Houston Texans",
        "city": "Houston",
        "aliases": ["HOU", "htx", "HTX"],
    },
    "IND": {
        "canonical": "IND",
        "name": "Indianapolis Colts",
        "city": "Indianapolis",
        "aliases": ["IND", "clt", "CLT"],
    },
    "JAX": {
        "canonical": "JAX",
        "name": "Jacksonville Jaguars",
        "city": "Jacksonville",
        "aliases": ["JAX", "jax", "JAC"],
    },
    "KC": {
        "canonical": "KC",
        "name": "Kansas City Chiefs",
        "city": "Kansas City",
        "aliases": ["KC", "kan", "KAN"],
    },
    "LAC": {
        "canonical": "LAC",
        "name": "Los Angeles Chargers",
        "city": "LA Chargers",
        "aliases": ["LAC", "sdg", "SDG", "SD"],
        "former_names": ["San Diego Chargers"],
    },
    "LAR": {
        "canonical": "LAR",
        "name": "Los Angeles Rams",
        "city": "LA Rams",
        "aliases": ["LAR", "LA", "ram", "RAM", "STL"],
        "former_names": ["St. Louis Rams"],
    },
    "LV": {
        "canonical": "LV",
        "name": "Las Vegas Raiders",
        "city": "Las Vegas",
        "aliases": ["LV", "LVR", "OAK", "rai", "RAI"],
        "former_names": ["Oakland Raiders"],
    },
    "MIA": {
        "canonical": "MIA",
        "name": "Miami Dolphins",
        "city": "Miami",
        "aliases": ["MIA", "mia"],
    },
    "MIN": {
        "canonical": "MIN",
        "name": "Minnesota Vikings",
        "city": "Minnesota",
        "aliases": ["MIN", "min"],
    },
    "NE": {
        "canonical": "NE",
        "name": "New England Patriots",
        "city": "New England",
        "aliases": ["NE", "nwe", "NWE"],
    },
    "NO": {
        "canonical": "NO",
        "name": "New Orleans Saints",
        "city": "New Orleans",
        "aliases": ["NO", "nor", "NOR"],
    },
    "NYG": {
        "canonical": "NYG",
        "name": "New York Giants",
        "city": "NY Giants",
        "aliases": ["NYG", "nyg"],
    },
    "NYJ": {
        "canonical": "NYJ",
        "name": "New York Jets",
        "city": "NY Jets",
        "aliases": ["NYJ", "nyj"],
    },
    "PHI": {
        "canonical": "PHI",
        "name": "Philadelphia Eagles",
        "city": "Philadelphia",
        "aliases": ["PHI", "phi"],
    },
    "PIT": {
        "canonical": "PIT",
        "name": "Pittsburgh Steelers",
        "city": "Pittsburgh",
        "aliases": ["PIT", "pit"],
    },
    "SEA": {
        "canonical": "SEA",
        "name": "Seattle Seahawks",
        "city": "Seattle",
        "aliases": ["SEA", "sea"],
    },
    "SF": {
        "canonical": "SF",
        "name": "San Francisco 49ers",
        "city": "San Francisco",
        "aliases": ["SF", "sfo", "SFO"],
    },
    "TB": {
        "canonical": "TB",
        "name": "Tampa Bay Buccaneers",
        "city": "Tampa Bay",
        "aliases": ["TB", "tam", "TAM"],
    },
    "TEN": {
        "canonical": "TEN",
        "name": "Tennessee Titans",
        "city": "Tennessee",
        "aliases": ["TEN", "oti", "OTI"],
    },
    "WSH": {
        "canonical": "WSH",
        "name": "Washington Commanders",
        "city": "Washington",
        "aliases": ["WSH", "WAS", "was"],
        "former_names": ["Washington Football Team", "Washington Redskins"],
    },
}


def _build_alias_to_canonical() -> dict[str, str]:
    """Build a reverse lookup from any alias to canonical abbreviation."""
    mapping = {}
    for canonical, info in TEAM_MAPPING.items():
        for alias in info["aliases"]:
            mapping[alias] = canonical
            mapping[alias.upper()] = canonical
            mapping[alias.lower()] = canonical
    return mapping


# Reverse lookup: any alias -> canonical abbreviation
ALIAS_TO_CANONICAL = _build_alias_to_canonical()


def normalize_team_abbr(abbr: str) -> str:
    """Normalize any team abbreviation to its canonical form."""
    if abbr is None:
        return abbr
    return ALIAS_TO_CANONICAL.get(abbr, ALIAS_TO_CANONICAL.get(abbr.upper(), abbr))


# List of all canonical team abbreviations
TEAM_ABBR = list(TEAM_MAPPING.keys())

# Dictionary used to map team names/cities to abbreviations (for TeamRankings scraping)
TEAMS_TO_ABBR = {info["city"]: canonical for canonical, info in TEAM_MAPPING.items()}

# Dictionary used to map full team names to abbreviations (for SurvivorGrid scraping)
TEAM_NAME_TO_ABBR = {info["name"]: canonical for canonical, info in TEAM_MAPPING.items()}

# Legacy mappings for backward compatibility with existing code
PFR_TEAM_ABBR = [
    "crd",
    "atl",
    "rav",
    "buf",
    "car",
    "chi",
    "cin",
    "cle",
    "dal",
    "den",
    "det",
    "gnb",
    "htx",
    "clt",
    "jax",
    "kan",
    "sdg",
    "ram",
    "rai",
    "mia",
    "min",
    "nwe",
    "nor",
    "nyg",
    "nyj",
    "phi",
    "pit",
    "sea",
    "sfo",
    "tam",
    "oti",
    "was",
    "was",
]
ELO_TEAM_ABBR = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "OAK",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
    "WSH",
]
PBP_TEAM_ABBR = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LA",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WSH",
    "WSH",
]

# Dictionary to map historical team names to modern names for consistency
MODERN_TEAM_NAMES = {
    "Oakland Raiders": "Las Vegas Raiders",
    "San Diego Chargers": "Los Angeles Chargers",
    "St. Louis Rams": "Los Angeles Rams",
    "Washington Football Team": "Washington Commanders",
    "Washington Redskins": "Washington Commanders",
}

# Divisional team groupings for analysis
DIVISION_TEAMS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WSH"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SF", "SEA"],
}

# Key statistics for analysis
BASE_COLUMNS = [
    "team_name",
    "team_abbr",
    "points_scored",
    "points_allowed",
    "game_won",
    "game_lost",
]
BOXSCORE_STATS = [
    "first_downs",
    "rush_attempts",
    "rush_yards",
    "rush_touchdowns",
    "pass_completions",
    "pass_attempts",
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "times_sacked",
    "yards_lost_from_sacks",
    "net_pass_yards",
    "total_yards",
    "fumbles",
    "fumbles_lost",
    "turnovers",
    "penalties",
    "yards_from_penalties",
    "third_down_conversions",
    "third_down_attempts",
    "fourth_down_conversions",
    "fourth_down_attempts",
    "time_of_possession",
]
AGG_STATS = [
    "win_perc",
    "points_scored",
    "points_allowed",
    "first_downs",
    "rush_attempts",
    "rush_yards",
    "rush_touchdowns",
    "pass_completions",
    "pass_attempts",
    "pass_yards",
    "pass_touchdowns",
    "interceptions",
    "times_sacked",
    "yards_lost_from_sacks",
    "net_pass_yards",
    "total_yards",
    "fumbles",
    "fumbles_lost",
    "turnovers",
    "penalties",
    "yards_from_penalties",
    "third_down_perc",
    "fourth_down_perc",
    "time_of_possession",
]
RATIOS_DICT = {
    "game_won": "sum",
    "game_lost": "sum",
    "third_down_conversions": "sum",
    "third_down_attempts": "sum",
    "fourth_down_conversions": "sum",
    "fourth_down_attempts": "sum",
    "opponent_third_down_conversions": "sum",
    "opponent_third_down_attempts": "sum",
    "opponent_fourth_down_conversions": "sum",
    "opponent_fourth_down_attempts": "sum",
}
AGG_DROP_COLS = [
    "team_name",
    "season",
    "week",
    "game_won",
    "game_lost",
    "third_down_conversions",
    "third_down_attempts",
    "fourth_down_conversions",
    "fourth_down_attempts",
    "opponent_third_down_conversions",
    "opponent_third_down_attempts",
    "opponent_fourth_down_conversions",
    "opponent_fourth_down_attempts",
]

# ============================================================================
# nflreadpy Column Mappings
# ============================================================================

# Mapping from nflreadpy team_stats columns to our internal column names
# This maps nflreadpy's load_team_stats() output to our feature names
NFLREADPY_TEAM_STATS_MAPPING = {
    # Pass offense
    "completions": "pass_completions",
    "attempts": "pass_attempts",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_touchdowns",
    "passing_interceptions": "interceptions_thrown",
    "sacks_suffered": "times_sacked",
    "sack_yards_lost": "yards_lost_from_sacks",
    "passing_air_yards": "passing_air_yards",
    "passing_yards_after_catch": "passing_yac",
    "passing_first_downs": "passing_first_downs",
    "passing_epa": "passing_epa",
    "passing_cpoe": "passing_cpoe",
    # Rush offense
    "carries": "rush_attempts",
    "rushing_yards": "rush_yards",
    "rushing_tds": "rush_touchdowns",
    "rushing_fumbles": "rushing_fumbles",
    "rushing_fumbles_lost": "rushing_fumbles_lost",
    "rushing_first_downs": "rushing_first_downs",
    "rushing_epa": "rushing_epa",
    # Receiving (team-level aggregates)
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_touchdowns",
    "receiving_fumbles": "receiving_fumbles",
    "receiving_fumbles_lost": "receiving_fumbles_lost",
    "receiving_air_yards": "receiving_air_yards",
    "receiving_yards_after_catch": "receiving_yac",
    "receiving_first_downs": "receiving_first_downs",
    "receiving_epa": "receiving_epa",
    # Special teams
    "special_teams_tds": "special_teams_tds",
}

# Columns to select from nflreadpy schedule for our use
NFLREADPY_SCHEDULE_COLUMNS = [
    "game_id",
    "season",
    "game_type",  # REG, WC, DIV, CON, SB for filtering
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "location",
    "result",
    "total",
    "overtime",
    "away_rest",
    "home_rest",
    "away_moneyline",
    "home_moneyline",
    "spread_line",
    "away_spread_odds",
    "home_spread_odds",
    "total_line",
    "under_odds",
    "over_odds",
    "div_game",
]

# nflreadpy schedule column renames to match our internal naming
NFLREADPY_SCHEDULE_RENAME = {
    "gameday": "date",
    "away_team": "away_abbr",
    "home_team": "home_abbr",
    "location": "neutral",  # Will need to transform: "Home" -> 0, "Neutral" -> 1
    "div_game": "division",
    "spread_line": "home_spread",  # nflreadpy uses home perspective for spread
}

# Columns to exclude from the ELO dataset for streamlined analysis
ELO_DROP_COLS = [
    "season",
    "playoff",
    "elo_prob1",
    "elo_prob2",
    "elo1_post",
    "elo2_post",
    "qb1_adj",
    "qb2_adj",
    "qbelo_prob1",
    "qbelo_prob2",
    "qb1_game_value",
    "qb2_game_value",
    "qb1_value_post",
    "qb2_value_post",
    "qbelo1_post",
    "qbelo2_post",
    "score1",
    "score2",
    "quality",
    "importance",
    "total_rating",
    "game_type",
    "game_id",
]

# Dictionary to swap columns for ELO dataset in the event of neutral site games
ELO_SWAP_COLS = {
    "team1": "team2",
    "team2": "team1",
    "qb1": "qb2",
    "qb2": "qb1",
    "elo1_pre": "elo2_pre",
    "elo2_pre": "elo1_pre",
    "qbelo1_pre": "qbelo2_pre",
    "qbelo2_pre": "qbelo1_pre",
    "qb1_value_pre": "qb2_value_pre",
    "qb2_value_pre": "qb1_value_pre",
}

# Dictionary to rename ELO columns to away/home format for consistency
ELO_RENAME_COLS = {
    "team1": "home_name",
    "team2": "away_name",
    "qb1": "home_qb",
    "qb2": "away_qb",
    "elo1_pre": "home_elo_pre",
    "elo2_pre": "away_elo_pre",
    "qbelo1_pre": "home_qb_elo_pre",
    "qbelo2_pre": "away_qb_elo_pre",
    "qb1_value_pre": "home_qb_value_pre",
    "qb2_value_pre": "away_qb_value_pre",
}

# Columns to exclude from the NFL lines dataset for streamlined analysis
LINES_DROP_COLS = [
    "Unnamed: 0",
    "game_id",
    "home_spread_open_source",
    "home_spread_open_timestamp",
    "home_spread_last_source",
    "home_spread_last_timestamp",
    "home_spread_tickets_pct",
    "home_spread_money_pct",
    "home_spread_pcts_source",
    "home_spread_pct_timestamp",
    "ml_open_source",
    "ml_open_timestamp",
    "ml_last_source",
    "ml_last_timestamp",
    "total_line_open_source",
    "total_line_open_timestamp",
    "total_line_last_source",
    "total_line_last_timestamp",
]

# Ordered list of columns for the final dataset
FIRST_COLUMNS = [
    "away_name",
    "away_abbr",
    "home_name",
    "home_abbr",
    "away_qb",
    "home_qb",
    "season",
    "week",
    "date",
    "away_game_number",
    "home_game_number",
    "neutral",
    "division",
]
LINES_COLUMNS = [
    "away_spread",
    "away_moneyline",
    "home_spread",
    "home_moneyline",
    "total_line",
]
RESULT_COLUMNS = [
    "away_score",
    "home_score",
    "result",
]

# ============================================================================
# Stadium Location Mapping (stadium_id -> city, state)
# ============================================================================

# Maps nflreadpy stadium_id to city and state/country
# Stadium IDs follow pattern: CITY## or ABBR## where ## is a version number
STADIUM_LOCATIONS = {
    # AFC East
    "BOS00": {"city": "Foxborough", "state": "MA"},  # Gillette Stadium
    "BUF00": {"city": "Orchard Park", "state": "NY"},  # Highmark Stadium/New Era Field
    "BUF01": {"city": "Toronto", "state": "ON"},  # Rogers Centre (Toronto games)
    "MIA00": {"city": "Miami Gardens", "state": "FL"},  # Hard Rock Stadium
    "NYC00": {"city": "East Rutherford", "state": "NJ"},  # Giants Stadium (old)
    "NYC01": {"city": "East Rutherford", "state": "NJ"},  # MetLife Stadium
    # AFC North
    "BAL00": {"city": "Baltimore", "state": "MD"},  # M&T Bank Stadium
    "CIN00": {"city": "Cincinnati", "state": "OH"},  # Paycor Stadium
    "CLE00": {"city": "Cleveland", "state": "OH"},  # Cleveland Browns Stadium
    "PIT00": {"city": "Pittsburgh", "state": "PA"},  # Acrisure Stadium/Heinz Field
    # AFC South
    "HOU00": {"city": "Houston", "state": "TX"},  # NRG Stadium/Reliant Stadium
    "IND00": {"city": "Indianapolis", "state": "IN"},  # Lucas Oil Stadium
    "IND99": {"city": "Indianapolis", "state": "IN"},  # RCA Dome (old)
    "JAX00": {"city": "Jacksonville", "state": "FL"},  # EverBank Stadium
    "NAS00": {"city": "Nashville", "state": "TN"},  # Nissan Stadium/LP Field
    # AFC West
    "DEN00": {"city": "Denver", "state": "CO"},  # Empower Field at Mile High
    "KAN00": {"city": "Kansas City", "state": "MO"},  # Arrowhead Stadium
    "LAX01": {"city": "Inglewood", "state": "CA"},  # SoFi Stadium
    "LAX97": {"city": "Carson", "state": "CA"},  # StubHub Center (Chargers temp)
    "LAX99": {"city": "Los Angeles", "state": "CA"},  # LA Memorial Coliseum (Rams temp)
    "OAK00": {"city": "Oakland", "state": "CA"},  # Oakland Coliseum (historical)
    "SDG00": {"city": "San Diego", "state": "CA"},  # Qualcomm Stadium (historical)
    "VEG00": {"city": "Las Vegas", "state": "NV"},  # Allegiant Stadium
    # NFC East
    "DAL00": {"city": "Arlington", "state": "TX"},  # AT&T Stadium/Cowboys Stadium
    "DAL99": {"city": "Irving", "state": "TX"},  # Texas Stadium (old)
    "PHI00": {"city": "Philadelphia", "state": "PA"},  # Lincoln Financial Field
    "WAS00": {"city": "Landover", "state": "MD"},  # Northwest Stadium/FedEx Field
    # NFC North
    "CHI98": {"city": "Chicago", "state": "IL"},  # Soldier Field
    "DET00": {"city": "Detroit", "state": "MI"},  # Ford Field
    "GNB00": {"city": "Green Bay", "state": "WI"},  # Lambeau Field
    "MIN00": {"city": "Minneapolis", "state": "MN"},  # Metrodome/Mall of America Field
    "MIN01": {"city": "Minneapolis", "state": "MN"},  # U.S. Bank Stadium
    "MIN98": {"city": "Minneapolis", "state": "MN"},  # TCF Bank Stadium (temp)
    # NFC South
    "ATL00": {"city": "Atlanta", "state": "GA"},  # Georgia Dome (old)
    "ATL97": {"city": "Atlanta", "state": "GA"},  # Mercedes-Benz Stadium
    "BRG00": {"city": "Baton Rouge", "state": "LA"},  # Tiger Stadium LSU (Katrina)
    "CAR00": {"city": "Charlotte", "state": "NC"},  # Bank of America Stadium
    "NOR00": {"city": "New Orleans", "state": "LA"},  # Caesars Superdome
    "SAN00": {"city": "San Antonio", "state": "TX"},  # Alamo Dome (neutral/Katrina)
    "TAM00": {"city": "Tampa", "state": "FL"},  # Raymond James Stadium
    # NFC West
    "PHO00": {"city": "Glendale", "state": "AZ"},  # State Farm Stadium
    "PHO99": {"city": "Tempe", "state": "AZ"},  # Sun Devil Stadium (old)
    "SEA00": {"city": "Seattle", "state": "WA"},  # Lumen Field
    "SFO00": {"city": "San Francisco", "state": "CA"},  # Candlestick Park (old)
    "SFO01": {"city": "Santa Clara", "state": "CA"},  # Levi's Stadium
    "STL00": {"city": "St. Louis", "state": "MO"},  # The Dome at America's Center
    # International Venues
    "FRA00": {"city": "Frankfurt", "state": "DE"},  # Deutsche Bank Park
    "GER00": {"city": "Munich", "state": "DE"},  # Allianz Arena
    "LON00": {"city": "London", "state": "UK"},  # Wembley Stadium
    "LON01": {"city": "London", "state": "UK"},  # Twickenham Stadium
    "LON02": {"city": "London", "state": "UK"},  # Tottenham Hotspur Stadium
    "MEX00": {"city": "Mexico City", "state": "MX"},  # Estadio Azteca
    "SAO00": {"city": "São Paulo", "state": "BR"},  # Arena Corinthians
}


# ============================================================================
# Polars Data Collection Column Definitions
# ============================================================================

# Metadata columns (24 total) - includes game info, teams, venue, and conditions
POLARS_METADATA_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",  # REG, WC, DIV, CON, SB
    "date",
    "away_abbr",
    "home_abbr",
    "away_qb",
    "home_qb",
    "away_rest",
    "home_rest",
    "neutral",
    "division",
]

# ELO rating columns (per team) - these get prefixed with away_/home_
POLARS_ELO_COLUMNS = [
    "elo_pre",
    "qb_value_pre",
    "qb_elo_pre",
]

# TeamRankings rating columns (per team) - these get prefixed with away_/home_
POLARS_TR_RATINGS = [
    "predictive_rating",
    "home_rating",
    "away_rating",
    "strength_of_schedule_rating",
    "future_sos_rating",
    "last_5_games_rating",
    "last_10_games_rating",
    "in_division_rating",
    "non_division_rating",
    "luck_rating",
]

# TeamRankings stat columns (per team) - these get prefixed with away_/home_
# NOTE: Only includes stats we must scrape from TR (can't calculate from nflreadpy)
POLARS_TR_STATS = [
    "third_down_pct",
    "opponent_third_down_pct",
    "fourth_down_pct",
    "opponent_fourth_down_pct",
    "red_zone_td_pct",
    "opponent_red_zone_td_pct",
]

# Calculated stats (computed from nflreadpy data during aggregation)
# These were previously scraped from TR but are now calculated to save scraping time
POLARS_CALCULATED_STATS = [
    # Per-game averages (stats already aggregated as means)
    "points_scored",  # Already in NFLREADPY_STATS - avg points per game
    "points_allowed",  # Already in NFLREADPY_STATS - avg points allowed per game
    "scoring_margin",  # Already in NFLREADPY_STATS - avg scoring margin per game
    "turnover_margin",  # Already in NFLREADPY_STATS - avg turnover margin per game
    "penalty_yards",  # Already in NFLREADPY_STATS - avg penalty yards per game
    # Ratio metrics (computed in _compute_derived_metrics)
    "yards_per_point",  # total_yards / points_scored
    "opponent_yards_per_point",  # opponent_total_yards / points_allowed
    "yards_per_point_margin",  # yards_per_point - opponent_yards_per_point
    "points_per_play",  # points_scored / total_plays
    "opponent_points_per_play",  # points_allowed / opponent_total_plays
    "points_per_play_margin",  # points_per_play - opponent_points_per_play
    "penalty_yards_per_penalty",  # penalty_yards / penalties
    "opponent_penalty_yards_per_penalty",  # opp_penalty_yards / opp_penalties
]

# nflreadpy stats to use (per team) - these get prefixed with away_/home_
POLARS_NFLREADPY_STATS = [
    # Passing offense
    "pass_completions",
    "pass_attempts",
    "pass_yards",
    "pass_touchdowns",
    "interceptions_thrown",
    "times_sacked",
    "passing_epa",
    "passing_cpoe",
    # Rushing offense
    "rush_attempts",
    "rush_yards",
    "rush_touchdowns",
    # Receiving (team level)
    "receptions",
    "receiving_touchdowns",
    # Combined stats
    "fumbles",  # Combined: sack + rushing + receiving fumbles
    "fumbles_lost",  # Combined: sack + rushing + receiving fumbles_lost
    "first_downs",  # Combined: passing + rushing first_downs (not receiving)
    "2pt_conversions",  # Combined: passing + rushing + receiving 2pt_conversions
    "turnover_margin",  # Computed: (def_interceptions + fumble_recovery_opp) - (INT + fumbles_lost)
    "total_yards",  # Computed: pass_yards + rush_yards - sack_yards_lost
    # Defense stats
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_tds",
    "def_fumbles",
    "def_safeties",
    "fumble_recoveries",  # Combined: fumble_recovery_own + fumble_recovery_opp
    "fumble_recovery_tds",
    # Penalties
    "penalties",
    "penalty_yards",
    # Special teams
    "special_teams_tds",
    # Scoring (from schedule)
    "points_scored",  # Team's score for the game
    "points_allowed",  # Opponent's score for the game
    "scoring_margin",  # Computed: points_scored - points_allowed
    # Derived ratio metrics (computed during aggregation in _compute_derived_metrics)
    "yards_per_point",  # Computed: total_yards / points_scored
    "opponent_yards_per_point",  # Computed: opponent_total_yards / points_allowed
    "yards_per_point_margin",  # Computed: yards_per_point - opponent_yards_per_point
    "points_per_play",  # Computed: points_scored / total_plays
    "opponent_points_per_play",  # Computed: points_allowed / opponent_total_plays
    "points_per_play_margin",  # Computed: points_per_play - opponent_points_per_play
    "penalty_yards_per_penalty",  # Computed: penalty_yards / penalties
    "opponent_penalty_yards_per_penalty",  # Computed: opponent equivalent
]

# Lines/Odds columns (5 total)
POLARS_LINES_COLUMNS = [
    "total_line",
    "home_spread",
    "away_spread",
    "away_moneyline",
    "home_moneyline",
]

# Result columns (2 total)
POLARS_RESULT_COLUMNS = [
    "away_score",
    "home_score",
]

# Regression factor for week-1 stats toward league mean
WEEK1_REGRESSION_FACTOR = 1 / 3

# Active quarterback IDs: names, draft years, and numbers for player tracking
ACTIVE_QB_IDS = {
    "00-0007059": {"draft_number": 263, "draft_year": 2000, "name": "Mark Hartsell"},
    "00-0010206": {"draft_number": 263, "draft_year": 2000, "name": "Matt Lytle"},
    "00-0013186": {"draft_number": 263, "draft_year": 2000, "name": "Ron Powlus"},
    "00-0013901": {"draft_number": 263, "draft_year": 2001, "name": "Rod Robinson"},
    "00-0019041": {"draft_number": 263, "draft_year": 2000, "name": "Billy Volek"},
    "00-0019087": {"draft_number": 263, "draft_year": 2000, "name": "Doug Johnson"},
    "00-0019192": {"draft_number": 263, "draft_year": 2000, "name": "Travis Brown"},
    "00-0019278": {"draft_number": 263, "draft_year": 2000, "name": "Kevin Thompson"},
    "00-0019295": {"draft_number": 263, "draft_year": 2000, "name": "Sean Keenan"},
    "00-0019359": {"draft_number": 263, "draft_year": 2001, "name": "Phil Stambaugh"},
    "00-0019442": {"draft_number": 263, "draft_year": 2000, "name": "Clint Stoerner"},
    "00-0019553": {"draft_number": 183, "draft_year": 2000, "name": "Spergon Wynn"},
    "00-0019559": {"draft_number": 18, "draft_year": 2000, "name": "Chad Pennington"},
    "00-0019579": {"draft_number": 234, "draft_year": 2000, "name": "Joe Hamilton"},
    "00-0019596": {"draft_number": 199, "draft_year": 2000, "name": "Tom Brady"},
    "00-0019599": {"draft_number": 168, "draft_year": 2001, "name": "Marc Bulger"},
    "00-0019622": {"draft_number": 202, "draft_year": 2000, "name": "Todd Husak"},
    "00-0019623": {"draft_number": 212, "draft_year": 2000, "name": "Tim Rattay"},
    "00-0019633": {"draft_number": 163, "draft_year": 2000, "name": "Tee Martin"},
    "00-0019658": {"draft_number": 263, "draft_year": 2000, "name": "Giovanni Carmazzi"},
    "00-0019682": {"draft_number": 214, "draft_year": 2000, "name": "Jarious Jackson"},
    "00-0019709": {"draft_number": 75, "draft_year": 2000, "name": "Chris Redman"},
    "00-0019764": {"draft_number": 263, "draft_year": 2001, "name": "Henry Burris"},
    "00-0019765": {"draft_number": 263, "draft_year": 2001, "name": "Dave Dickenson"},
    "00-0019816": {"draft_number": 263, "draft_year": 2002, "name": "Tim Hasselbeck"},
    "00-0019957": {"draft_number": 263, "draft_year": 2004, "name": "David Rivers"},
    "00-0020126": {"draft_number": 263, "draft_year": 2001, "name": "Tory Woodbury"},
    "00-0020245": {"draft_number": 1, "draft_year": 2001, "name": "Mike Vick"},
    "00-0020246": {"draft_number": 263, "draft_year": 2001, "name": "Romaro Miller"},
    "00-0020305": {"draft_number": 155, "draft_year": 2001, "name": "A.J. Feeley"},
    "00-0020308": {"draft_number": 149, "draft_year": 2001, "name": "Mike McMahon"},
    "00-0020404": {"draft_number": 53, "draft_year": 2001, "name": "Quincy Carter"},
    "00-0020405": {"draft_number": 59, "draft_year": 2001, "name": "Marques Tuiasosopo"},
    "00-0020434": {"draft_number": 106, "draft_year": 2001, "name": "Chris Weinke"},
    "00-0020436": {"draft_number": 263, "draft_year": 2001, "name": "Josh Heupel"},
    "00-0020480": {"draft_number": 109, "draft_year": 2001, "name": "Sage Rosenfels"},
    "00-0020483": {"draft_number": 125, "draft_year": 2001, "name": "Jesse Palmer"},
    "00-0020486": {"draft_number": 263, "draft_year": 2001, "name": "Josh Booty"},
    "00-0020511": {"draft_number": 263, "draft_year": 2001, "name": "Ricky Ray"},
    "00-0020531": {"draft_number": 32, "draft_year": 2001, "name": "Drew Brees"},
    "00-0020562": {"draft_number": 263, "draft_year": 2002, "name": "Chad Hutchinson"},
    "00-0020565": {"draft_number": 263, "draft_year": 2004, "name": "Cleo Lemon"},
    "00-0020608": {"draft_number": 1, "draft_year": 2002, "name": "David Carr"},
    "00-0020613": {"draft_number": 263, "draft_year": 2003, "name": "Quinn Gray"},
    "00-0020679": {"draft_number": 263, "draft_year": 2002, "name": "Shaun Hill"},
    "00-0021062": {"draft_number": 263, "draft_year": 2002, "name": "Preston Parsons"},
    "00-0021141": {"draft_number": 3, "draft_year": 2002, "name": "Joey Harrington"},
    "00-0021206": {"draft_number": 81, "draft_year": 2002, "name": "Josh McCown"},
    "00-0021231": {"draft_number": 108, "draft_year": 2002, "name": "David Garrard"},
    "00-0021238": {"draft_number": 117, "draft_year": 2002, "name": "Rohan Davey"},
    "00-0021254": {"draft_number": 137, "draft_year": 2002, "name": "Randy Fasani"},
    "00-0021271": {"draft_number": 158, "draft_year": 2002, "name": "Kurt Kittner"},
    "00-0021276": {"draft_number": 263, "draft_year": 2002, "name": "Brandon Doman"},
    "00-0021277": {"draft_number": 164, "draft_year": 2002, "name": "Craig Nall"},
    "00-0021296": {"draft_number": 186, "draft_year": 2002, "name": "J.T. O'Sullivan"},
    "00-0021322": {"draft_number": 263, "draft_year": 2002, "name": "Seth Burford"},
    "00-0021334": {"draft_number": 232, "draft_year": 2002, "name": "Jeff Kelly"},
    "00-0021338": {"draft_number": 263, "draft_year": 2003, "name": "Wes Pate"},
    "00-0021379": {"draft_number": 32, "draft_year": 2002, "name": "Patrick Ramsey"},
    "00-0021429": {"draft_number": 1, "draft_year": 2003, "name": "Carson Palmer"},
    "00-0021592": {"draft_number": 263, "draft_year": 2003, "name": "Nate Hybl"},
    "00-0021640": {"draft_number": 263, "draft_year": 2003, "name": "Tom Arth"},
    "00-0021678": {"draft_number": 263, "draft_year": 2003, "name": "Tony Romo"},
    "00-0021720": {"draft_number": 263, "draft_year": 2003, "name": "Marquel Blackwell"},
    "00-0021759": {"draft_number": 263, "draft_year": 2003, "name": "Jason Gesser"},
    "00-0021822": {"draft_number": 263, "draft_year": 2003, "name": "Kirk Farmer"},
    "00-0022005": {"draft_number": 97, "draft_year": 2003, "name": "Chris Simms"},
    "00-0022026": {"draft_number": 200, "draft_year": 2003, "name": "Brooks Bollinger"},
    "00-0022027": {"draft_number": 201, "draft_year": 2003, "name": "Kliff Kingsbury"},
    "00-0022039": {"draft_number": 241, "draft_year": 2003, "name": "Ken Dorsey"},
    "00-0022055": {"draft_number": 110, "draft_year": 2003, "name": "Seneca Wallace"},
    "00-0022091": {"draft_number": 88, "draft_year": 2003, "name": "Dave Ragone"},
    "00-0022101": {"draft_number": 163, "draft_year": 2003, "name": "Brian St. Pierre"},
    "00-0022112": {"draft_number": 232, "draft_year": 2003, "name": "Gibran Hamdan"},
    "00-0022121": {"draft_number": 22, "draft_year": 2003, "name": "Rex Grossman"},
    "00-0022164": {"draft_number": 19, "draft_year": 2003, "name": "Kyle Boller"},
    "00-0022177": {"draft_number": 7, "draft_year": 2003, "name": "Byron Leftwich"},
    "00-0022181": {"draft_number": 192, "draft_year": 2004, "name": "Drew Henson"},
    "00-0022330": {"draft_number": 263, "draft_year": 2006, "name": "Jason Fife"},
    "00-0022373": {"draft_number": 263, "draft_year": 2004, "name": "Rod Rutherford"},
    "00-0022590": {"draft_number": 263, "draft_year": 2005, "name": "Jared Lorenzen"},
    "00-0022649": {"draft_number": 263, "draft_year": 2006, "name": "Bryson Spinner"},
    "00-0022720": {"draft_number": 263, "draft_year": 2004, "name": "B.J. Symons"},
    "00-0022731": {"draft_number": 217, "draft_year": 2004, "name": "Cody Pickett"},
    "00-0022744": {"draft_number": 193, "draft_year": 2004, "name": "Jim Sorgi"},
    "00-0022761": {"draft_number": 185, "draft_year": 2005, "name": "Andy Hall"},
    "00-0022765": {"draft_number": 263, "draft_year": 2004, "name": "Jeff Smoker"},
    "00-0022768": {"draft_number": 225, "draft_year": 2004, "name": "Matt Mauck"},
    "00-0022774": {"draft_number": 250, "draft_year": 2005, "name": "Bradlee Van Pelt"},
    "00-0022787": {"draft_number": 90, "draft_year": 2004, "name": "Matt Schaub"},
    "00-0022800": {"draft_number": 263, "draft_year": 2004, "name": "Josh Harris"},
    "00-0022803": {"draft_number": 1, "draft_year": 2004, "name": "Eli Manning"},
    "00-0022818": {"draft_number": 148, "draft_year": 2004, "name": "Craig Krenzel"},
    "00-0022827": {"draft_number": 202, "draft_year": 2004, "name": "John Navarre"},
    "00-0022829": {"draft_number": 263, "draft_year": 2004, "name": "Casey Bramlet"},
    "00-0022864": {"draft_number": 106, "draft_year": 2004, "name": "Luke McCown"},
    "00-0022912": {"draft_number": 22, "draft_year": 2004, "name": "J.P. Losman"},
    "00-0022924": {"draft_number": 11, "draft_year": 2004, "name": "Ben Roethlisberger"},
    "00-0022942": {"draft_number": 4, "draft_year": 2004, "name": "Philip Rivers"},
    "00-0023126": {"draft_number": 263, "draft_year": 2006, "name": "Shane Boyd"},
    "00-0023145": {"draft_number": 263, "draft_year": 2005, "name": "Marcus Randall"},
    "00-0023158": {"draft_number": 263, "draft_year": 2007, "name": "Brock Berlin"},
    "00-0023237": {"draft_number": 263, "draft_year": 2006, "name": "Bryan Randall"},
    "00-0023306": {"draft_number": 263, "draft_year": 2006, "name": "Lang Campbell"},
    "00-0023373": {"draft_number": 263, "draft_year": 2006, "name": "Craig Ochs"},
    "00-0023431": {"draft_number": 263, "draft_year": 2007, "name": "Kevin Eakin"},
    "00-0023436": {"draft_number": 263, "draft_year": 2005, "name": "Alex Smith"},
    "00-0023459": {"draft_number": 24, "draft_year": 2005, "name": "Aaron Rodgers"},
    "00-0023460": {"draft_number": 25, "draft_year": 2005, "name": "Jason Campbell"},
    "00-0023502": {"draft_number": 67, "draft_year": 2005, "name": "Charlie Frye"},
    "00-0023504": {"draft_number": 69, "draft_year": 2005, "name": "Andrew Walter"},
    "00-0023520": {"draft_number": 263, "draft_year": 2005, "name": "David Greene"},
    "00-0023541": {"draft_number": 106, "draft_year": 2005, "name": "Kyle Orton"},
    "00-0023555": {"draft_number": 263, "draft_year": 2005, "name": "Stefan LeFors"},
    "00-0023578": {"draft_number": 145, "draft_year": 2005, "name": "Dan Orlovsky"},
    "00-0023585": {"draft_number": 263, "draft_year": 2005, "name": "Adrian McPherson"},
    "00-0023645": {"draft_number": 213, "draft_year": 2005, "name": "Derek Anderson"},
    "00-0023662": {"draft_number": 230, "draft_year": 2005, "name": "Matt Cassel"},
    "00-0023682": {"draft_number": 250, "draft_year": 2005, "name": "Ryan Fitzpatrick"},
    "00-0023739": {"draft_number": 263, "draft_year": 2006, "name": "Casey Printers"},
    "00-0023769": {"draft_number": 263, "draft_year": 2007, "name": "Bruce Eugene"},
    "00-0023838": {"draft_number": 263, "draft_year": 2006, "name": "Matt Baker"},
    "00-0023859": {"draft_number": 263, "draft_year": 2006, "name": "Brett Basanez"},
    "00-0023899": {"draft_number": 263, "draft_year": 2007, "name": "Travis Lulay"},
    "00-0023923": {"draft_number": 263, "draft_year": 2006, "name": "Josh Betts"},
    "00-0023943": {"draft_number": 263, "draft_year": 2007, "name": "Darrell Hackney"},
    "00-0023946": {"draft_number": 263, "draft_year": 2006, "name": "Kent Smith"},
    "00-0024059": {"draft_number": 263, "draft_year": 2006, "name": "Quinton Porter"},
    "00-0024074": {"draft_number": 263, "draft_year": 2006, "name": "Brett Elliott"},
    "00-0024149": {"draft_number": 263, "draft_year": 2006, "name": "Drew Olson"},
    "00-0024210": {"draft_number": 263, "draft_year": 2007, "name": "Jeff Otis"},
    "00-0024218": {"draft_number": 3, "draft_year": 2006, "name": "Vince Young"},
    "00-0024225": {"draft_number": 10, "draft_year": 2006, "name": "Matt Leinart"},
    "00-0024226": {"draft_number": 11, "draft_year": 2006, "name": "Jay Cutler"},
    "00-0024264": {"draft_number": 49, "draft_year": 2006, "name": "Kellen Clemens"},
    "00-0024279": {"draft_number": 64, "draft_year": 2006, "name": "Tarvaris Jackson"},
    "00-0024296": {"draft_number": 81, "draft_year": 2006, "name": "Charlie Whitehurst"},
    "00-0024300": {"draft_number": 85, "draft_year": 2006, "name": "Brodie Croyle"},
    "00-0024362": {"draft_number": 148, "draft_year": 2006, "name": "Ingle Martin"},
    "00-0024378": {"draft_number": 263, "draft_year": 2006, "name": "Omar Jacobs"},
    "00-0024408": {"draft_number": 194, "draft_year": 2006, "name": "Bruce Gradkowski"},
    "00-0024437": {"draft_number": 263, "draft_year": 2006, "name": "D.J. Shockley"},
    "00-0024560": {"draft_number": 263, "draft_year": 2007, "name": "Sam Hollenbach"},
    "00-0024652": {"draft_number": 263, "draft_year": 2007, "name": "Dalton Bell"},
    "00-0024812": {"draft_number": 263, "draft_year": 2007, "name": "Tyler Palko"},
    "00-0024824": {"draft_number": 263, "draft_year": 2007, "name": "Matt Gutierrez"},
    "00-0024836": {"draft_number": 263, "draft_year": 2007, "name": "Jared Zabransky"},
    "00-0025388": {"draft_number": 1, "draft_year": 2007, "name": "JaMarcus Russell"},
    "00-0025409": {"draft_number": 22, "draft_year": 2007, "name": "Brady Quinn"},
    "00-0025423": {"draft_number": 36, "draft_year": 2007, "name": "Kevin Kolb"},
    "00-0025427": {"draft_number": 40, "draft_year": 2007, "name": "John Beck"},
    "00-0025430": {"draft_number": 43, "draft_year": 2007, "name": "Drew Stanton"},
    "00-0025479": {"draft_number": 92, "draft_year": 2007, "name": "Trent Edwards"},
    "00-0025538": {"draft_number": 151, "draft_year": 2007, "name": "Jeff Rowe"},
    "00-0025561": {"draft_number": 174, "draft_year": 2007, "name": "Troy Smith"},
    "00-0025592": {"draft_number": 205, "draft_year": 2008, "name": "Jordan Palmer"},
    "00-0025604": {"draft_number": 217, "draft_year": 2007, "name": "Tyler Thigpen"},
    "00-0025664": {"draft_number": 263, "draft_year": 2007, "name": "Lester Ricard"},
    "00-0025694": {"draft_number": 263, "draft_year": 2007, "name": "Brett Ratliff"},
    "00-0025708": {"draft_number": 263, "draft_year": 2007, "name": "Matt Moore"},
    "00-0025749": {"draft_number": 263, "draft_year": 2007, "name": "Richard Bartel"},
    "00-0025759": {"draft_number": 263, "draft_year": 2007, "name": "Cullen Finnerty"},
    "00-0025766": {"draft_number": 263, "draft_year": 2007, "name": "Paul Thompson"},
    "00-0025970": {"draft_number": 263, "draft_year": 2008, "name": "Caleb Hanie"},
    "00-0026143": {"draft_number": 3, "draft_year": 2008, "name": "Matt Ryan"},
    "00-0026158": {"draft_number": 18, "draft_year": 2008, "name": "Joe Flacco"},
    "00-0026196": {"draft_number": 56, "draft_year": 2008, "name": "Brian Brohm"},
    "00-0026197": {"draft_number": 57, "draft_year": 2008, "name": "Chad Henne"},
    "00-0026234": {"draft_number": 94, "draft_year": 2008, "name": "Kevin O'Connell"},
    "00-0026277": {"draft_number": 263, "draft_year": 2008, "name": "John David Booty"},
    "00-0026296": {"draft_number": 156, "draft_year": 2008, "name": "Dennis Dixon"},
    "00-0026300": {"draft_number": 263, "draft_year": 2008, "name": "Josh Johnson"},
    "00-0026302": {"draft_number": 162, "draft_year": 2008, "name": "Erik Ainge"},
    "00-0026326": {"draft_number": 186, "draft_year": 2008, "name": "Colt Brennan"},
    "00-0026338": {"draft_number": 198, "draft_year": 2008, "name": "Andre Woodson"},
    "00-0026349": {"draft_number": 209, "draft_year": 2008, "name": "Matt Flynn"},
    "00-0026363": {"draft_number": 223, "draft_year": 2008, "name": "Alex Brink"},
    "00-0026443": {"draft_number": 263, "draft_year": 2008, "name": "Paul Smith"},
    "00-0026498": {"draft_number": 1, "draft_year": 2009, "name": "Matthew Stafford"},
    "00-0026505": {"draft_number": 263, "draft_year": 2009, "name": "John Parker Wilson"},
    "00-0026544": {"draft_number": 263, "draft_year": 2009, "name": "Chase Daniel"},
    "00-0026567": {"draft_number": 263, "draft_year": 2009, "name": "Mike Reilly"},
    "00-0026625": {"draft_number": 263, "draft_year": 2009, "name": "Brian Hoyer"},
    "00-0026657": {"draft_number": 263, "draft_year": 2009, "name": "Chris Pizzotti"},
    "00-0026702": {"draft_number": 263, "draft_year": 2009, "name": "Hunter Cantwell"},
    "00-0026775": {"draft_number": 263, "draft_year": 2009, "name": "Rudy Carpenter"},
    "00-0026865": {"draft_number": 263, "draft_year": 2009, "name": "Drew Willy"},
    "00-0026898": {"draft_number": 5, "draft_year": 2009, "name": "Mark Sanchez"},
    "00-0026916": {"draft_number": 151, "draft_year": 2009, "name": "Rhett Bomar"},
    "00-0026927": {"draft_number": 178, "draft_year": 2009, "name": "Mike Teel"},
    "00-0026993": {"draft_number": 17, "draft_year": 2009, "name": "Josh Freeman"},
    "00-0027020": {"draft_number": 44, "draft_year": 2009, "name": "Pat White"},
    "00-0027071": {"draft_number": 101, "draft_year": 2009, "name": "Stephen McGee"},
    "00-0027118": {"draft_number": 171, "draft_year": 2009, "name": "Nate Davis"},
    "00-0027121": {"draft_number": 174, "draft_year": 2009, "name": "Tom Brandstater"},
    "00-0027131": {"draft_number": 196, "draft_year": 2009, "name": "Keith Null"},
    "00-0027134": {"draft_number": 201, "draft_year": 2009, "name": "Curtis Painter"},
    "00-0027253": {"draft_number": 263, "draft_year": 2010, "name": "Thad Lewis"},
    "00-0027347": {"draft_number": 263, "draft_year": 2010, "name": "Max Hall"},
    "00-0027376": {"draft_number": 263, "draft_year": 2011, "name": "Jarrett Brown"},
    "00-0027384": {"draft_number": 263, "draft_year": 2010, "name": "R.J. Archer"},
    "00-0027602": {"draft_number": 263, "draft_year": 2010, "name": "Graham Harrell"},
    "00-0027605": {"draft_number": 181, "draft_year": 2010, "name": "Dan LeFevour"},
    "00-0027659": {"draft_number": 48, "draft_year": 2010, "name": "Jimmy Clausen"},
    "00-0027688": {"draft_number": 85, "draft_year": 2010, "name": "Colt McCoy"},
    "00-0027722": {"draft_number": 122, "draft_year": 2010, "name": "Mike Kafka"},
    "00-0027754": {"draft_number": 155, "draft_year": 2010, "name": "John Skelton"},
    "00-0027767": {"draft_number": 168, "draft_year": 2010, "name": "Jonathan Crompton"},
    "00-0027775": {"draft_number": 176, "draft_year": 2010, "name": "Rusty Smith"},
    "00-0027796": {"draft_number": 199, "draft_year": 2010, "name": "Joe Webb"},
    "00-0027801": {"draft_number": 204, "draft_year": 2010, "name": "Tony Pike"},
    "00-0027804": {"draft_number": 209, "draft_year": 2010, "name": "Levi Brown"},
    "00-0027830": {"draft_number": 239, "draft_year": 2010, "name": "Sean Canfield"},
    "00-0027841": {"draft_number": 250, "draft_year": 2010, "name": "Zac Robinson"},
    "00-0027854": {"draft_number": 1, "draft_year": 2010, "name": "Sam Bradford"},
    "00-0027876": {"draft_number": 25, "draft_year": 2010, "name": "Tim Tebow"},
    "00-0027931": {"draft_number": 263, "draft_year": 2011, "name": "Ryan Perrilloux"},
    "00-0027939": {"draft_number": 1, "draft_year": 2011, "name": "Cam Newton"},
    "00-0027946": {"draft_number": 8, "draft_year": 2011, "name": "Jake Locker"},
    "00-0027948": {"draft_number": 10, "draft_year": 2011, "name": "Blaine Gabbert"},
    "00-0027950": {"draft_number": 12, "draft_year": 2011, "name": "Christian Ponder"},
    "00-0027973": {"draft_number": 35, "draft_year": 2011, "name": "Andy Dalton"},
    "00-0027974": {"draft_number": 36, "draft_year": 2011, "name": "Colin Kaepernick"},
    "00-0028012": {"draft_number": 74, "draft_year": 2011, "name": "Ryan Mallett"},
    "00-0028073": {"draft_number": 135, "draft_year": 2011, "name": "Ricky Stanzi"},
    "00-0028090": {"draft_number": 152, "draft_year": 2011, "name": "T.J. Yates"},
    "00-0028098": {"draft_number": 160, "draft_year": 2011, "name": "Nathan Enderle"},
    "00-0028118": {"draft_number": 180, "draft_year": 2011, "name": "Tyrod Taylor"},
    "00-0028146": {"draft_number": 208, "draft_year": 2011, "name": "Greg McElroy"},
    "00-0028230": {"draft_number": 263, "draft_year": 2011, "name": "Adam Weber"},
    "00-0028359": {"draft_number": 263, "draft_year": 2011, "name": "McLeod Bethel-Thompson"},
    "00-0028370": {"draft_number": 263, "draft_year": 2013, "name": "Jerrod Johnson"},
    "00-0028446": {"draft_number": 263, "draft_year": 2011, "name": "Josh Portis"},
    "00-0028453": {"draft_number": 263, "draft_year": 2011, "name": "Pat Devlin"},
    "00-0028508": {"draft_number": 263, "draft_year": 2011, "name": "Joshua Nesbitt"},
    "00-0028595": {"draft_number": 263, "draft_year": 2011, "name": "Scott Tolzien"},
    "00-0028647": {"draft_number": 263, "draft_year": 2011, "name": "Mike Hartline"},
    "00-0028863": {"draft_number": 263, "draft_year": 2012, "name": "Dominique Davis"},
    "00-0028957": {"draft_number": 263, "draft_year": 2012, "name": "Austin Davis"},
    "00-0028986": {"draft_number": 263, "draft_year": 2012, "name": "Case Keenum"},
    "00-0029041": {"draft_number": 263, "draft_year": 2013, "name": "G.J. Kinne"},
    "00-0029133": {"draft_number": 263, "draft_year": 2013, "name": "Nick Stephens"},
    "00-0029151": {"draft_number": 263, "draft_year": 2013, "name": "Matt Simms"},
    "00-0029219": {"draft_number": 243, "draft_year": 2012, "name": "B.J. Coleman"},
    "00-0029234": {"draft_number": 263, "draft_year": 2012, "name": "Kellen Moore"},
    "00-0029263": {"draft_number": 75, "draft_year": 2012, "name": "Russell Wilson"},
    "00-0029495": {"draft_number": 263, "draft_year": 2012, "name": "Matt Blanchard"},
    "00-0029531": {"draft_number": 185, "draft_year": 2012, "name": "Ryan Lindley"},
    "00-0029554": {"draft_number": 253, "draft_year": 2012, "name": "Chandler Harnish"},
    "00-0029567": {"draft_number": 88, "draft_year": 2012, "name": "Nick Foles"},
    "00-0029604": {"draft_number": 102, "draft_year": 2012, "name": "Kirk Cousins"},
    "00-0029623": {"draft_number": 263, "draft_year": 2012, "name": "Alex Tanney"},
    "00-0029665": {"draft_number": 263, "draft_year": 2012, "name": "Robert Griffin III"},
    "00-0029668": {"draft_number": 1, "draft_year": 2012, "name": "Andrew Luck"},
    "00-0029677": {"draft_number": 22, "draft_year": 2012, "name": "Brandon Weeden"},
    "00-0029682": {"draft_number": 57, "draft_year": 2012, "name": "Brock Osweiler"},
    "00-0029701": {"draft_number": 8, "draft_year": 2012, "name": "Ryan Tannehill"},
    "00-0029772": {"draft_number": 263, "draft_year": 2013, "name": "Seth Doege"},
    "00-0029857": {"draft_number": 263, "draft_year": 2013, "name": "Ryan Griffin"},
    "00-0029956": {"draft_number": 263, "draft_year": 2013, "name": "Jordan Rodgers"},
    "00-0029957": {"draft_number": 263, "draft_year": 2013, "name": "Matt Scott"},
    "00-0030024": {"draft_number": 263, "draft_year": 2013, "name": "Jeff Tuel"},
    "00-0030151": {"draft_number": 263, "draft_year": 2013, "name": "Tyler Bray"},
    "00-0030292": {"draft_number": 221, "draft_year": 2013, "name": "Brad Sorensen"},
    "00-0030394": {"draft_number": 234, "draft_year": 2013, "name": "Zac Dysert"},
    "00-0030419": {"draft_number": 263, "draft_year": 2013, "name": "Matt McGloin"},
    "00-0030520": {"draft_number": 73, "draft_year": 2013, "name": "Mike Glennon"},
    "00-0030524": {"draft_number": 115, "draft_year": 2013, "name": "Landry Jones"},
    "00-0030526": {"draft_number": 16, "draft_year": 2013, "name": "EJ Manuel"},
    "00-0030533": {"draft_number": 98, "draft_year": 2013, "name": "Matt Barkley"},
    "00-0030565": {"draft_number": 39, "draft_year": 2013, "name": "Geno Smith"},
    "00-0030568": {"draft_number": 112, "draft_year": 2013, "name": "Tyler Wilson"},
    "00-0030569": {"draft_number": 249, "draft_year": 2013, "name": "Sean Renfree"},
    "00-0030586": {"draft_number": 110, "draft_year": 2013, "name": "Ryan Nassib"},
    "00-0030661": {"draft_number": 263, "draft_year": 2014, "name": "Connor Shaw"},
    "00-0030673": {"draft_number": 263, "draft_year": 2015, "name": "Bryn Renner"},
    "00-0030738": {"draft_number": 263, "draft_year": 2014, "name": "Jeff Mathews"},
    "00-0030825": {"draft_number": 263, "draft_year": 2014, "name": "Stephen Morris"},
    "00-0030846": {"draft_number": 263, "draft_year": 2014, "name": "Seth Lobato"},
    "00-0030970": {"draft_number": 263, "draft_year": 2014, "name": "Dustin Vaughan"},
    "00-0030998": {"draft_number": 194, "draft_year": 2014, "name": "Keith Wenning"},
    "00-0031064": {"draft_number": 135, "draft_year": 2014, "name": "Tom Savage"},
    "00-0031076": {"draft_number": 183, "draft_year": 2014, "name": "David Fales"},
    "00-0031237": {"draft_number": 32, "draft_year": 2014, "name": "Teddy Bridgewater"},
    "00-0031266": {"draft_number": 178, "draft_year": 2014, "name": "Zach Mettenberger"},
    "00-0031280": {"draft_number": 36, "draft_year": 2014, "name": "Derek Carr"},
    "00-0031287": {"draft_number": 163, "draft_year": 2014, "name": "Aaron Murray"},
    "00-0031288": {"draft_number": 164, "draft_year": 2014, "name": "AJ McCarron"},
    "00-0031345": {"draft_number": 62, "draft_year": 2014, "name": "Jimmy Garoppolo"},
    "00-0031395": {"draft_number": 214, "draft_year": 2014, "name": "Garrett Gilbert"},
    "00-0031407": {"draft_number": 3, "draft_year": 2014, "name": "Blake Bortles"},
    "00-0031409": {"draft_number": 22, "draft_year": 2014, "name": "Johnny Manziel"},
    "00-0031503": {"draft_number": 1, "draft_year": 2015, "name": "Jameis Winston"},
    "00-0031510": {"draft_number": 263, "draft_year": 2015, "name": "Dylan Thompson"},
    "00-0031568": {"draft_number": 103, "draft_year": 2015, "name": "Bryce Petty"},
    "00-0031589": {"draft_number": 147, "draft_year": 2015, "name": "Brett Hundley"},
    "00-0031800": {"draft_number": 263, "draft_year": 2015, "name": "Taylor Heinicke"},
    "00-0032073": {"draft_number": 263, "draft_year": 2016, "name": "Jake Heaps"},
    "00-0032148": {"draft_number": 75, "draft_year": 2015, "name": "Garrett Grayson"},
    "00-0032156": {"draft_number": 250, "draft_year": 2015, "name": "Trevor Siemian"},
    "00-0032245": {"draft_number": 89, "draft_year": 2015, "name": "Sean Mannion"},
    "00-0032268": {"draft_number": 2, "draft_year": 2015, "name": "Marcus Mariota"},
    "00-0032431": {"draft_number": 191, "draft_year": 2016, "name": "Jake Rudock"},
    "00-0032434": {"draft_number": 201, "draft_year": 2016, "name": "Brandon Allen"},
    "00-0032436": {"draft_number": 207, "draft_year": 2016, "name": "Jeff Driskel"},
    "00-0032446": {"draft_number": 223, "draft_year": 2016, "name": "Brandon Doughty"},
    "00-0032462": {"draft_number": 263, "draft_year": 2016, "name": "Trevone Boykin"},
    "00-0032614": {"draft_number": 263, "draft_year": 2016, "name": "Joel Stave"},
    "00-0032630": {"draft_number": 263, "draft_year": 2016, "name": "Joe Callahan"},
    "00-0032658": {"draft_number": 263, "draft_year": 2016, "name": "Josh Woodrum"},
    "00-0032784": {"draft_number": 162, "draft_year": 2016, "name": "Kevin Hogan"},
    "00-0032792": {"draft_number": 187, "draft_year": 2016, "name": "Nate Sudfeld"},
    "00-0032893": {"draft_number": 100, "draft_year": 2016, "name": "Connor Cook"},
    "00-0032901": {"draft_number": 263, "draft_year": 2017, "name": "Mike Bercovici"},
    "00-0032950": {"draft_number": 2, "draft_year": 2016, "name": "Carson Wentz"},
    "00-0032953": {"draft_number": 51, "draft_year": 2016, "name": "Christian Hackenberg"},
    "00-0033077": {"draft_number": 135, "draft_year": 2016, "name": "Dak Prescott"},
    "00-0033098": {"draft_number": 139, "draft_year": 2016, "name": "Cardale Jones"},
    "00-0033104": {"draft_number": 93, "draft_year": 2016, "name": "Cody Kessler"},
    "00-0033106": {"draft_number": 1, "draft_year": 2016, "name": "Jared Goff"},
    "00-0033108": {"draft_number": 26, "draft_year": 2016, "name": "Paxton Lynch"},
    "00-0033119": {"draft_number": 91, "draft_year": 2016, "name": "Jacoby Brissett"},
    "00-0033238": {"draft_number": 263, "draft_year": 2017, "name": "Alek Torgersen"},
    "00-0033275": {"draft_number": 263, "draft_year": 2017, "name": "PJ Walker"},
    "00-0033319": {"draft_number": 263, "draft_year": 2017, "name": "Nick Mullens"},
    "00-0033537": {"draft_number": 12, "draft_year": 2017, "name": "Deshaun Watson"},
    "00-0033550": {"draft_number": 87, "draft_year": 2017, "name": "Davis Webb"},
    "00-0033585": {"draft_number": 215, "draft_year": 2017, "name": "Brad Kaaya"},
    "00-0033597": {"draft_number": 253, "draft_year": 2017, "name": "Chad Kelly"},
    "00-0033607": {"draft_number": 263, "draft_year": 2017, "name": "Trevor Knight"},
    "00-0033662": {"draft_number": 263, "draft_year": 2017, "name": "Cooper Rush"},
    "00-0033672": {"draft_number": 263, "draft_year": 2017, "name": "Kyle Sloter"},
    "00-0033799": {"draft_number": 263, "draft_year": 2017, "name": "Tyler Ferguson"},
    "00-0033869": {"draft_number": 2, "draft_year": 2017, "name": "Mitchell Trubisky"},
    "00-0033873": {"draft_number": 10, "draft_year": 2017, "name": "Patrick Mahomes"},
    "00-0033899": {"draft_number": 52, "draft_year": 2017, "name": "DeShone Kizer"},
    "00-0033936": {"draft_number": 104, "draft_year": 2017, "name": "C.J. Beathard"},
    "00-0033949": {"draft_number": 135, "draft_year": 2017, "name": "Joshua Dobbs"},
    "00-0033958": {"draft_number": 171, "draft_year": 2017, "name": "Nathan Peterman"},
    "00-0034126": {"draft_number": 263, "draft_year": 2018, "name": "J.T. Barrett"},
    "00-0034131": {"draft_number": 263, "draft_year": 2018, "name": "Kurt Benkert"},
    "00-0034177": {"draft_number": 263, "draft_year": 2018, "name": "Tim Boyle"},
    "00-0034291": {"draft_number": 263, "draft_year": 2018, "name": "Chase Litton"},
    "00-0034343": {"draft_number": 10, "draft_year": 2018, "name": "Josh Rosen"},
    "00-0034369": {"draft_number": 108, "draft_year": 2018, "name": "Kyle Lauletta"},
    "00-0034401": {"draft_number": 171, "draft_year": 2018, "name": "Mike White"},
    "00-0034412": {"draft_number": 199, "draft_year": 2018, "name": "Luke Falk"},
    "00-0034416": {"draft_number": 203, "draft_year": 2018, "name": "Tanner Lee"},
    "00-0034423": {"draft_number": 219, "draft_year": 2018, "name": "Danny Etling"},
    "00-0034438": {"draft_number": 249, "draft_year": 2018, "name": "Logan Woodside"},
    "00-0034478": {"draft_number": 263, "draft_year": 2018, "name": "Chad Kanoff"},
    "00-0034577": {"draft_number": 263, "draft_year": 2018, "name": "Kyle Allen"},
    "00-0034732": {"draft_number": 220, "draft_year": 2018, "name": "Alex McGough"},
    "00-0034737": {"draft_number": 263, "draft_year": 2018, "name": "Luis Perez"},
    "00-0034771": {"draft_number": 76, "draft_year": 2018, "name": "Mason Rudolph"},
    "00-0034796": {"draft_number": 32, "draft_year": 2018, "name": "Lamar Jackson"},
    "00-0034855": {"draft_number": 1, "draft_year": 2018, "name": "Baker Mayfield"},
    "00-0034857": {"draft_number": 7, "draft_year": 2018, "name": "Josh Allen"},
    "00-0034869": {"draft_number": 3, "draft_year": 2018, "name": "Sam Darnold"},
    "00-0034899": {"draft_number": 263, "draft_year": 2019, "name": "John Wolford"},
    "00-0034955": {"draft_number": 263, "draft_year": 2019, "name": "Brett Rypien"},
    "00-0035010": {"draft_number": 263, "draft_year": 2019, "name": "Eric Dungey"},
    "00-0035040": {"draft_number": 263, "draft_year": 2019, "name": "David Blough"},
    "00-0035077": {"draft_number": 263, "draft_year": 2019, "name": "Manny Wilkins"},
    "00-0035087": {"draft_number": 263, "draft_year": 2019, "name": "Taryn Christion"},
    "00-0035100": {"draft_number": 263, "draft_year": 2019, "name": "Jake Browning"},
    "00-0035146": {"draft_number": 197, "draft_year": 2019, "name": "Trace McSorley"},
    "00-0035163": {"draft_number": 263, "draft_year": 2019, "name": "Kyle Shurmur"},
    "00-0035228": {"draft_number": 1, "draft_year": 2019, "name": "Kyler Murray"},
    "00-0035232": {"draft_number": 15, "draft_year": 2019, "name": "Dwayne Haskins"},
    "00-0035251": {"draft_number": 100, "draft_year": 2019, "name": "Will Grier"},
    "00-0035264": {"draft_number": 133, "draft_year": 2019, "name": "Jarrett Stidham"},
    "00-0035282": {"draft_number": 166, "draft_year": 2019, "name": "Easton Stick"},
    "00-0035283": {"draft_number": 167, "draft_year": 2019, "name": "Clayton Thorson"},
    "00-0035289": {"draft_number": 178, "draft_year": 2019, "name": "Gardner Minshew"},
    "00-0035394": {"draft_number": 263, "draft_year": 2019, "name": "Jake Dolegala"},
    "00-0035471": {"draft_number": 263, "draft_year": 2019, "name": "Nick Fitzgerald"},
    "00-0035483": {"draft_number": 263, "draft_year": 2019, "name": "Drew Anderson"},
    "00-0035577": {"draft_number": 263, "draft_year": 2019, "name": "Devlin Hodges"},
    "00-0035652": {"draft_number": 104, "draft_year": 2019, "name": "Ryan Finley"},
    "00-0035704": {"draft_number": 42, "draft_year": 2019, "name": "Drew Lock"},
    "00-0035710": {"draft_number": 6, "draft_year": 2019, "name": "Daniel Jones"},
    "00-0035735": {"draft_number": 263, "draft_year": 2020, "name": "Jordan Ta'amu"},
    "00-0035752": {"draft_number": 263, "draft_year": 2020, "name": "Chris Streveler"},
    "00-0035812": {"draft_number": 263, "draft_year": 2022, "name": "Case Cookus"},
    "00-0035939": {"draft_number": 263, "draft_year": 2020, "name": "Bryce Perkins"},
    "00-0035988": {"draft_number": 263, "draft_year": 2021, "name": "Anthony Gordon"},
    "00-0035993": {"draft_number": 263, "draft_year": 2020, "name": "Tyler Huntley"},
    "00-0036022": {"draft_number": 263, "draft_year": 2020, "name": "Steven Montez"},
    "00-0036052": {"draft_number": 263, "draft_year": 2020, "name": "Reid Sinnett"},
    "00-0036092": {"draft_number": 263, "draft_year": 2021, "name": "Brian Lewerke"},
    "00-0036197": {"draft_number": 167, "draft_year": 2020, "name": "Jake Fromm"},
    "00-0036212": {"draft_number": 5, "draft_year": 2020, "name": "Tua Tagovailoa"},
    "00-0036226": {"draft_number": 122, "draft_year": 2020, "name": "Jacob Eason"},
    "00-0036264": {"draft_number": 26, "draft_year": 2020, "name": "Jordan Love"},
    "00-0036279": {"draft_number": 244, "draft_year": 2020, "name": "Nate Stanley"},
    "00-0036301": {"draft_number": 125, "draft_year": 2020, "name": "James Morgan"},
    "00-0036312": {"draft_number": 189, "draft_year": 2020, "name": "Jake Luton"},
    "00-0036355": {"draft_number": 6, "draft_year": 2020, "name": "Justin Herbert"},
    "00-0036384": {"draft_number": 231, "draft_year": 2020, "name": "Ben DiNucci"},
    "00-0036389": {"draft_number": 53, "draft_year": 2020, "name": "Jalen Hurts"},
    "00-0036433": {"draft_number": 240, "draft_year": 2020, "name": "Tommy Stevens"},
    "00-0036442": {"draft_number": 1, "draft_year": 2020, "name": "Joe Burrow"},
    "00-0036549": {"draft_number": 263, "draft_year": 2021, "name": "Kenji Bahar"},
    "00-0036679": {"draft_number": 263, "draft_year": 2021, "name": "Shane Buechele"},
    "00-0036879": {"draft_number": 218, "draft_year": 2021, "name": "Sam Ehlinger"},
    "00-0036898": {"draft_number": 67, "draft_year": 2021, "name": "Davis Mills"},
    "00-0036928": {"draft_number": 64, "draft_year": 2021, "name": "Kyle Trask"},
    "00-0036929": {"draft_number": 133, "draft_year": 2021, "name": "Ian Book"},
    "00-0036945": {"draft_number": 11, "draft_year": 2021, "name": "Justin Fields"},
    "00-0036946": {"draft_number": 66, "draft_year": 2021, "name": "Kellen Mond"},
    "00-0036971": {"draft_number": 1, "draft_year": 2021, "name": "Trevor Lawrence"},
    "00-0036972": {"draft_number": 15, "draft_year": 2021, "name": "Mac Jones"},
    "00-0037012": {"draft_number": 3, "draft_year": 2021, "name": "Trey Lance"},
    "00-0037013": {"draft_number": 2, "draft_year": 2021, "name": "Zach Wilson"},
    "00-0037028": {"draft_number": 263, "draft_year": 2021, "name": "Ryan Willis"},
    "00-0037068": {"draft_number": 263, "draft_year": 2022, "name": "E.J. Perry"},
    "00-0037077": {"draft_number": 144, "draft_year": 2022, "name": "Sam Howell"},
    "00-0037139": {"draft_number": 263, "draft_year": 2022, "name": "Carson Strong"},
    "00-0037175": {"draft_number": 263, "draft_year": 2022, "name": "Anthony Brown"},
    "00-0037201": {"draft_number": 263, "draft_year": 2022, "name": "D'Eriq King"},
    "00-0037324": {"draft_number": 241, "draft_year": 2022, "name": "Chris Oladokun"},
    "00-0037327": {"draft_number": 247, "draft_year": 2022, "name": "Skylar Thompson"},
    "00-0037360": {"draft_number": 263, "draft_year": 2022, "name": "Davis Cheek"},
    "00-0037507": {"draft_number": 263, "draft_year": 2022, "name": "Chase Garbers"},
    "00-0037650": {"draft_number": 263, "draft_year": 2022, "name": "Jarrett Guarantano"},
    "00-0037834": {"draft_number": 262, "draft_year": 2022, "name": "Brock Purdy"},
    "00-0038102": {"draft_number": 20, "draft_year": 2022, "name": "Kenny Pickett"},
    "00-0038108": {"draft_number": 137, "draft_year": 2022, "name": "Bailey Zappe"},
    "00-0038122": {"draft_number": 74, "draft_year": 2022, "name": "Desmond Ridder"},
    "00-0038128": {"draft_number": 86, "draft_year": 2022, "name": "Malik Willis"},
    "00-0038132": {"draft_number": 94, "draft_year": 2022, "name": "Matt Corral"},
    "00-0038137": {"draft_number": 263, "draft_year": 2023, "name": "Drew Plitt"},
    "00-0038150": {"draft_number": 263, "draft_year": 2023, "name": "Nathan Rourke"},
    "00-0038385": {"draft_number": 263, "draft_year": 2023, "name": "Dresser Winn"},
    "00-0038391": {"draft_number": 149, "draft_year": 2023, "name": "Sean Clifford"},
    "00-0038400": {"draft_number": 188, "draft_year": 2023, "name": "Tanner McKee"},
    "00-0038416": {"draft_number": 263, "draft_year": 2023, "name": "Tyson Bagent"},
    "00-0038476": {"draft_number": 263, "draft_year": 2023, "name": "Tommy DeVito"},
    "00-0038550": {"draft_number": 68, "draft_year": 2023, "name": "Hendon Hooker"},
    "00-0038579": {"draft_number": 135, "draft_year": 2023, "name": "Aidan O'Connell"},
    "00-0038582": {"draft_number": 139, "draft_year": 2023, "name": "Clayton Tune"},
    "00-0038583": {"draft_number": 140, "draft_year": 2023, "name": "Dorian Thompson-Robinson"},
    "00-0038598": {"draft_number": 164, "draft_year": 2023, "name": "Jaren Hall"},
    "00-0038637": {"draft_number": 239, "draft_year": 2023, "name": "Max Duggan"},
    "00-0038658": {"draft_number": 263, "draft_year": 2024, "name": "Adrian Martinez"},
    "00-0038744": {"draft_number": 263, "draft_year": 2023, "name": "Tanner Morgan"},
    "00-0038749": {"draft_number": 263, "draft_year": 2023, "name": "Holton Ahlers"},
    "00-0038998": {"draft_number": 127, "draft_year": 2023, "name": "Jake Haener"},
    "00-0039107": {"draft_number": 128, "draft_year": 2023, "name": "Stetson Bennett"},
    "00-0039150": {"draft_number": 1, "draft_year": 2023, "name": "Bryce Young"},
    "00-0039152": {"draft_number": 33, "draft_year": 2023, "name": "Will Levis"},
    "00-0039163": {"draft_number": 2, "draft_year": 2023, "name": "C.J. Stroud"},
    "00-0039164": {"draft_number": 4, "draft_year": 2023, "name": "Anthony Richardson"},
    "00-0039238": {"draft_number": 245, "draft_year": 2024, "name": "Michael Pratt"},
    "00-0039331": {"draft_number": 263, "draft_year": 2024, "name": "Emory Jones"},
    "00-0039376": {"draft_number": 150, "draft_year": 2024, "name": "Spencer Rattler"},
    "00-0039398": {"draft_number": 193, "draft_year": 2024, "name": "Joe Milton III"},
    "00-0039464": {"draft_number": 263, "draft_year": 2024, "name": "Austin Reed"},
    "00-0039501": {"draft_number": 263, "draft_year": 2024, "name": "Jack Plummer"},
    "00-0039534": {"draft_number": 263, "draft_year": 2024, "name": "Carter Bradley"},
    "00-0039575": {"draft_number": 263, "draft_year": 2024, "name": "Jason Bean"},
    "00-0039577": {"draft_number": 263, "draft_year": 2024, "name": "Kedon Slovis"},
    "00-0039677": {"draft_number": 263, "draft_year": 2024, "name": "Sam Hartman"},
    "00-0039699": {"draft_number": 263, "draft_year": 2024, "name": "Tanner Mordecai"},
    "00-0039732": {"draft_number": 12, "draft_year": 2024, "name": "Bo Nix"},
    "00-0039797": {"draft_number": 171, "draft_year": 2024, "name": "Jordan Travis"},
    "00-0039801": {"draft_number": 218, "draft_year": 2024, "name": "Devin Leary"},
    "00-0039851": {"draft_number": 3, "draft_year": 2024, "name": "Drake Maye"},
    "00-0039910": {"draft_number": 2, "draft_year": 2024, "name": "Jayden Daniels"},
    "00-0039917": {"draft_number": 8, "draft_year": 2024, "name": "Michael Penix Jr."},
    "00-0039918": {"draft_number": 1, "draft_year": 2024, "name": "Caleb Williams"},
    "00-0039923": {"draft_number": 10, "draft_year": 2024, "name": "J.J. McCarthy"},
    "00-0040014": {"draft_number": 181, "draft_year": 2025, "name": "Kyle McCord"},
    "00-0040203": {"draft_number": 185, "draft_year": 2025, "name": "Will Howard"},
    "00-0040206": {"draft_number": 189, "draft_year": 2025, "name": "Riley Leonard"},
    "00-0040211": {"draft_number": 197, "draft_year": 2025, "name": "Graham Mertz"},
    "00-0040222": {"draft_number": 215, "draft_year": 2025, "name": "Cam Miller"},
    "00-0040234": {"draft_number": 231, "draft_year": 2025, "name": "Quinn Ewers"},
    "00-0040330": {"draft_number": 263, "draft_year": 2025, "name": "Seth Henigan"},
    "00-0040378": {"draft_number": 263, "draft_year": 2025, "name": "Payton Thorne"},
    "00-0040398": {"draft_number": 263, "draft_year": 2025, "name": "Brady Cook"},
    "00-0040415": {"draft_number": 263, "draft_year": 2025, "name": "Connor Bazelak"},
    "00-0040464": {"draft_number": 263, "draft_year": 2025, "name": "DJ Uiagalelei"},
    "00-0040494": {"draft_number": 263, "draft_year": 2025, "name": "Max Brosmer"},
    "00-0040569": {"draft_number": 263, "draft_year": 2025, "name": "Ben Wooldridge"},
    "00-0040589": {"draft_number": 263, "draft_year": 2025, "name": "Kurtis Rourke"},
    "00-0040591": {"draft_number": 263, "draft_year": 2025, "name": "Taylor Elgersma"},
    "00-0040656": {"draft_number": 263, "draft_year": 2025, "name": "Hunter Dekkers"},
    "00-0040668": {"draft_number": 144, "draft_year": 2025, "name": "Shedeur Sanders"},
    "00-0040673": {"draft_number": 92, "draft_year": 2025, "name": "Jalen Milroe"},
    "00-0040676": {"draft_number": 1, "draft_year": 2025, "name": "Cam Ward"},
    "00-0040691": {"draft_number": 25, "draft_year": 2025, "name": "Jaxson Dart"},
    "00-0040704": {"draft_number": 94, "draft_year": 2025, "name": "Dillon Gabriel"},
    "00-0040743": {"draft_number": 40, "draft_year": 2025, "name": "Tyler Shough"},
    "DOU094821": {"draft_number": 263, "draft_year": 2006, "name": "Ben Dougherty"},
    "EVA297676": {"draft_number": 263, "draft_year": 2017, "name": "Jerod Evans"},
    "LOV131275": {"draft_number": 263, "draft_year": 2021, "name": "Josh Love"},
    "WOR632160": {"draft_number": 263, "draft_year": 2015, "name": "Justin Worley"},
}
