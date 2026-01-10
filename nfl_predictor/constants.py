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

# Regression factor for week-1 stats toward league mean
WEEK1_REGRESSION_FACTOR = 1 / 3

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
    "two-point-conversion-pct": "two_point_conversion_pct",
    "opponent-two-point-conversion-pct": "opponent_two_point_conversion_pct",
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
    for team_abbr, team_meta in TEAM_MAPPING.items():
        for alias in team_meta["aliases"]:
            mapping[alias] = team_abbr
            mapping[alias.upper()] = team_abbr
            mapping[alias.lower()] = team_abbr
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
TEAMS_TO_ABBR = {}
for canonical_abbr, team_info in TEAM_MAPPING.items():
    TEAMS_TO_ABBR[team_info["city"]] = canonical_abbr
    TEAMS_TO_ABBR[team_info["name"]] = canonical_abbr


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
    "spread_line": "away_spread",  # nflreadpy spread_line is from away team perspective
}

# ============================================================================
# Data Collection Column Definitions
# ============================================================================

# Team division/conference mapping (modern alignment; applies to all seasons in this repo).
# Used for record splits (division/conference), divisional matchup indicator,
# and motivation proxies.
TEAM_TO_DIVISION: dict[str, str] = {
    # AFC East
    "BUF": "AFC East",
    "MIA": "AFC East",
    "NE": "AFC East",
    "NYJ": "AFC East",
    # AFC North
    "BAL": "AFC North",
    "CIN": "AFC North",
    "CLE": "AFC North",
    "PIT": "AFC North",
    # AFC South
    "HOU": "AFC South",
    "IND": "AFC South",
    "JAX": "AFC South",
    "TEN": "AFC South",
    # AFC West
    "DEN": "AFC West",
    "KC": "AFC West",
    "LAC": "AFC West",
    "LV": "AFC West",
    # NFC East
    "DAL": "NFC East",
    "NYG": "NFC East",
    "PHI": "NFC East",
    "WSH": "NFC East",
    # NFC North
    "CHI": "NFC North",
    "DET": "NFC North",
    "GB": "NFC North",
    "MIN": "NFC North",
    # NFC South
    "ATL": "NFC South",
    "CAR": "NFC South",
    "NO": "NFC South",
    "TB": "NFC South",
    # NFC West
    "ARI": "NFC West",
    "LAR": "NFC West",
    "SEA": "NFC West",
    "SF": "NFC West",
}

TEAM_TO_CONFERENCE: dict[str, str] = {
    team: "AFC" if div.startswith("AFC") else "NFC" for team, div in TEAM_TO_DIVISION.items()
}

RECORD_FEATURE_COLUMNS = [
    # Overall
    "away_wins",
    "away_losses",
    "away_ties",
    "away_games_played",
    "away_win_pct",
    "home_wins",
    "home_losses",
    "home_ties",
    "home_games_played",
    "home_win_pct",
    # Division
    "away_division_wins",
    "away_division_losses",
    "away_division_ties",
    "home_division_wins",
    "home_division_losses",
    "home_division_ties",
    # Conference
    "away_conference_wins",
    "away_conference_losses",
    "away_conference_ties",
    "home_conference_wins",
    "home_conference_losses",
    "home_conference_ties",
]

DIVISIONAL_FEATURE_COLUMNS = [
    "is_divisional_matchup",
]

LOOKAHEAD_FEATURE_COLUMNS = [
    "away_next_opponent_abbr",
    "away_next_is_home",
    "away_days_to_next_game",
    "away_next_location_change",
    "away_next_is_divisional_matchup",
    "away_next_opponent_win_pct",
    "home_next_opponent_abbr",
    "home_next_is_home",
    "home_days_to_next_game",
    "home_next_location_change",
    "home_next_is_divisional_matchup",
    "home_next_opponent_win_pct",
]

MOTIVATION_FEATURE_COLUMNS = [
    "away_division_rank",
    "home_division_rank",
    "away_conference_rank",
    "home_conference_rank",
    "away_division_games_behind",
    "home_division_games_behind",
    "away_conference_games_behind_seed7",
    "home_conference_games_behind_seed7",
    "away_division_clinched_proxy",
    "home_division_clinched_proxy",
    "away_division_eliminated_proxy",
    "home_division_eliminated_proxy",
    "away_conference_clinched_proxy",
    "home_conference_clinched_proxy",
    "away_conference_eliminated_proxy",
    "home_conference_eliminated_proxy",
]

# Metadata columns (24 total) - includes game info, teams, venue, and conditions
METADATA_COLUMNS = [
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
    # Feature column groups (joined into game rows)
    *DIVISIONAL_FEATURE_COLUMNS,
    *RECORD_FEATURE_COLUMNS,
    *LOOKAHEAD_FEATURE_COLUMNS,
    *MOTIVATION_FEATURE_COLUMNS,
]

# ELO rating columns (per team) - these get prefixed with away_/home_
ELO_COLUMNS = [
    "elo_pre",
    "qb_value_pre",
    "qb_elo_pre",
]

# TeamRankings rating columns (per team) - these get prefixed with away_/home_
TR_RATINGS = [
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
TR_STATS = [
    "third_down_pct",
    "opponent_third_down_pct",
    "fourth_down_pct",
    "opponent_fourth_down_pct",
    "red_zone_td_pct",
    "opponent_red_zone_td_pct",
    "two_point_conversion_pct",
    "opponent_two_point_conversion_pct",
]

# Stats to EXCLUDE from opponent stat generation (they create duplicates)
# These stats are duplicates when viewed from opponent's perspective:
# - scoring_margin: opponent_scoring_margin = -scoring_margin
# - points_scored: opponent_points_scored = points_allowed
# - points_allowed: opponent_points_allowed = points_scored
# - def_interceptions: opponent_def_interceptions = interceptions_thrown
# - interceptions_thrown: opponent_interceptions_thrown = def_interceptions
# - turnover_margin: opponent_turnover_margin = -turnover_margin
# - Also exclude computed ratio stats that use the above
EXCLUDE_FROM_OPPONENT_STATS = [
    "scoring_margin",
    "points_scored",
    "points_allowed",
    "def_interceptions",
    "interceptions_thrown",
    "turnover_margin",
    # Derived ratio metrics that would be duplicates
    "yards_per_point",
    "yards_per_point_margin",
    "points_per_play",
    "points_per_play_margin",
    "penalty_yards_per_penalty",
]

# nflreadpy stats to use (per team) - these get prefixed with away_/home_
NFLREADPY_STATS = [
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
LINES_COLUMNS = [
    "total_line",
    "away_spread",
    "home_spread",
    "away_moneyline",
    "home_moneyline",
]

# Result columns (2 total)
RESULT_COLUMNS = [
    "away_score",
    "home_score",
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
