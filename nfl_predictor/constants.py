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
NFLREADPY_CACHE_DIR = os.path.join(DATA_PATH, "cache", "nflreadpy")

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


# Standard deviation of score differences
SCORE_DIFF_STD_DEV = 14.21377923  # Standard deviation of score differences for analysis

# URL for SurvivorGrid spreads (future games)
SURVIVOR_GRID_URL = "https://www.survivorgrid.com/"
DEFAULT_TOTAL_LINE = 45.6  # Average total score across 20+ seasons

# Minimum season for data collection (default)
MIN_SEASON = 2003
# NFLverse (nflreadpy) schedule data availability begins in 1999.
NFLREADPY_MIN_SEASON = 1999
# TeamRankings availability (week 2 of 2003 is the earliest reliable week).
TEAMRANKINGS_MIN_SEASON = 2003
TEAMRANKINGS_MIN_WEEK = 2

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
    "stadium_id",
    "stadium",
    "roof",
    "surface",
    "home_coach",
    "away_coach",
    # Kickoff time fields (availability depends on nflreadpy/nflverse version)
    "gametime",
    "game_time",
    "kickoff_time",
    "start_time",
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
    "roof": "stadium_roof",
    "surface": "stadium_surface",
    "stadium": "stadium_name",
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

SEASON_PHASE_COLUMNS = [
    "week_in_season_norm",
    "season_phase_early",
    "season_phase_mid",
    "season_phase_late",
]

STADIUM_REFERENCE_COLUMNS = [
    "stadium_name",
    "stadium_city",
    "stadium_state",
]

STADIUM_FEATURE_COLUMNS = [
    "stadium_surface",
    "stadium_type",
    "stadium_elevation",
]

COACH_REFERENCE_COLUMNS = [
    "away_coach",
    "home_coach",
]

COACH_FEATURE_COLUMNS = [
    "away_coach_games_prior",
    "home_coach_games_prior",
    "away_coach_win_pct_prior",
    "home_coach_win_pct_prior",
    "away_coach_team_games_prior",
    "home_coach_team_games_prior",
    "away_coach_team_win_pct_prior",
    "home_coach_team_win_pct_prior",
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

PRUNED_FEATURE_COLUMNS = [
    "neutral",
    "home_ties",
    "home_games_played",
    "away_division_ties",
    "home_division_ties",
    "away_conference_ties",
    "home_conference_ties",
    "away_next_location_change",
    "home_next_location_change",
    "home_division_rank",
    "away_division_clinched_proxy",
    "home_division_clinched_proxy",
    "away_division_eliminated_proxy",
    "away_conference_clinched_proxy",
    "home_conference_clinched_proxy",
    "away_conference_eliminated_proxy",
    "home_conference_eliminated_proxy",
]

# Metadata columns - includes game info, teams, venue, and conditions
METADATA_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",  # REG, WC, DIV, CON, SB
    "date",
    # Kickoff time fields (when available)
    "gametime",
    "game_datetime",
    "away_abbr",
    "home_abbr",
    "away_qb",
    "home_qb",
    *STADIUM_REFERENCE_COLUMNS,
    *COACH_REFERENCE_COLUMNS,
    "away_rest",
    "home_rest",
    *STADIUM_FEATURE_COLUMNS,
    *COACH_FEATURE_COLUMNS,
    *SEASON_PHASE_COLUMNS,
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

TREND_FEATURE_COLUMNS = [
    "last_5_games_rating_trend",
    "elo_4wk_trend",
    "qb_elo_4wk_trend",
    "qb_value_4wk_trend",
    "scoring_margin_4wk_trend",
    "turnover_margin_4wk_trend",
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

# =============================================================================
# Stadium Metadata (stadium_id -> name, city, state/country, elevation_ft)
# =============================================================================
STADIUMS = {
    # AFC East
    "BOS00": {
        "name": "Gillette Stadium",
        "city": "Foxborough",
        "state": "MA",
        "elevation_ft": 289.0,
    },
    "BUF00": {
        "name": "Highmark Stadium",
        "city": "Orchard Park",
        "state": "NY",
        "elevation_ft": 866.0,
    },
    "BUF01": {
        "name": "Rogers Centre",
        "city": "Toronto",
        "state": "ON",
        "elevation_ft": 251.0,
    },
    "MIA00": {
        "name": "Hard Rock Stadium",
        "city": "Miami Gardens",
        "state": "FL",
        "elevation_ft": 10.0,
    },
    "NYC00": {
        "name": "Giants Stadium",
        "city": "East Rutherford",
        "state": "NJ",
        "elevation_ft": 7.0,
    },
    "NYC01": {
        "name": "MetLife Stadium",
        "city": "East Rutherford",
        "state": "NJ",
        "elevation_ft": 7.0,
    },
    # AFC North
    "BAL00": {
        "name": "M&T Bank Stadium",
        "city": "Baltimore",
        "state": "MD",
        "elevation_ft": 10.0,
    },
    "CIN00": {
        "name": "Paycor Stadium",
        "city": "Cincinnati",
        "state": "OH",
        "elevation_ft": 482.0,
    },
    "CLE00": {
        "name": "Cleveland Browns Stadium",
        "city": "Cleveland",
        "state": "OH",
        "elevation_ft": 580.0,
    },
    "PIT00": {
        "name": "Acrisure Stadium",
        "city": "Pittsburgh",
        "state": "PA",
        "elevation_ft": 712.0,
    },
    # AFC South
    "HOU00": {
        "name": "NRG Stadium",
        "city": "Houston",
        "state": "TX",
        "elevation_ft": 49.0,
    },
    "IND00": {
        "name": "Lucas Oil Stadium",
        "city": "Indianapolis",
        "state": "IN",
        "elevation_ft": 715.0,
    },
    "IND99": {
        "name": "RCA Dome",
        "city": "Indianapolis",
        "state": "IN",
        "elevation_ft": 715.0,
    },
    "JAX00": {
        "name": "EverBank Stadium",
        "city": "Jacksonville",
        "state": "FL",
        "elevation_ft": 16.0,
    },
    "NAS00": {
        "name": "Nissan Stadium",
        "city": "Nashville",
        "state": "TN",
        "elevation_ft": 597.0,
    },
    # AFC West
    "DEN00": {
        "name": "Empower Field at Mile High",
        "city": "Denver",
        "state": "CO",
        "elevation_ft": 5280.0,
    },
    "KAN00": {
        "name": "Arrowhead Stadium",
        "city": "Kansas City",
        "state": "MO",
        "elevation_ft": 889.0,
    },
    "LAX01": {
        "name": "SoFi Stadium",
        "city": "Inglewood",
        "state": "CA",
        "elevation_ft": 25.0,
    },
    "LAX97": {
        "name": "StubHub Center (Chargers temp)",
        "city": "Carson",
        "state": "CA",
        "elevation_ft": 39.0,
    },
    "LAX99": {
        "name": "Los Angeles Memorial Coliseum (Rams temp)",
        "city": "Los Angeles",
        "state": "CA",
        "elevation_ft": 305.0,
    },
    "OAK00": {
        "name": "Oakland Coliseum",
        "city": "Oakland",
        "state": "CA",
        "elevation_ft": 43.0,
    },
    "SDG00": {
        "name": "Qualcomm Stadium",
        "city": "San Diego",
        "state": "CA",
        "elevation_ft": 52.0,
    },
    "VEG00": {
        "name": "Allegiant Stadium",
        "city": "Las Vegas",
        "state": "NV",
        "elevation_ft": 2190.0,
    },
    # NFC East
    "DAL00": {
        "name": "AT&T Stadium",
        "city": "Arlington",
        "state": "TX",
        "elevation_ft": 604.0,
    },
    "DAL99": {
        "name": "Texas Stadium",
        "city": "Irving",
        "state": "TX",
        "elevation_ft": 482.0,
    },
    "PHI00": {
        "name": "Lincoln Financial Field",
        "city": "Philadelphia",
        "state": "PA",
        "elevation_ft": 39.0,
    },
    "WAS00": {
        "name": "Northwest Stadium (FedEx Field)",
        "city": "Landover",
        "state": "MD",
        "elevation_ft": 207.0,
    },
    # NFC North
    "CHI98": {
        "name": "Soldier Field",
        "city": "Chicago",
        "state": "IL",
        "elevation_ft": 590.0,
    },
    "DET00": {
        "name": "Ford Field",
        "city": "Detroit",
        "state": "MI",
        "elevation_ft": 600.0,
    },
    "GNB00": {
        "name": "Lambeau Field",
        "city": "Green Bay",
        "state": "WI",
        "elevation_ft": 640.0,
    },
    "MIN00": {
        "name": "Hubert H. Humphrey Metrodome",
        "city": "Minneapolis",
        "state": "MN",
        "elevation_ft": 849.0,
    },
    "MIN01": {
        "name": "U.S. Bank Stadium",
        "city": "Minneapolis",
        "state": "MN",
        "elevation_ft": 830.0,
    },
    "MIN98": {
        "name": "TCF Bank Stadium (temp)",
        "city": "Minneapolis",
        "state": "MN",
        "elevation_ft": 830.0,
    },
    # NFC South
    "ATL00": {
        "name": "Georgia Dome",
        "city": "Atlanta",
        "state": "GA",
        "elevation_ft": 1050.0,
    },
    "ATL97": {
        "name": "Mercedes-Benz Stadium",
        "city": "Atlanta",
        "state": "GA",
        "elevation_ft": 997.0,
    },
    "BRG00": {
        "name": "Tiger Stadium (LSU)",
        "city": "Baton Rouge",
        "state": "LA",
        "elevation_ft": 82.0,
    },
    "CAR00": {
        "name": "Bank of America Stadium",
        "city": "Charlotte",
        "state": "NC",
        "elevation_ft": 751.0,
    },
    "NOR00": {
        "name": "Caesars Superdome",
        "city": "New Orleans",
        "state": "LA",
        "elevation_ft": 3.0,
    },
    "SAN00": {
        "name": "Alamodome",
        "city": "San Antonio",
        "state": "TX",
        "elevation_ft": 722.0,
    },
    "TAM00": {
        "name": "Raymond James Stadium",
        "city": "Tampa",
        "state": "FL",
        "elevation_ft": 52.0,
    },
    # NFC West
    "PHO00": {
        "name": "State Farm Stadium",
        "city": "Glendale",
        "state": "AZ",
        "elevation_ft": 1070.0,
    },
    "PHO99": {
        "name": "Sun Devil Stadium",
        "city": "Tempe",
        "state": "AZ",
        "elevation_ft": 1181.0,
    },
    "SEA00": {
        "name": "Lumen Field",
        "city": "Seattle",
        "state": "WA",
        "elevation_ft": 16.0,
    },
    "SFO00": {
        "name": "Candlestick Park",
        "city": "San Francisco",
        "state": "CA",
        "elevation_ft": 26.0,
    },
    "SFO01": {
        "name": "Levi's Stadium",
        "city": "Santa Clara",
        "state": "CA",
        "elevation_ft": 176.0,
    },
    "STL00": {
        "name": "The Dome at America's Center",
        "city": "St. Louis",
        "state": "MO",
        "elevation_ft": 466.0,
    },
    # International Venues
    "FRA00": {
        "name": "Deutsche Bank Park",
        "city": "Frankfurt",
        "state": "DE",
        "elevation_ft": 367.0,
    },
    "GER00": {
        "name": "Allianz Arena",
        "city": "Munich",
        "state": "DE",
        "elevation_ft": 1617.0,
    },
    "LON00": {
        "name": "Wembley Stadium",
        "city": "London",
        "state": "UK",
        "elevation_ft": 82.0,
    },
    "LON01": {
        "name": "Twickenham Stadium",
        "city": "London",
        "state": "UK",
        "elevation_ft": 82.0,
    },
    "LON02": {
        "name": "Tottenham Hotspur Stadium",
        "city": "London",
        "state": "UK",
        "elevation_ft": 82.0,
    },
    "MEX00": {
        "name": "Estadio Azteca",
        "city": "Mexico City",
        "state": "MX",
        "elevation_ft": 7380.0,
    },
    "SAO00": {
        "name": "Arena Corinthians (Neo Química Arena)",
        "city": "São Paulo",
        "state": "BR",
        "elevation_ft": 2490.0,
    },
}
