"""Define shared constants for the NFL predictor project.

This module centralizes paths, season rules, team mappings, and schema metadata so ETL,
modeling, and reporting code stay aligned.
"""

from pathlib import Path

# Project directory configurations
ROOT_DIR = Path(__file__).parents[2]
DATA_PATH = ROOT_DIR / "data"
# Run directories and walk-forward checkpoints. Code reads it when it runs, never into a copy at
# import, so the test suite can point it at a temporary directory (tests/data_isolation.py).
MODELS_PATH = ROOT_DIR / "models"
NFLREADPY_CACHE_DIR = DATA_PATH / "cache" / "nflreadpy"
# Finished seasons' ETL builds for `nfl-predictor data --incremental`, kept under the output
# directory's `cache/`. Safe to delete: a missing entry is rebuilt.
ETL_SEASON_CACHE_DIRNAME = "etl_seasons"

# NFL season configurations
SEASON_END_MONTH = 2  # NFL season typically ends in February
WEEKS_BEFORE_2021 = 17  # Number of weeks in NFL seasons before 2021
WEEKS_FROM_2021_ONWARDS = 18  # Number of weeks in NFL seasons from 2021 onwards
FIRST_18_WEEK_SEASON = 2021  # First season with an 18-week regular season


# A win probability at even odds; at or above it, a side is the favorite and the pick.
EVEN_ODDS_PROBABILITY = 0.5


def get_regular_season_weeks(season: int) -> int:
    """Return the number of regular season weeks for a given season.

    Before 2021, the NFL had 17-week regular seasons.
    From 2021 onwards, the NFL has 18-week regular seasons.

    Args:
        season: The NFL season year

    Returns:
        Number of regular season weeks (17 before 2021, 18 from 2021 onwards)

    """
    return WEEKS_FROM_2021_ONWARDS if season >= FIRST_18_WEEK_SEASON else WEEKS_BEFORE_2021


# Standard deviation of score differences
SCORE_DIFF_STD_DEV = 14.21377923  # Standard deviation of score differences for analysis

# The spread (sigma) of the probability floor, Phi(predicted margin / sigma). For a game in
# (season, week) it is the root-mean-square out-of-fold margin error of every prediction strictly
# before that week (`src/nfl_predictor/ml/floor_sigma.py`). Until that pool spans this many earlier
# seasons (any weeks of them; the season's own earlier weeks do not count) the floor uses
# SCORE_DIFF_STD_DEV instead, and records that it fell back.
FLOOR_SIGMA_MIN_POOL_SEASONS = 3
# The walk-forward runs whose out-of-fold errors form the production pool (the weekly run and
# `nfl-predictor train`), relative to the repository root; their fold checkpoints are only read.
# Both seeds of the reviewed walk-forward of production's settings: market anchored, with
# PRODUCTION_DISABLED_FEATURE_GROUPS left out.
FLOOR_SIGMA_REFERENCE_RUNS = (
    "models/step6/ratings_anchored_seed42",
    "models/step6/ratings_anchored_seed7",
)

# The feature groups (FEATURE_GROUP_COLUMN_MARKERS) production leaves out: the strength snapshot
# and both presets of this repository's own Elo, so production trains on the ported team ratings
# (`ratings`) and the `qb_elos.csv` Elo (`elo`). The default of the weekly run,
# `nfl-predictor train` and `nfl-predictor backtest`, so a backtest measures production's features.
PRODUCTION_DISABLED_FEATURE_GROUPS = ("strength", "own_elo", "own_elo_wt")

# URL for SurvivorGrid spreads (future games)
SURVIVOR_GRID_URL = "https://www.survivorgrid.com/"
DEFAULT_TOTAL_LINE = 45.6  # Average total score across 20+ seasons

# Where the ETL's market lines come from (`nfl-predictor data --line-source`). "stored" keeps the
# nflverse schedule lines, a late, probably closing, snapshot. "pick_time" uses the line known when
# picks are made: nfelo's opening line for completed games and its latest line for upcoming ones,
# with the stored line where nfelo has no real opener.
LINE_SOURCE_STORED = "stored"
LINE_SOURCE_PICK_TIME = "pick_time"
LINE_SOURCES = (LINE_SOURCE_STORED, LINE_SOURCE_PICK_TIME)
# The line source of every build that does not name one: `nfl-predictor data`, the weekly run's
# data refresh and the web ETL and weekly jobs.
DEFAULT_LINE_SOURCE = LINE_SOURCE_PICK_TIME
# nfelo's market-lines file (`greerreNFL/nfelomarket_data`), rewritten several times a day. The
# latest good download is cached under NFELO_CACHE_DIR, and every copy a run used is kept there
# under `snapshots/`, named by its SHA-256.
NFELO_LINES_URL = (
    "https://raw.githubusercontent.com/greerreNFL/nfelomarket_data/main/Data/lines.csv"
)
NFELO_CACHE_DIR = DATA_PATH / "cache" / "nfelo"
# nfelo's openers are real from 2007 on; before 2007 every opener equals the last line, and in
# 2022 many do (59%), so 2022's openers are not trusted either.
NFELO_FIRST_REAL_OPENER_SEASON = 2007
NFELO_UNRELIABLE_OPENER_SEASONS = frozenset({2022})
# The fitted spread-to-moneyline map for a season is fitted on every earlier season's games with
# a real spread and both real moneylines; with fewer such games than this, that season keeps the
# fixed conversion (`game_utils.spread_to_moneyline`).
MONEYLINE_MAP_MIN_GAMES = 200
# Written beside the datasets by a pick-time build: the nfelo snapshot, match rates, fallback
# rows and the moneyline maps the run used.
MARKET_LINES_METADATA_NAME = "market_lines_metadata"
# Base name of the own Elo's build record (both presets and the win-total file it read), JSON.
OWN_ELO_METADATA_NAME = "own_elo_metadata"
# nfelosrs's preseason win-total ratings (`greerreNFL/nfelosrs`, 2003 on): each team's rating
# implied by the sportsbooks' season win totals and its schedule, on the Elo scale. Market
# information: only the own Elo's win-total preset reads it, cached and snapshotted like the
# lines under NFELO_CACHE_DIR / "win_totals".
NFELOSRS_WIN_TOTALS_URL = (
    "https://raw.githubusercontent.com/greerreNFL/nfelosrs/main/wt_ratings.csv"
)

# Minimum season for data collection (default)
MIN_SEASON = 1999
# NFLverse (nflreadpy) schedule data availability begins in 1999.
NFLREADPY_MIN_SEASON = 1999
# TeamRankings availability (week 2 of 2003 is the earliest reliable week).
TEAMRANKINGS_MIN_SEASON = 2003
TEAMRANKINGS_MIN_WEEK = 2

# Regression factor for week-1 stats toward league mean
WEEK1_REGRESSION_FACTOR = 1 / 3

# Games at which an early-season in-season sample and the regressed previous season are
# weighted equally. The default of both the adjusted-strength snapshot's blend and the
# season-to-date stat blend; `nfl-predictor data` sets each on its own
# (`--strength-prior-blend-games`, `--stat-prior-blend-games`).
#
# The in-season weight is `games / (games + PRIOR_BLEND_GAMES)`, so with K = 4 a team
# reaches parity after four games (about week 5), three-quarters in-season after twelve,
# and the prior is never fully discarded. Four games is roughly the point at which a
# per-snap EPA sample stops being dominated by one blowout, and it matches the
# quarter-season scale the rest of this repository already uses for early-season
# fallbacks.
PRIOR_BLEND_GAMES = 4.0

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
# nflverse player sources: weekly player stats, snap counts, the players file, rosters, ESPN QBR
# ============================================================================

# The first season each source publishes. Its loader returns no rows for an earlier season
# without asking nflreadpy, which refuses some of them with a ValueError. nflverse's 2012 snap
# counts file holds no rows, so snap counts start in 2013 in practice.
NFLREADPY_PLAYER_STATS_MIN_SEASON = 1999
NFLREADPY_SNAP_COUNTS_MIN_SEASON = 2012
NFLREADPY_ROSTERS_MIN_SEASON = 1920
NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON = 2002

# Columns kept from `nflreadpy.load_player_stats(summary_level="week")`, one row per player and
# game: identity, the official passing and rushing totals the quarterback rating reads, and the
# defensive stats only the box score charts.
NFLREADPY_PLAYER_STATS_COLUMNS = [
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "sack_yards_lost",
    "passing_epa",
    "passing_cpoe",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "rushing_fumbles",
    "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "def_tackles_for_loss",
    "def_fumbles_forced",
    "def_sacks",
    "def_qb_hits",
    "def_interceptions",
    "def_pass_defended",
    "def_safeties",
]

# Weekly player stats credit no one with these stats in these seasons (inclusive) and publish 0
# instead of a missing value; the loader leaves them null there. nflverse charts one tackle for
# loss in all of 2006.
NFLREADPY_PLAYER_STAT_GAPS: dict[str, tuple[int, int]] = {
    "def_tackles_for_loss": (2003, 2011),
    "def_qb_hits": (2003, 2005),
}

# Columns kept from `nflreadpy.load_snap_counts`, keyed by the Pro Football Reference player id.
NFLREADPY_SNAP_COUNT_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "week",
    "player",
    "pfr_player_id",
    "position",
    "team",
    "opponent",
    "offense_snaps",
    "offense_pct",
]

# Columns kept from `nflreadpy.load_players`, one row per player with his latest position and the
# ids other sources key on (Pro Football Reference for snap counts, ESPN for QBR).
NFLREADPY_PLAYERS_COLUMNS = [
    "gsis_id",
    "display_name",
    "position",
    "pfr_id",
    "espn_id",
]

# Columns kept from `nflreadpy.load_rosters` and `nflreadpy.load_rosters_weekly`, which share a
# schema. Season rosters before 2002 carry no week or game type.
NFLREADPY_ROSTER_COLUMNS = [
    "season",
    "week",
    "game_type",
    "team",
    "position",
    "full_name",
    "gsis_id",
    "pfr_id",
]

# League (GSIS) team codes the 2002-2015 season and weekly rosters use, with nflverse's code for
# each; the loaders then normalize nflverse's codes through ALIAS_TO_CANONICAL as usual.
NFLVERSE_ROSTER_TEAM_CODES = {
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "SL": "STL",
}

# The position a source lists a quarterback at. A player the players file or any regular-season
# week of the season's weekly roster lists here is a quarterback that season, whatever else he
# plays (`utils/polars/player_identity.py`).
QB_POSITION = "QB"

# Games with at least this many player team values that all name one team are credited wholly to
# one team (`utils/polars/player_team_repair.py`): every real game credits plays to both teams,
# but a short fixture or a partly published game may not.
ONE_TEAM_GAME_MIN_VALUES = 20

# Play-by-play team columns that name a player's team, with the player id each one belongs to.
PBP_PLAYER_TEAM_ID_COLUMNS = {
    "td_team": "td_player_id",
    "penalty_team": "penalty_player_id",
    "fumbled_1_team": "fumbled_1_player_id",
    "fumbled_2_team": "fumbled_2_player_id",
    "fumble_recovery_1_team": "fumble_recovery_1_player_id",
    "fumble_recovery_2_team": "fumble_recovery_2_player_id",
    "forced_fumble_player_1_team": "forced_fumble_player_1_player_id",
    "forced_fumble_player_2_team": "forced_fumble_player_2_player_id",
    "solo_tackle_1_team": "solo_tackle_1_player_id",
    "solo_tackle_2_team": "solo_tackle_2_player_id",
    "assist_tackle_1_team": "assist_tackle_1_player_id",
    "assist_tackle_2_team": "assist_tackle_2_player_id",
    "assist_tackle_3_team": "assist_tackle_3_player_id",
    "assist_tackle_4_team": "assist_tackle_4_player_id",
    "tackle_with_assist_1_team": "tackle_with_assist_1_player_id",
    "tackle_with_assist_2_team": "tackle_with_assist_2_player_id",
}

# Play-by-play team columns with no player behind them, unknown in a game credited to one team.
PBP_TEAM_ONLY_COLUMNS = ("return_team", "timeout_team")

# ESPN QBR (utils/polars/espn_qbr.py), a validation reference only: never a game-dataset column
# or a model input. nflverse publishes it from 2006 as release assets, every season in one file
# per level.
ESPN_QBR_MIN_SEASON = 2006
ESPN_QBR_RELEASE_URLS = {
    "season": (
        "https://github.com/nflverse/nflverse-data/releases/download/espn_data/"
        "qbr_season_level.parquet"
    ),
    "week": (
        "https://github.com/nflverse/nflverse-data/releases/download/espn_data/"
        "qbr_week_level.parquet"
    ),
}

# Columns kept from each QBR asset. `player_id` is ESPN's id (`espn_id` in the players file).
ESPN_QBR_SEASON_COLUMNS = [
    "season",
    "season_type",
    "game_week",
    "team_abb",
    "player_id",
    "name_display",
    "qb_plays",
    "qbr_total",
    "qbr_raw",
    "qualified",
]
ESPN_QBR_WEEK_COLUMNS = [
    "season",
    "season_type",
    "game_id",
    "game_week",
    "team_abb",
    "opp_abb",
    "player_id",
    "name_display",
    "qb_plays",
    "qbr_total",
    "qbr_raw",
    "qualified",
]

# ============================================================================
# Data Collection Column Definitions
# ============================================================================

# Team division/conference mapping. `TEAM_TO_DIVISION` is the modern alignment; callers that need
# historical context should use `division_map_for_season` / `conference_map_for_season`.
NFL_REALIGNMENT_SEASON = 2002

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

PRE_2002_TEAM_TO_DIVISION: dict[str, str] = {
    # AFC East
    "BUF": "AFC East",
    "IND": "AFC East",
    "MIA": "AFC East",
    "NE": "AFC East",
    "NYJ": "AFC East",
    # AFC Central
    "BAL": "AFC Central",
    "CIN": "AFC Central",
    "CLE": "AFC Central",
    "JAX": "AFC Central",
    "PIT": "AFC Central",
    "TEN": "AFC Central",
    # AFC West
    "DEN": "AFC West",
    "KC": "AFC West",
    "LAC": "AFC West",
    "LV": "AFC West",
    "SEA": "AFC West",
    # NFC East
    "ARI": "NFC East",
    "DAL": "NFC East",
    "NYG": "NFC East",
    "PHI": "NFC East",
    "WSH": "NFC East",
    # NFC Central
    "CHI": "NFC Central",
    "DET": "NFC Central",
    "GB": "NFC Central",
    "MIN": "NFC Central",
    "TB": "NFC Central",
    # NFC West
    "ATL": "NFC West",
    "CAR": "NFC West",
    "NO": "NFC West",
    "LAR": "NFC West",
    "SF": "NFC West",
}

PRE_2002_TEAM_TO_CONFERENCE: dict[str, str] = {
    team: "AFC" if div.startswith("AFC") else "NFC"
    for team, div in PRE_2002_TEAM_TO_DIVISION.items()
}


def division_map_for_season(season: int | None) -> dict[str, str]:
    """Return the NFL division alignment that applied in a season."""
    if season is not None and int(season) < NFL_REALIGNMENT_SEASON:
        return PRE_2002_TEAM_TO_DIVISION
    return TEAM_TO_DIVISION


def conference_map_for_season(season: int | None) -> dict[str, str]:
    """Return the NFL conference alignment that applied in a season."""
    if season is not None and int(season) < NFL_REALIGNMENT_SEASON:
        return PRE_2002_TEAM_TO_CONFERENCE
    return TEAM_TO_CONFERENCE


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
    # Both sides of the season-to-date game count: `away_/home_strength_games_played` already
    # publish it per side, with a diff, and agree with these in every row but the prior-season
    # fallbacks.
    "away_games_played",
    "home_games_played",
    # The two teams in a game have almost always played the same number of games, so the diff
    # is a constant zero outside bye weeks. The ETL does not publish it (STATS_WITHOUT_DIFF);
    # it stays pruned so a dataset that still carries it trains the same way.
    "strength_games_played_diff",
    # Equal to the schedule's `division` flag from the 2002 realignment on, and derived from
    # the current division map before it, so the schedule flag is the one to keep.
    "is_divisional_matchup",
    # One-hot repeats of `week_in_season_norm`.
    "season_phase_early",
    "season_phase_mid",
    "season_phase_late",
    # Near-constant: a handful of ties per season.
    "away_ties",
    # Next-game context flags that trees almost never split on; the location-change flags and
    # the opponent's win percentage carry what signal the lookahead family has.
    "away_next_is_home",
    "home_next_is_home",
    "away_next_is_divisional_matchup",
    "home_next_is_divisional_matchup",
    "away_days_to_next_game",
    "home_days_to_next_game",
    # Mirrors of `times_sacked` and `def_sacks` seen from the other sideline: the same events,
    # counted for the opponent. The interception and turnover mirrors are already excluded at
    # ETL time (`EXCLUDE_FROM_OPPONENT_STATS`); these are excluded at training time.
    "away_opponent_def_sacks",
    "home_opponent_def_sacks",
    "opponent_def_sacks_diff",
    "away_opponent_times_sacked",
    "home_opponent_times_sacked",
    "opponent_times_sacked_diff",
    "away_division_ties",
    "home_division_ties",
    "away_conference_ties",
    "home_conference_ties",
    "away_next_location_change",
    "home_next_location_change",
    "away_division_rank",
    "home_division_rank",
    "away_division_clinched_proxy",
    "home_division_clinched_proxy",
    "away_division_eliminated_proxy",
    "home_division_eliminated_proxy",
    "away_conference_clinched_proxy",
    "home_conference_clinched_proxy",
    "away_conference_eliminated_proxy",
    "home_conference_eliminated_proxy",
    # The next opponent's team identity, one-hot encoded per side (32 columns each): the trees
    # split on it rarely, and `*_next_opponent_win_pct` carries the next opponent's strength.
    # The ETL still publishes the pair.
    "away_next_opponent_abbr",
    "home_next_opponent_abbr",
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

# Four-week trends of the ELO_COLUMNS values (per team, and per quarterback for the QB ones).
ELO_TREND_COLUMNS = [
    "elo_4wk_trend",
    "qb_elo_4wk_trend",
    "qb_value_4wk_trend",
]

# This repository's own Elo (FiveThirtyEight's method, computed from game results by
# `utils/polars/elo_ratings.py`), separate from the `data/qb_elos.csv` columns above so one
# dataset can carry both. It is published as two families, one per preset. The market-free one:
# per team, prefixed with away_/home_ and with an away-minus-home `_diff`, the team's pre-game
# rating, its quarterback adjustment included.
OWN_ELO_TEAM_STATS = ["own_elo"]
# Per game: the home team's pre-game win probability from the same ratings, home field and the
# other game-level adjustments included.
OWN_ELO_GAME_COLUMNS = ["own_elo_home_win_prob"]
OWN_ELO_COLUMNS = [
    *(f"{side}_{stat}" for stat in OWN_ELO_TEAM_STATS for side in ("away", "home")),
    *(f"{stat}_diff" for stat in OWN_ELO_TEAM_STATS),
    *OWN_ELO_GAME_COLUMNS,
]
# The same columns from the preset with the win-total pull (market information). The stat is
# `own_wt_elo`, not `own_elo_wt`: group markers match substrings, and `away_own_elo` is a
# substring of `away_own_elo_wt`, so the market-free group could not leave those columns out.
OWN_WT_ELO_TEAM_STATS = ["own_wt_elo"]
OWN_WT_ELO_GAME_COLUMNS = ["own_wt_elo_home_win_prob"]
OWN_WT_ELO_COLUMNS = [
    *(f"{side}_{stat}" for stat in OWN_WT_ELO_TEAM_STATS for side in ("away", "home")),
    *(f"{stat}_diff" for stat in OWN_WT_ELO_TEAM_STATS),
    *OWN_WT_ELO_GAME_COLUMNS,
]

TREND_FEATURE_COLUMNS = [
    "last_5_games_rating_trend",
    *ELO_TREND_COLUMNS,
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

# Play-by-play columns selected from nflreadpy.load_pbp before caching.
# Every column is existence-guarded at load time: older seasons do not publish all of
# them, and a missing column becomes a null-filled default rather than an error.
# The list also carries every column the ratings ported from nfl-sos-ratings read (its team,
# quarterback, special-teams and win-probability inputs, its loader's team-code repairs and the
# botched-snap rule), so they are cached before anything here reads them.
PBP_COLUMNS = [
    # Identity and join keys
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "posteam_type",
    # Game state before the snap, and the score after the play
    "qtr",
    "half_seconds_remaining",
    "score_differential",
    "score_differential_post",
    "posteam_score",
    "posteam_score_post",
    "wp",
    # Play classification
    "play_type",
    "qb_dropback",
    "qb_kneel",
    "qb_spike",
    "qb_scramble",
    "aborted_play",
    "rush",
    "pass",
    "rush_attempt",
    "pass_attempt",
    "shotgun",
    "no_huddle",
    "pass_length",
    "down",
    "ydstogo",
    "yardline_100",
    "goal_to_go",
    "yrdln",
    "side_of_field",
    "end_yard_line",
    # Value columns
    "epa",
    "qb_epa",
    "wpa",
    "success",
    "yards_gained",
    "passing_yards",
    "rushing_yards",
    "return_yards",
    "air_yards",
    "air_epa",
    "yards_after_catch",
    "yac_epa",
    "xyac_mean_yardage",
    "cpoe",
    "xpass",
    "pass_oe",
    # Situational outcomes
    "third_down_converted",
    "third_down_failed",
    "fourth_down_converted",
    "fourth_down_failed",
    "two_point_attempt",
    "two_point_conv_result",
    "defensive_two_point_conv",
    "touchdown",
    "td_team",
    "sack",
    "qb_hit",
    "interception",
    "fumble",
    "fumble_forced",
    "fumble_lost",
    "complete_pass",
    "incomplete_pass",
    "tackled_for_loss",
    "pass_touchdown",
    "rush_touchdown",
    "first_down",
    "first_down_pass",
    "first_down_rush",
    "first_down_penalty",
    "penalty",
    "penalty_team",
    "penalty_yards",
    "penalty_type",
    "timeout_team",
    # Special teams
    "special",
    "special_teams_play",
    "punt_attempt",
    "field_goal_attempt",
    "kickoff_attempt",
    "return_team",
    # Drives
    "fixed_drive",
    "fixed_drive_result",
    "drive_start_yard_line",
    "drive_end_yard_line",
    "drive_play_count",
    "drive_first_downs",
    "drive_inside20",
    "drive_time_of_possession",
    "drive_yards_penalized",
    "ydsnet",
    # Series (a set of downs that ends in a first down, a score or a change of possession)
    "series",
    "series_success",
    "series_result",
    # Passer identity (bridge to Elo QB identities)
    "passer_player_id",
    "passer_player_name",
    # The other players nflverse credits on a play, by id, with the team it credits each to
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "td_player_id",
    "penalty_player_id",
    "pass_defense_1_player_id",
    "fumbled_1_team",
    "fumbled_1_player_id",
    "fumbled_2_team",
    "fumbled_2_player_id",
    "fumble_recovery_1_team",
    "fumble_recovery_1_player_id",
    "fumble_recovery_2_team",
    "fumble_recovery_2_player_id",
    "forced_fumble_player_1_team",
    "forced_fumble_player_1_player_id",
    "forced_fumble_player_2_team",
    "forced_fumble_player_2_player_id",
    "solo_tackle_1_team",
    "solo_tackle_1_player_id",
    "solo_tackle_2_team",
    "solo_tackle_2_player_id",
    "assist_tackle_1_team",
    "assist_tackle_1_player_id",
    "assist_tackle_2_team",
    "assist_tackle_2_player_id",
    "assist_tackle_3_team",
    "assist_tackle_3_player_id",
    "assist_tackle_4_team",
    "assist_tackle_4_player_id",
    "tackle_with_assist_1_team",
    "tackle_with_assist_1_player_id",
    "tackle_with_assist_2_team",
    "tackle_with_assist_2_player_id",
    # Play text (the gamebook description)
    "desc",
]

# Columns of PBP_COLUMNS that hold a team code. nflverse writes the Rams as LA and Washington as
# WAS in every one of them, so the loader normalizes each through ALIAS_TO_CANONICAL before
# caching. Yard-line text that opens with a team code ("LA 25": yrdln, end_yard_line and the
# drive yard lines) is cached as nflverse writes it.
PBP_TEAM_COLUMNS = (
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "side_of_field",
    "td_team",
    "penalty_team",
    "timeout_team",
    "return_team",
    "fumbled_1_team",
    "fumbled_2_team",
    "fumble_recovery_1_team",
    "fumble_recovery_2_team",
    "forced_fumble_player_1_team",
    "forced_fumble_player_2_team",
    "solo_tackle_1_team",
    "solo_tackle_2_team",
    "assist_tackle_1_team",
    "assist_tackle_2_team",
    "assist_tackle_3_team",
    "assist_tackle_4_team",
    "tackle_with_assist_1_team",
    "tackle_with_assist_2_team",
)

# Candidate column names for the special-teams play flag, in preference order.
# nflreadpy has published this flag under both names; the first one present wins.
PBP_SPECIAL_TEAMS_FLAG_CANDIDATES = ("special", "special_teams_play")

# Feature groups that can be ablated in walk-forward evaluation.
# A published column belongs to a group when any marker is a substring of its name,
# which covers the away_/home_/opponent_ prefixes and the _diff suffix at once.
FEATURE_GROUP_COLUMN_MARKERS: dict[str, tuple[str, ...]] = {
    # Every published play-by-play stat name is its own marker, so the group tracks
    # PBP_STATS automatically and stays disjoint from the older stat families.
    "pbp": (),  # populated below, once PBP_STATS is defined
    # The next opponent's team identity, one-hot encoded per side. The next opponent's
    # strength stays in the model through `*_next_opponent_win_pct`.
    "next_opponent_identity": ("next_opponent_abbr",),
    # Every column built from `data/qb_elos.csv` (the `../nfeloqb` Elo file): the Elo values
    # and their four-week trends. That Elo pulls each preseason rating toward one built from
    # betting win totals, so the group is market-informed (MARKET_INFORMED_FEATURE_GROUPS): a
    # run that leaves the market columns out keeps it unless it disables the group as well.
    "elo": (*ELO_COLUMNS, *ELO_TREND_COLUMNS),
    # This repository's own Elo, market-free preset (OWN_ELO_COLUMNS).
    "own_elo": tuple(OWN_ELO_TEAM_STATS),
    # The own Elo's win-total preset (OWN_WT_ELO_COLUMNS). Market-informed like the `elo` group,
    # and likewise kept by a run without the market columns unless it disables the group.
    "own_elo_wt": tuple(OWN_WT_ELO_TEAM_STATS),
    # Rare event counts that led the audit's noise-family ranking.
    "rare_events": (
        "special_teams_tds",
        "def_fumbles",
        "fumble_recovery_tds",
        "2pt_conversions",
        "def_safeties",
        "def_tds",
    ),
}

# Feature groups built on betting-market information without being line columns. Leaving the
# market columns out (`include_market` off) does not leave these out: a run with no market
# information at all disables them too, and a run that keeps any of them while leaving the
# market columns out logs a warning naming it (`ml.feature_spec.build_feature_spec`).
MARKET_INFORMED_FEATURE_GROUPS = ("elo", "own_elo_wt")

# Raw per-team-game counts and sums produced from play-by-play.
# These are carried through season-to-date aggregation so rates can be computed as a
# ratio of sums afterwards; they are not published in the final schema themselves,
# apart from the snap volumes that also appear in PBP_STATS. Each one feeds a published
# rate: a count no rate reads is not carried.
PBP_COUNT_COLUMNS = [
    "offensive_snaps",
    "defensive_snaps",
    "dropbacks",
    "carries",
    "dropbacks_allowed",
    "carries_allowed",
    "pass_epa_sum",
    "rush_epa_sum",
    "pass_epa_allowed_sum",
    "rush_epa_allowed_sum",
    "pass_success_count",
    "rush_success_count",
    "pass_success_allowed_count",
    "rush_success_allowed_count",
    "explosive_pass_count",
    "explosive_rush_count",
    "stuffed_rush_count",
    "explosive_pass_allowed_count",
    "explosive_rush_allowed_count",
    "stuffed_rush_allowed_count",
    "early_down_plays",
    "early_down_passes",
    "st_epa_for",
    "st_epa_against",
    "st_plays",
    "third_down_conversions",
    "third_down_attempts",
    "third_down_conversions_allowed",
    "third_down_attempts_allowed",
    "fourth_down_conversions",
    "fourth_down_attempts",
    "fourth_down_conversions_allowed",
    "fourth_down_attempts_allowed",
    "red_zone_trips",
    "red_zone_td_drives",
    "red_zone_trips_allowed",
    "red_zone_td_drives_allowed",
    "two_point_attempts",
    "two_point_successes",
    "two_point_attempts_allowed",
    "two_point_successes_allowed",
]

# Play-by-play stats published in the final schema (per team, prefixed with away_/home_).
# Each is derived in _compute_derived_metrics as a ratio of season-to-date summed
# components. Defensive metrics are named explicitly with `def_`/`_allowed` rather than
# relying on the generic `opponent_` mirror.
PBP_STATS = [
    # Snap volume (pace)
    "offensive_snaps",
    "defensive_snaps",
    # Per-snap EPA
    "off_pass_epa_per_snap",
    "off_rush_epa_per_snap",
    "def_pass_epa_allowed_per_snap",
    "def_rush_epa_allowed_per_snap",
    # Per-attempt EPA
    "epa_per_dropback",
    "epa_per_carry",
    "epa_per_dropback_allowed",
    "epa_per_carry_allowed",
    "epa_margin_per_play",
    # Success rates
    "success_rate",
    "pass_success_rate",
    "rush_success_rate",
    "success_rate_allowed",
    "pass_success_rate_allowed",
    "rush_success_rate_allowed",
    # Explosive and stuffed rates
    "explosive_pass_rate",
    "explosive_rush_rate",
    "explosive_pass_rate_allowed",
    "explosive_rush_rate_allowed",
    "stuffed_rush_rate",
    "stuffed_rush_rate_allowed",
    # Tendency
    "early_down_pass_rate",
    # Special teams
    "st_epa_margin_per_play",
]

FEATURE_GROUP_COLUMN_MARKERS["pbp"] = tuple(PBP_STATS)

# Schedule-adjusted team strength, solved per (season, week) from prior-week games only.
# Published per team and prefixed with away_/home_, plus a _diff companion.
STRENGTH_TEAM_STATS = [
    # Simultaneous ridge coefficients, centered per side. For the defensive pair a
    # HIGHER value is a BETTER defense: the solve models a team's per-snap EPA as
    # offense[team] - defense[opponent], so a large defense coefficient is what
    # suppresses the opponent's output.
    "adj_off_pass_epa_snap",
    "adj_off_rush_epa_snap",
    "adj_def_pass_epa_snap",
    "adj_def_rush_epa_snap",
    # Simple rating system companions on point margin and special-teams EPA margin.
    "adj_srs",
    "st_rating",
    # Display composite over within-snapshot standardized components.
    "adj_strength_composite",
    # Prior in-season games behind the solve, which drives the early-season prior blend.
    "strength_games_played",
]

# Schedule strength derived from the same pre-week snapshot.
# `sos_played_raw` is the one-hop, head-to-head-excluded companion to `sos_played_adj`.
SCHEDULE_STRENGTH_STATS = [
    "sos_played_adj",
    "sos_remaining_adj",
    "sos_played_raw",
]

# Every per-team strength column published in the final schema.
ADJUSTED_STRENGTH_STATS = [*STRENGTH_TEAM_STATS, *SCHEDULE_STRENGTH_STATS]

# Stats published per side without the away-minus-home `_diff` companion: the two teams in a
# game have almost always played the same number of games, so that diff is a constant zero
# outside bye weeks.
STATS_WITHOUT_DIFF = ["strength_games_played"]

# Columns the weekly snapshot builder emits, keyed by team. `adj_hfa` is the shared
# home-field term: one league-wide value per (season, week), so it is deliberately
# NOT published to the game schema, where it would become two identical away_/home_
# columns and an always-zero diff. It stays on the snapshot for reporting and for
# anything that consumes the solve directly.
STRENGTH_SNAPSHOT_STATS = [*STRENGTH_TEAM_STATS, "adj_hfa"]

# Columns of the per-team weekly strength file the ETL writes next to the game datasets:
# every published strength column plus the shared home-field term, keyed by
# (season, week, team). Teams on a bye are included, which is what lets a report read a
# team's pre-week strength for a week in which it has no game row.
STRENGTH_SNAPSHOT_FILE_COLUMNS = [
    "season",
    "week",
    "team_abbr",
    *ADJUSTED_STRENGTH_STATS,
    "adj_hfa",
]

# Base name of that file under DATA_PATH, without the extension.
STRENGTH_SNAPSHOTS_NAME = "strength_snapshots"

FEATURE_GROUP_COLUMN_MARKERS["strength"] = tuple(ADJUSTED_STRENGTH_STATS)

# Quarterback per-dropback family for the expected starter (`away_qb` / `home_qb`), built from
# play-by-play games of that quarterback strictly before the row's week, across teams and
# seasons. Published per side plus a _diff companion. The names deliberately contain no
# play-by-play or strength marker, so the three ablation groups stay disjoint.
QB_PBP_STATS = [
    # Career EPA per dropback, shrunk toward the league (scrambles and sacks included).
    "qb_dropback_epa",
    # EPA per dropback over the last QB_RECENT_GAMES games, shrunk toward the career rate.
    "qb_dropback_epa_recent",
    # Completion percentage over expectation (2006+), shrunk toward the league.
    "qb_cpoe",
    # Sacks per dropback, shrunk toward the league.
    "qb_sack_rate",
    # Adjusted net yards per attempt, shrunk toward the league.
    "qb_any_a",
    # ANY/A over the last QB_RECENT_GAMES games, shrunk toward the career rate.
    "qb_any_a_recent",
    # Career dropbacks behind the rates: the sample size the model sees.
    "qb_history_dropbacks",
]

FEATURE_GROUP_COLUMN_MARKERS["qb"] = tuple(QB_PBP_STATS)

# Pseudo-dropbacks of prior evidence in every quarterback rate: a career rate is
# (sum + K * league_rate) / (count + K) and a recent rate is shrunk the same way toward the
# career rate. About half a season of dropbacks; chosen, not tuned (see the sweep milestone).
QB_PRIOR_DROPBACKS = 300

# Quarterback games in the recent window (any teams, any seasons).
QB_RECENT_GAMES = 8

# Base name under DATA_PATH of the quarterback identity file, a read-only copy of
# `../nfeloqb/Other Data/meta_data.csv` (`name_id` -> `gsis_id`, the play-by-play passer id).
QB_META_DATA_NAME = "meta_data"

# Quarterback names used by the Elo source that the identity file spells differently.
QB_NAME_ALIASES = {
    "A.J. McCarron": "AJ McCarron",
    "Mitch Trubisky": "Mitchell Trubisky",
    "Robert Griffin": "Robert Griffin III",
}

# The team and quarterback ratings ported from nfl-sos-ratings (`nfl_predictor.ratings`), fit by
# the ETL on the games before each week (`utils/polars/ratings_build.py`). Published per side
# with a _diff companion; every name starts with "ratings_", which no other column contains, so
# the group stays disjoint from the strength, quarterback and TeamRankings rating columns.
RATINGS_FEATURE_STATS = [
    # Points per game against an average team on a neutral field, from opponent-adjusted EPA.
    "ratings_offense",
    "ratings_defense",
    "ratings_special_teams",
    "ratings_team",
    # Mean team rating of the opponents played, each rated without the team's games (null in
    # weeks 1 and 2, before any opponent has played someone else).
    "ratings_sos",
    # The expected starter's adjusted EPA per dropback, from this season's games before the week.
    "ratings_qb",
    # His dropbacks behind it (0 before his first dropback of the season).
    "ratings_qb_dropbacks",
]

FEATURE_GROUP_COLUMN_MARKERS["ratings"] = tuple(RATINGS_FEATURE_STATS)

# Directory under DATA_PATH for the published ratings tables, `{season}_{kind}.parquet`.
RATINGS_DIRNAME = "ratings"

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
    "times_sacked",
    "def_sacks",
    "def_interceptions",
    "interceptions_thrown",
    "turnover_margin",
    # Derived ratio metrics that would be duplicates
    "yards_per_point",
    "yards_per_point_margin",
    "points_per_play",
    "points_per_play_margin",
    "penalty_yards_per_penalty",
    # Play-by-play counts: the *_allowed_* columns already carry the opponent's offense
    # for the same game, so a generic opponent_ mirror would duplicate them.
    *PBP_COUNT_COLUMNS,
    # Context flag rather than a stat: opponent_is_home is the exact inverse of is_home.
    "is_home",
]

# Excluded mirrors that the ETL still builds per game because a derived metric reads them,
# and drops again when the final schema is selected. `opponent_points_per_play` divides
# `points_allowed` by the opponent's plays (`opponent_pass_attempts + opponent_rush_attempts +
# opponent_times_sacked`), so the times-sacked mirror has to exist at derivation time even
# though it is never published.
OPPONENT_MIRROR_INTERMEDIATES: tuple[str, ...] = ("times_sacked",)

# Columns of a per-team-game row that are NOT part of the team's own box score: the row's
# identity, the scoring columns taken from the schedule, and the play-by-play counts and
# their context flag. The box score is the complement of this set, defined this way rather
# than enumerated because the team-stats source publishes well over a hundred box-score
# columns and adds more between seasons. It names what stays when a row that describes two
# teams at once has to give up the values it cannot own.
TEAM_GAME_NON_BOX_SCORE_COLUMNS: tuple[str, ...] = (
    "season",
    "week",
    "team_abbr",
    "opponent_abbr",
    "season_type",
    "game_id",
    "points_scored",
    "points_allowed",
    "scoring_margin",
    "is_home",
    *PBP_COUNT_COLUMNS,
)

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
    "rushing_epa",
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
    "BOS99": {
        "name": "Foxboro Stadium",
        "city": "Foxborough",
        "state": "MA",
        "elevation_ft": 313.0,
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
    "CIN99": {
        "name": "Cinergy Field",
        "city": "Cincinnati",
        "state": "OH",
        "elevation_ft": 505.0,
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
    "PIT99": {
        "name": "Three Rivers Stadium",
        "city": "Pittsburgh",
        "state": "PA",
        "elevation_ft": 731.0,
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
    "DEN99": {
        "name": "Mile High Stadium",
        "city": "Denver",
        "state": "CO",
        "elevation_ft": 5210.0,
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
    "PHI99": {
        "name": "Veterans Stadium",
        "city": "Philadelphia",
        "state": "PA",
        "elevation_ft": 21.0,
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
    "CHI99": {
        "name": "Memorial Stadium (Champaign)",
        "city": "Champaign",
        "state": "IL",
        "elevation_ft": 732.0,
    },
    "DET00": {
        "name": "Ford Field",
        "city": "Detroit",
        "state": "MI",
        "elevation_ft": 600.0,
    },
    "DET99": {
        "name": "Pontiac Silverdome",
        "city": "Pontiac",
        "state": "MI",
        "elevation_ft": 892.0,
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
    "SEA98": {
        "name": "Kingdome",
        "city": "Seattle",
        "state": "WA",
        "elevation_ft": 18.0,
    },
    "SEA99": {
        "name": "Husky Stadium",
        "city": "Seattle",
        "state": "WA",
        "elevation_ft": 32.0,
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
    "MAD01": {
        "name": "Santiago Bernabéu Stadium",
        "city": "Madrid",
        "state": "ES",
        "elevation_ft": 2315.0,
    },
    "MEL00": {
        "name": "Melbourne Cricket Ground",
        "city": "Melbourne",
        "state": "AU",
        "elevation_ft": 54.0,
    },
    "MEX00": {
        "name": "Estadio Azteca",
        "city": "Mexico City",
        "state": "MX",
        "elevation_ft": 7380.0,
    },
    # The Allianz Arena again, under the name it has at UEFA and NFL events.
    "MUN01": {
        "name": "Allianz Arena",
        "city": "Munich",
        "state": "DE",
        "elevation_ft": 1617.0,
    },
    "PAR00": {
        "name": "Stade de France",
        "city": "Saint-Denis",
        "state": "FR",
        "elevation_ft": 138.0,
    },
    "RIO00": {
        "name": "Maracanã Stadium",
        "city": "Rio de Janeiro",
        "state": "BR",
        "elevation_ft": 38.0,
    },
    "SAO00": {
        "name": "Arena Corinthians (Neo Química Arena)",
        "city": "São Paulo",
        "state": "BR",
        "elevation_ft": 2490.0,
    },
}

# Approximate stadium coordinates (latitude, longitude in degrees, north and east positive) for
# every nflverse `stadium_id` on the 1999-2026 schedules, temporary and international venues
# included. Only the own Elo's travel-distance term reads them: 4 Elo points per 1,000 miles,
# so an error of a few miles moves a rating by a hundredth of a point.
STADIUM_COORDINATES: dict[str, tuple[float, float]] = {
    "ATL00": (33.758, -84.401),  # Georgia Dome, Atlanta
    "ATL97": (33.755, -84.401),  # Mercedes-Benz Stadium, Atlanta
    "BAL00": (39.278, -76.623),  # M&T Bank Stadium, Baltimore
    "BOS00": (42.091, -71.264),  # Gillette Stadium, Foxborough
    "BOS99": (42.093, -71.266),  # Foxboro Stadium, Foxborough
    "BRG00": (30.412, -91.184),  # Tiger Stadium, Baton Rouge
    "BUF00": (42.774, -78.787),  # Highmark Stadium, Orchard Park
    "BUF01": (43.641, -79.389),  # Rogers Centre, Toronto
    "CAR00": (35.226, -80.853),  # Bank of America Stadium, Charlotte
    "CHI98": (41.862, -87.617),  # Soldier Field, Chicago
    "CHI99": (40.099, -88.236),  # Memorial Stadium, Champaign
    "CIN00": (39.095, -84.516),  # Paycor Stadium, Cincinnati
    "CIN99": (39.097, -84.508),  # Cinergy Field, Cincinnati
    "CLE00": (41.506, -81.700),  # Cleveland Browns Stadium, Cleveland
    "DAL00": (32.748, -97.093),  # AT&T Stadium, Arlington
    "DAL99": (32.840, -96.911),  # Texas Stadium, Irving
    "DEN00": (39.744, -105.020),  # Empower Field at Mile High, Denver
    "DEN99": (39.746, -105.022),  # Mile High Stadium, Denver
    "DET00": (42.340, -83.046),  # Ford Field, Detroit
    "DET99": (42.646, -83.255),  # Pontiac Silverdome, Pontiac
    "FRA00": (50.069, 8.645),  # Deutsche Bank Park, Frankfurt
    "GER00": (48.219, 11.625),  # Allianz Arena, Munich
    "GNB00": (44.501, -88.062),  # Lambeau Field, Green Bay
    "HOU00": (29.685, -95.411),  # NRG Stadium, Houston
    "IND00": (39.760, -86.164),  # Lucas Oil Stadium, Indianapolis
    "IND99": (39.763, -86.163),  # RCA Dome, Indianapolis
    "JAX00": (30.324, -81.637),  # EverBank Stadium, Jacksonville
    "KAN00": (39.049, -94.484),  # Arrowhead Stadium, Kansas City
    "LAX01": (33.953, -118.339),  # SoFi Stadium, Inglewood
    "LAX97": (33.864, -118.261),  # StubHub Center, Carson
    "LAX99": (34.014, -118.288),  # Los Angeles Memorial Coliseum
    "LON00": (51.556, -0.280),  # Wembley Stadium, London
    "LON01": (51.456, -0.342),  # Twickenham Stadium, London
    "LON02": (51.604, -0.066),  # Tottenham Hotspur Stadium, London
    "MAD01": (40.453, -3.688),  # Santiago Bernabeu, Madrid
    "MEL00": (-37.820, 144.983),  # Melbourne Cricket Ground
    "MEX00": (19.303, -99.150),  # Estadio Azteca, Mexico City
    "MIA00": (25.958, -80.239),  # Hard Rock Stadium, Miami Gardens
    "MIN00": (44.974, -93.258),  # Metrodome, Minneapolis
    "MIN01": (44.974, -93.258),  # U.S. Bank Stadium, Minneapolis
    "MIN98": (44.977, -93.225),  # TCF Bank Stadium, Minneapolis
    "MUN01": (48.219, 11.625),  # FC Bayern Munich Stadium (Allianz Arena), Munich
    "NAS00": (36.166, -86.771),  # Nissan Stadium, Nashville
    "NOR00": (29.951, -90.081),  # Superdome, New Orleans
    "NYC00": (40.812, -74.077),  # Giants Stadium, East Rutherford
    "NYC01": (40.814, -74.074),  # MetLife Stadium, East Rutherford
    "OAK00": (37.752, -122.201),  # Oakland Coliseum
    "PAR00": (48.924, 2.360),  # Stade de France, Saint-Denis
    "PHI00": (39.901, -75.168),  # Lincoln Financial Field, Philadelphia
    "PHI99": (39.907, -75.171),  # Veterans Stadium, Philadelphia
    "PHO00": (33.528, -112.263),  # State Farm Stadium, Glendale
    "PHO99": (33.426, -111.933),  # Sun Devil Stadium, Tempe
    "PIT00": (40.447, -80.016),  # Acrisure Stadium, Pittsburgh
    "PIT99": (40.447, -80.013),  # Three Rivers Stadium, Pittsburgh
    "RIO00": (-22.912, -43.230),  # Maracana Stadium, Rio de Janeiro
    "SAN00": (29.417, -98.479),  # Alamodome, San Antonio
    "SAO00": (-23.545, -46.474),  # Arena Corinthians, Sao Paulo
    "SDG00": (32.783, -117.120),  # Qualcomm Stadium, San Diego
    "SEA00": (47.595, -122.332),  # Lumen Field, Seattle
    "SEA98": (47.595, -122.332),  # Kingdome, Seattle
    "SEA99": (47.650, -122.302),  # Husky Stadium, Seattle
    "SFO00": (37.714, -122.386),  # Candlestick Park, San Francisco
    "SFO01": (37.403, -121.970),  # Levi's Stadium, Santa Clara
    "STL00": (38.633, -90.188),  # Edward Jones Dome, St. Louis
    "TAM00": (27.976, -82.503),  # Raymond James Stadium, Tampa
    "VEG00": (36.091, -115.184),  # Allegiant Stadium, Las Vegas
    "WAS00": (38.908, -76.864),  # Northwest Stadium, Landover
}
