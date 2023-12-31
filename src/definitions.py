import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data")
ELO_DATA_URL = "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
AWAY_STATS = [
    "away_first_downs",
    "away_fourth_down_attempts",
    "away_fourth_down_conversions",
    "away_fumbles",
    "away_fumbles_lost",
    "away_interceptions",
    "away_net_pass_yards",
    "away_pass_attempts",
    "away_pass_completions",
    "away_pass_touchdowns",
    "away_pass_yards",
    "away_penalties",
    "away_points",
    "away_rush_attempts",
    "away_rush_touchdowns",
    "away_rush_yards",
    "away_third_down_attempts",
    "away_third_down_conversions",
    "away_time_of_possession",
    "away_times_sacked",
    "away_total_yards",
    "away_turnovers",
    "away_yards_from_penalties",
    "away_yards_lost_from_sacks",
]
AWAY_STATS_DROP = {
    "away_first_downs": "first_downs",
    "away_fourth_down_attempts": "fourth_down_attempts",
    "away_fourth_down_conversions": "fourth_down_conversions",
    "away_fumbles": "fumbles",
    "away_fumbles_lost": "fumbles_lost",
    "away_interceptions": "interceptions",
    "away_net_pass_yards": "net_pass_yards",
    "away_pass_attempts": "pass_attempts",
    "away_pass_completions": "pass_completions",
    "away_pass_touchdowns": "pass_touchdowns",
    "away_pass_yards": "pass_yards",
    "away_penalties": "penalties",
    "away_points": "points",
    "away_rush_attempts": "rush_attempts",
    "away_rush_touchdowns": "rush_touchdowns",
    "away_rush_yards": "rush_yards",
    "away_third_down_attempts": "third_down_attempts",
    "away_third_down_conversions": "third_down_conversions",
    "away_time_of_possession": "time_of_possession",
    "away_times_sacked": "times_sacked",
    "away_total_yards": "total_yards",
    "away_turnovers": "turnovers",
    "away_yards_from_penalties": "yards_from_penalties",
    "away_yards_lost_from_sacks": "yards_lost_from_sacks",
}
HOME_STATS = [
    "home_first_downs",
    "home_fourth_down_attempts",
    "home_fourth_down_conversions",
    "home_fumbles",
    "home_fumbles_lost",
    "home_interceptions",
    "home_net_pass_yards",
    "home_pass_attempts",
    "home_pass_completions",
    "home_pass_touchdowns",
    "home_pass_yards",
    "home_penalties",
    "home_points",
    "home_rush_attempts",
    "home_rush_touchdowns",
    "home_rush_yards",
    "home_third_down_attempts",
    "home_third_down_conversions",
    "home_time_of_possession",
    "home_times_sacked",
    "home_total_yards",
    "home_turnovers",
    "home_yards_from_penalties",
    "home_yards_lost_from_sacks",
]
HOME_STATS_DROP = {
    "home_first_downs": "first_downs",
    "home_fourth_down_attempts": "fourth_down_attempts",
    "home_fourth_down_conversions": "fourth_down_conversions",
    "home_fumbles": "fumbles",
    "home_fumbles_lost": "fumbles_lost",
    "home_interceptions": "interceptions",
    "home_net_pass_yards": "net_pass_yards",
    "home_pass_attempts": "pass_attempts",
    "home_pass_completions": "pass_completions",
    "home_pass_touchdowns": "pass_touchdowns",
    "home_pass_yards": "pass_yards",
    "home_penalties": "penalties",
    "home_points": "points",
    "home_rush_attempts": "rush_attempts",
    "home_rush_touchdowns": "rush_touchdowns",
    "home_rush_yards": "rush_yards",
    "home_third_down_attempts": "third_down_attempts",
    "home_third_down_conversions": "third_down_conversions",
    "home_time_of_possession": "time_of_possession",
    "home_times_sacked": "times_sacked",
    "home_total_yards": "total_yards",
    "home_turnovers": "turnovers",
    "home_yards_from_penalties": "yards_from_penalties",
    "home_yards_lost_from_sacks": "yards_lost_from_sacks",
}
AGG_RENAME_AWAY = {
    "win_perc": "away_win_perc",
    "first_downs": "away_first_downs",
    "fumbles": "away_fumbles",
    "fumbles_lost": "away_fumbles_lost",
    "interceptions": "away_interceptions",
    "net_pass_yards": "away_net_pass_yards",
    "pass_attempts": "away_pass_attempts",
    "pass_completions": "away_pass_completions",
    "pass_touchdowns": "away_pass_touchdowns",
    "pass_yards": "away_pass_yards",
    "penalties": "away_penalties",
    "points": "away_points",
    "rush_attempts": "away_rush_attempts",
    "rush_touchdowns": "away_rush_touchdowns",
    "rush_yards": "away_rush_yards",
    "time_of_possession": "away_time_of_possession",
    "times_sacked": "away_times_sacked",
    "total_yards": "away_total_yards",
    "turnovers": "away_turnovers",
    "yards_from_penalties": "away_yards_from_penalties",
    "yards_lost_from_sacks": "away_yards_lost_from_sacks",
    "fourth_down_perc": "away_fourth_down_perc",
    "third_down_perc": "away_third_down_perc",
}
AGG_RENAME_HOME = {
    "win_perc": "home_win_perc",
    "first_downs": "home_first_downs",
    "fumbles": "home_fumbles",
    "fumbles_lost": "home_fumbles_lost",
    "interceptions": "home_interceptions",
    "net_pass_yards": "home_net_pass_yards",
    "pass_attempts": "home_pass_attempts",
    "pass_completions": "home_pass_completions",
    "pass_touchdowns": "home_pass_touchdowns",
    "pass_yards": "home_pass_yards",
    "penalties": "home_penalties",
    "points": "home_points",
    "rush_attempts": "home_rush_attempts",
    "rush_touchdowns": "home_rush_touchdowns",
    "rush_yards": "home_rush_yards",
    "time_of_possession": "home_time_of_possession",
    "times_sacked": "home_times_sacked",
    "total_yards": "home_total_yards",
    "turnovers": "home_turnovers",
    "yards_from_penalties": "home_yards_from_penalties",
    "yards_lost_from_sacks": "home_yards_lost_from_sacks",
    "fourth_down_perc": "home_fourth_down_perc",
    "third_down_perc": "home_third_down_perc",
}
AGG_MERGE_ON = [
    "away_name",
    "away_abbr",
    "home_name",
    "home_abbr",
    "winning_name",
    "winning_abbr",
    "week",
]
AGG_DROP_COLS = [
    "away_win_perc",
    "away_first_downs",
    "away_fumbles",
    "away_fumbles_lost",
    "away_interceptions",
    "away_net_pass_yards",
    "away_pass_attempts",
    "away_pass_completions",
    "away_pass_touchdowns",
    "away_pass_yards",
    "away_penalties",
    "away_points",
    "away_rush_attempts",
    "away_rush_touchdowns",
    "away_rush_yards",
    "away_time_of_possession",
    "away_times_sacked",
    "away_total_yards",
    "away_turnovers",
    "away_yards_from_penalties",
    "away_yards_lost_from_sacks",
    "away_fourth_down_perc",
    "away_third_down_perc",
    "home_win_perc",
    "home_first_downs",
    "home_fumbles",
    "home_fumbles_lost",
    "home_interceptions",
    "home_net_pass_yards",
    "home_pass_attempts",
    "home_pass_completions",
    "home_pass_touchdowns",
    "home_pass_yards",
    "home_penalties",
    "home_points",
    "home_rush_attempts",
    "home_rush_touchdowns",
    "home_rush_yards",
    "home_time_of_possession",
    "home_times_sacked",
    "home_total_yards",
    "home_turnovers",
    "home_yards_from_penalties",
    "home_yards_lost_from_sacks",
    "home_fourth_down_perc",
    "home_third_down_perc",
]
ELO_DROP_COLS = [
    "season",
    "neutral",
    "playoff",
    "elo_prob1",
    "elo_prob2",
    "elo1_post",
    "elo2_post",
    "qbelo1_pre",
    "qbelo2_pre",
    "qb1",
    "qb2",
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
]
ELO_TEAMS = [
    "KC",
    "JAX",
    "CAR",
    "BAL",
    "BUF",
    "MIN",
    "DET",
    "ATL",
    "NE",
    "WSH",
    "CIN",
    "NO",
    "SF",
    "LAR",
    "NYG",
    "DEN",
    "CLE",
    "IND",
    "TEN",
    "NYJ",
    "TB",
    "MIA",
    "PIT",
    "PHI",
    "GB",
    "CHI",
    "DAL",
    "ARI",
    "LAC",
    "HOU",
    "SEA",
    "OAK",
]
STD_TEAMS = [
    "kan",
    "jax",
    "car",
    "rav",
    "buf",
    "min",
    "det",
    "atl",
    "nwe",
    "was",
    "cin",
    "nor",
    "sfo",
    "ram",
    "nyg",
    "den",
    "cle",
    "clt",
    "oti",
    "nyj",
    "tam",
    "mia",
    "pit",
    "phi",
    "gnb",
    "chi",
    "dal",
    "crd",
    "sdg",
    "htx",
    "sea",
    "rai",
]
