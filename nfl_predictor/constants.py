import os


class Constants:
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
    AWAY_STATS_RENAME = {
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
    HOME_STATS_RENAME = {
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
    TEAMS = [
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
        "LAR",
        "LAC",
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
    ]
    PBP_TEAMS = [
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
        "LA",
        "LAC",
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
    ACTIVE_QB_IDS = {
        "00-0019596": {
            "name": "Tom Brady",
            "draft_year": 2000,
            "draft_number": 199,
        },
        "00-0020531": {
            "name": "Drew Brees",
            "draft_year": 2001,
            "draft_number": 32,
        },
        "00-0020679": {
            "name": "Shaun Hill",
            "draft_year": 2002,
            "draft_number": 263,
        },
        "00-0021206": {
            "name": "Josh McCown",
            "draft_year": 2002,
            "draft_number": 81,
        },
        "00-0021429": {
            "name": "Carson Palmer",
            "draft_year": 2003,
            "draft_number": 1,
        },
        "00-0022787": {
            "name": "Matt Schaub",
            "draft_year": 2004,
            "draft_number": 90,
        },
        "00-0022803": {
            "name": "Eli Manning",
            "draft_year": 2004,
            "draft_number": 1,
        },
        "00-0022924": {
            "name": "Ben Roethlisberger",
            "draft_year": 2004,
            "draft_number": 11,
        },
        "00-0022942": {
            "name": "Philip Rivers",
            "draft_year": 2004,
            "draft_number": 4,
        },
        "00-0023436": {
            "name": "Alex Smith",
            "draft_year": 2005,
            "draft_number": 1,
        },
        "00-0023459": {
            "name": "Aaron Rodgers",
            "draft_year": 2005,
            "draft_number": 24,
        },
        "00-0023645": {
            "name": "Derek Anderson",
            "draft_year": 2005,
            "draft_number": 213,
        },
        "00-0023662": {
            "name": "Matt Cassel",
            "draft_year": 2005,
            "draft_number": 230,
        },
        "00-0023682": {
            "name": "Ryan Fitzpatrick",
            "draft_year": 2005,
            "draft_number": 250,
        },
        "00-0024226": {
            "name": "Jay Cutler",
            "draft_year": 2006,
            "draft_number": 11,
        },
        "00-0025430": {
            "name": "Drew Stanton",
            "draft_year": 2007,
            "draft_number": 43,
        },
        "00-0025708": {
            "name": "Matt Moore",
            "draft_year": 2007,
            "draft_number": 263,
        },
        "00-0026143": {
            "name": "Matt Ryan",
            "draft_year": 2008,
            "draft_number": 3,
        },
        "00-0026158": {
            "name": "Joe Flacco",
            "draft_year": 2008,
            "draft_number": 18,
        },
        "00-0026197": {
            "name": "Chad Henne",
            "draft_year": 2008,
            "draft_number": 57,
        },
        "00-0026300": {
            "name": "Josh Johnson",
            "draft_year": 2008,
            "draft_number": 160,
        },
        "00-0026498": {
            "name": "Matthew Stafford",
            "draft_year": 2009,
            "draft_number": 1,
        },
        "00-0026544": {
            "name": "Chase Daniel",
            "draft_year": 2009,
            "draft_number": 263,
        },
        "00-0026625": {
            "name": "Brian Hoyer",
            "draft_year": 2009,
            "draft_number": 263,
        },
        "00-0026898": {
            "name": "Mark Sanchez",
            "draft_year": 2009,
            "draft_number": 5,
        },
        "00-0027688": {
            "name": "Colt McCoy",
            "draft_year": 2010,
            "draft_number": 85,
        },
        "00-0027854": {
            "name": "Sam Bradford",
            "draft_year": 2010,
            "draft_number": 1,
        },
        "00-0027939": {
            "name": "Cam Newton",
            "draft_year": 2011,
            "draft_number": 1,
        },
        "00-0027948": {
            "name": "Blaine Gabbert",
            "draft_year": 2011,
            "draft_number": 10,
        },
        "00-0027973": {
            "name": "Andy Dalton",
            "draft_year": 2011,
            "draft_number": 35,
        },
        "00-0027974": {
            "name": "Colin Kaepernick",
            "draft_year": 2011,
            "draft_number": 36,
        },
        "00-0028090": {
            "name": "T.J. Yates",
            "draft_year": 2011,
            "draft_number": 152,
        },
        "00-0028118": {
            "name": "Tyrod Taylor",
            "draft_year": 2011,
            "draft_number": 180,
        },
        "00-0028595": {
            "name": "Scott Tolzien",
            "draft_year": 2011,
            "draft_number": 263,
        },
        "00-0028986": {
            "name": "Case Keenum",
            "draft_year": 2012,
            "draft_number": 263,
        },
        "00-0029263": {
            "name": "Russell Wilson",
            "draft_year": 2012,
            "draft_number": 75,
        },
        "00-0029567": {
            "name": "Nick Foles",
            "draft_year": 2012,
            "draft_number": 88,
        },
        "00-0029604": {
            "name": "Kirk Cousins",
            "draft_year": 2012,
            "draft_number": 102,
        },
        "00-0029665": {
            "name": "Robert Griffin",
            "draft_year": 2012,
            "draft_number": 2,
        },
        "00-0029668": {
            "name": "Andrew Luck",
            "draft_year": 2012,
            "draft_number": 1,
        },
        "00-0029682": {
            "name": "Brock Osweiler",
            "draft_year": 2012,
            "draft_number": 57,
        },
        "00-0029701": {
            "name": "Ryan Tannehill",
            "draft_year": 2012,
            "draft_number": 8,
        },
        "00-0030419": {
            "name": "Matt McGloin",
            "draft_year": 2013,
            "draft_number": 263,
        },
        "00-0030520": {
            "name": "Mike Glennon",
            "draft_year": 2013,
            "draft_number": 73,
        },
        "00-0030524": {
            "name": "Landry Jones",
            "draft_year": 2013,
            "draft_number": 115,
        },
        "00-0030526": {
            "name": "EJ Manuel",
            "draft_year": 2013,
            "draft_number": 16,
        },
        "00-0030533": {
            "name": "Matt Barkley",
            "draft_year": 2013,
            "draft_number": 98,
        },
        "00-0030565": {
            "name": "Geno Smith",
            "draft_year": 2013,
            "draft_number": 39,
        },
        "00-0031064": {
            "name": "Tom Savage",
            "draft_year": 2014,
            "draft_number": 135,
        },
        "00-0031237": {
            "name": "Teddy Bridgewater",
            "draft_year": 2014,
            "draft_number": 32,
        },
        "00-0031260": {
            "name": "Logan Thomas",
            "draft_year": 2014,
            "draft_number": 120,
        },
        "00-0031280": {
            "name": "Derek Carr",
            "draft_year": 2014,
            "draft_number": 36,
        },
        "00-0031288": {
            "name": "A.J. McCarron",
            "draft_year": 2014,
            "draft_number": 164,
        },
        "00-0031345": {
            "name": "Jimmy Garoppolo",
            "draft_year": 2014,
            "draft_number": 62,
        },
        "00-0031395": {
            "name": "Garrett Gilbert",
            "draft_year": 2014,
            "draft_number": 263,
        },
        "00-0031407": {
            "name": "Blake Bortles",
            "draft_year": 2014,
            "draft_number": 3,
        },
        "00-0031503": {
            "name": "Jameis Winston",
            "draft_year": 2015,
            "draft_number": 1,
        },
        "00-0031568": {
            "name": "Bryce Petty",
            "draft_year": 2015,
            "draft_number": 103,
        },
        "00-0031589": {
            "name": "Brett Hundley",
            "draft_year": 2015,
            "draft_number": 147,
        },
        "00-0031800": {
            "name": "Taylor Heinicke",
            "draft_year": 2015,
            "draft_number": 263,
        },
        "00-0032156": {
            "name": "Trevor Siemian",
            "draft_year": 2015,
            "draft_number": 250,
        },
        "00-0032245": {
            "name": "Sean Mannion",
            "draft_year": 2015,
            "draft_number": 89,
        },
        "00-0032268": {
            "name": "Marcus Mariota",
            "draft_year": 2015,
            "draft_number": 2,
        },
        "00-0032434": {
            "name": "Brandon Allen",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032436": {
            "name": "Jeff Driskel",
            "draft_year": 2016,
            "draft_number": 207,
        },
        "00-0032446": {
            "name": "Brandon Doughty",
            "draft_year": 2016,
            "draft_number": 223,
        },
        "00-0032462": {
            "name": "Trevone Boykin",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032614": {
            "name": "Joel Stave",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032630": {
            "name": "Joe Callahan",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032784": {
            "name": "Kevin Hogan",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032792": {
            "name": "Nate Sudfeld",
            "draft_year": 2016,
            "draft_number": 187,
        },
        "00-0032893": {
            "name": "Connor Cook",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0032950": {
            "name": "Carson Wentz",
            "draft_year": 2016,
            "draft_number": 2,
        },
        "00-0033077": {
            "name": "Dak Prescott",
            "draft_year": 2016,
            "draft_number": 135,
        },
        "00-0033104": {
            "name": "Cody Kessler",
            "draft_year": 2016,
            "draft_number": 263,
        },
        "00-0033106": {
            "name": "Jared Goff",
            "draft_year": 2016,
            "draft_number": 1,
        },
        "00-0033108": {
            "name": "Paxton Lynch",
            "draft_year": 2016,
            "draft_number": 26,
        },
        "00-0033119": {
            "name": "Jacoby Brissett",
            "draft_year": 2016,
            "draft_number": 91,
        },
        "00-0033238": {
            "name": "Alek Torgersen",
            "draft_year": 2017,
            "draft_number": 263,
        },
        "00-0033275": {
            "name": "P.J. Walker",
            "draft_year": 2017,
            "draft_number": 263,
        },
        "00-0033319": {
            "name": "Nick Mullens",
            "draft_year": 2017,
            "draft_number": 263,
        },
        "00-0033357": {
            "name": "Taysom Hill",
            "draft_year": 2017,
            "draft_number": 263,
        },
        "00-0033537": {
            "name": "Deshaun Watson",
            "draft_year": 2017,
            "draft_number": 12,
        },
        "00-0033550": {
            "name": "Davis Webb",
            "draft_year": 2017,
            "draft_number": 87,
        },
        "00-0033662": {
            "name": "Cooper Rush",
            "draft_year": 2017,
            "draft_number": 263,
        },
        "00-0033869": {
            "name": "Mitchell Trubisky",
            "draft_year": 2017,
            "draft_number": 2,
        },
        "00-0033873": {
            "name": "Patrick Mahomes",
            "draft_year": 2017,
            "draft_number": 10,
        },
        "00-0033899": {
            "name": "DeShone Kizer",
            "draft_year": 2017,
            "draft_number": 52,
        },
        "00-0033936": {
            "name": "C.J. Beathard",
            "draft_year": 2017,
            "draft_number": 104,
        },
        "00-0033949": {
            "name": "Joshua Dobbs",
            "draft_year": 2017,
            "draft_number": 135,
        },
        "00-0033958": {
            "name": "Nathan Peterman",
            "draft_year": 2017,
            "draft_number": 171,
        },
        "00-0034126": {
            "name": "J.T. Barrett",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034177": {
            "name": "Tim Boyle",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034343": {
            "name": "Josh Rosen",
            "draft_year": 2018,
            "draft_number": 10,
        },
        "00-0034401": {
            "name": "Mike White",
            "draft_year": 2018,
            "draft_number": 171,
        },
        "00-0034412": {
            "name": "Luke Falk",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034438": {
            "name": "Logan Woodside",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034478": {
            "name": "Chad Kanoff",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034577": {
            "name": "Kyle Allen",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034732": {
            "name": "Alex McGough",
            "draft_year": 2018,
            "draft_number": 220,
        },
        "00-0034757": {
            "name": "Nick Stevens",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034771": {
            "name": "Mason Rudolph",
            "draft_year": 2018,
            "draft_number": 76,
        },
        "00-0034796": {
            "name": "Lamar Jackson",
            "draft_year": 2018,
            "draft_number": 32,
        },
        "00-0034855": {
            "name": "Baker Mayfield",
            "draft_year": 2018,
            "draft_number": 1,
        },
        "00-0034857": {
            "name": "Josh Allen",
            "draft_year": 2018,
            "draft_number": 7,
        },
        "00-0034869": {
            "name": "Sam Darnold",
            "draft_year": 2018,
            "draft_number": 3,
        },
        "00-0034899": {
            "name": "John Wolford",
            "draft_year": 2018,
            "draft_number": 263,
        },
        "00-0034955": {
            "name": "Brett Rypien",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035040": {
            "name": "David Blough",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035077": {
            "name": "Manny Wilkins",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035100": {
            "name": "Jake Browning",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035146": {
            "name": "Trace McSorley",
            "draft_year": 2019,
            "draft_number": 197,
        },
        "00-0035228": {
            "name": "Kyler Murray",
            "draft_year": 2019,
            "draft_number": 1,
        },
        "00-0035232": {
            "name": "Dwayne Haskins",
            "draft_year": 2019,
            "draft_number": 15,
        },
        "00-0035251": {
            "name": "Will Grier",
            "draft_year": 2019,
            "draft_number": 100,
        },
        "00-0035264": {
            "name": "Jarrett Stidham",
            "draft_year": 2019,
            "draft_number": 133,
        },
        "00-0035282": {
            "name": "Easton Stick",
            "draft_year": 2019,
            "draft_number": 166,
        },
        "00-0035283": {
            "name": "Clayton Thorson",
            "draft_year": 2019,
            "draft_number": 167,
        },
        "00-0035289": {
            "name": "Gardner Minshew",
            "draft_year": 2019,
            "draft_number": 178,
        },
        "00-0035483": {
            "name": "Drew Anderson",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035577": {
            "name": "Devlin Hodges",
            "draft_year": 2019,
            "draft_number": 263,
        },
        "00-0035652": {
            "name": "Ryan Finley",
            "draft_year": 2019,
            "draft_number": 104,
        },
        "00-0035704": {
            "name": "Drew Lock",
            "draft_year": 2019,
            "draft_number": 42,
        },
        "00-0035710": {
            "name": "Daniel Jones",
            "draft_year": 2019,
            "draft_number": 6,
        },
        "00-0035812": {
            "name": "Case Cookus",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035864": {
            "name": "Kendall Hinton",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035937": {
            "name": "Josh Love",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035939": {
            "name": "Bryce Perkins",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035968": {
            "name": "Jalen Morton",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035988": {
            "name": "Anthony Gordon",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0035993": {
            "name": "Tyler Huntley",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0036022": {
            "name": "Steven Montez",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0036052": {
            "name": "Reid Sinnett",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0036092": {
            "name": "Brian Lewerke",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0036197": {
            "name": "Jake Fromm",
            "draft_year": 2020,
            "draft_number": 167,
        },
        "00-0036212": {
            "name": "Tua Tagovailoa",
            "draft_year": 2020,
            "draft_number": 5,
        },
        "00-0036257": {
            "name": "Cole McDonald",
            "draft_year": 2020,
            "draft_number": 224,
        },
        "00-0036264": {
            "name": "Jordan Love",
            "draft_year": 2020,
            "draft_number": 26,
        },
        "00-0036312": {
            "name": "Jake Luton",
            "draft_year": 2020,
            "draft_number": 189,
        },
        "00-0036355": {
            "name": "Justin Herbert",
            "draft_year": 2020,
            "draft_number": 6,
        },
        "00-0036384": {
            "name": "Ben DiNucci",
            "draft_year": 2020,
            "draft_number": 231,
        },
        "00-0036389": {
            "name": "Jalen Hurts",
            "draft_year": 2020,
            "draft_number": 53,
        },
        "00-0036442": {
            "name": "Joe Burrow",
            "draft_year": 2020,
            "draft_number": 1,
        },
        "00-0036468": {
            "name": "Kai Locksley",
            "draft_year": 2021,
            "draft_number": 263,
        },
        "00-0036679": {
            "name": "Shane Buechele",
            "draft_year": 2021,
            "draft_number": 263,
        },
        "00-0036825": {
            "name": "Feleipe Franks",
            "draft_year": 2021,
            "draft_number": 263,
        },
        "00-0036879": {
            "name": "Sam Ehlinger",
            "draft_year": 2021,
            "draft_number": 218,
        },
        "00-0036898": {
            "name": "Davis Mills",
            "draft_year": 2021,
            "draft_number": 67,
        },
        "00-0036928": {
            "name": "Kyle Trask",
            "draft_year": 2021,
            "draft_number": 64,
        },
        "00-0036929": {
            "name": "Ian Book",
            "draft_year": 2021,
            "draft_number": 133,
        },
        "00-0036945": {
            "name": "Justin Fields",
            "draft_year": 2021,
            "draft_number": 11,
        },
        "00-0036971": {
            "name": "Trevor Lawrence",
            "draft_year": 2021,
            "draft_number": 1,
        },
        "00-0036972": {
            "name": "Mac Jones",
            "draft_year": 2021,
            "draft_number": 15,
        },
        "00-0037012": {
            "name": "Trey Lance",
            "draft_year": 2021,
            "draft_number": 3,
        },
        "00-0037013": {
            "name": "Zach Wilson",
            "draft_year": 2021,
            "draft_number": 2,
        },
        "00-0037077": {
            "name": "Sam Howell",
            "draft_year": 2022,
            "draft_number": 144,
        },
        "00-0037175": {
            "name": "Anthony Brown",
            "draft_year": 2022,
            "draft_number": 263,
        },
        "00-0037324": {
            "name": "Chris Oladokun",
            "draft_year": 2022,
            "draft_number": 241,
        },
        "00-0037327": {
            "name": "Skylar Thompson",
            "draft_year": 2022,
            "draft_number": 247,
        },
        "00-0037360": {
            "name": "Davis Cheek",
            "draft_year": 2022,
            "draft_number": 263,
        },
        "00-0037834": {
            "name": "Brock Purdy",
            "draft_year": 2022,
            "draft_number": 262,
        },
        "00-0038102": {
            "name": "Kenny Pickett",
            "draft_year": 2022,
            "draft_number": 20,
        },
        "00-0038108": {
            "name": "Bailey Zappe",
            "draft_year": 2022,
            "draft_number": 137,
        },
        "00-0038122": {
            "name": "Desmond Ridder",
            "draft_year": 2022,
            "draft_number": 74,
        },
        "00-0038128": {
            "name": "Malik Willis",
            "draft_year": 2022,
            "draft_number": 86,
        },
        "00-0038150": {
            "name": "Nathan Rourke",
            "draft_year": 2020,
            "draft_number": 263,
        },
        "00-0038391": {
            "name": "Sean Clifford",
            "draft_year": 2023,
            "draft_number": 149,
        },
        "00-0038400": {
            "name": "Tanner McKee",
            "draft_year": 2023,
            "draft_number": 188,
        },
        "00-0038416": {
            "name": "Tyson Bagent",
            "draft_year": 2023,
            "draft_number": 263,
        },
        "00-0038476": {
            "name": "Tommy DeVito",
            "draft_year": 2023,
            "draft_number": 263,
        },
        "00-0038550": {
            "name": "Hendon Hooker",
            "draft_year": 2023,
            "draft_number": 68,
        },
        "00-0038579": {
            "name": "Aidan O'Connell",
            "draft_year": 2023,
            "draft_number": 135,
        },
        "00-0038582": {
            "name": "Clayton Tune",
            "draft_year": 2023,
            "draft_number": 139,
        },
        "00-0038583": {
            "name": "Dorian Thompson-Robinson",
            "draft_year": 2023,
            "draft_number": 140,
        },
        "00-0038598": {
            "name": "Jaren Hall",
            "draft_year": 2023,
            "draft_number": 164,
        },
        "00-0038637": {
            "name": "Max Duggan",
            "draft_year": 2023,
            "draft_number": 239,
        },
        "00-0038911": {
            "name": "Malik Cunningham",
            "draft_year": 2023,
            "draft_number": 263,
        },
        "00-0038998": {
            "name": "Jake Haener",
            "draft_year": 2023,
            "draft_number": 127,
        },
        "00-0039107": {
            "name": "Stetson Bennett",
            "draft_year": 2023,
            "draft_number": 128,
        },
        "00-0039150": {
            "name": "Bryce Young",
            "draft_year": 2023,
            "draft_number": 1,
        },
        "00-0039152": {
            "name": "Will Levis",
            "draft_year": 2023,
            "draft_number": 33,
        },
        "00-0039163": {
            "name": "C.J. Stroud",
            "draft_year": 2023,
            "draft_number": 2,
        },
        "00-0039164": {
            "name": "Anthony Richardson",
            "draft_year": 2023,
            "draft_number": 4,
        },
    }
