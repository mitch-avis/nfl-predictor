"""
This module defines constants for the NFL predictor project. It includes configurations such as
paths, NFL season details, team abbreviations, and various statistics relevant to player and team
performance analysis. By centralizing these configurations, the module aids in maintaining
consistency and integrity across the project, making it easier to manage and update.

Key Constants:
- Directories: ROOT_DIR, DATA_PATH for project root and data storage.
- NFL Season: SEASON_END_MONTH, WEEKS_BEFORE_2021, WEEKS_FROM_2021_ONWARDS for season
  configurations.
- Data URLs: ELO_DATA_URL for fetching ELO ratings.
- Team Abbreviations: TEAMS, PBP_TEAMS, ELO_TEAMS, STD_TEAMS for different dataset formats.
- Statistics: BASE_COLUMNS, BOXSCORE_STATS, AGG_STATS for analysis metrics.
- Exclusions: ELO_DROP_COLS for columns to exclude from the ELO dataset.
- Player Information: ACTIVE_QB_IDS for active quarterbacks' names, draft years, and numbers.

This structured approach ensures easy reference and modification, supporting the project's
scalability and adaptability.
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

# URL for ELO ratings data
ELO_DATA_URL = "https://github.com/greerreNFL/nfeloqb/raw/main/qb_elos.csv"

# Team abbreviations in various formats for compatibility across datasets
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
    "LAC",
    "LAR",
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
]
ELO_TEAMS = [
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
    "WSH",
]
STD_TEAMS = [
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
]

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
AGG_DROP_COLS = [
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

# Columns to exclude from the ELO dataset for streamlined analysis
ELO_DROP_COLS = [
    "season",
    "neutral",
    "playoff",
    "elo_prob1",
    "elo_prob2",
    "elo1_post",
    "elo2_post",
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

# Active quarterback IDs: names, draft years, and numbers for player tracking
ACTIVE_QB_IDS = {
    "00-0019596": {"name": "Tom Brady", "draft_year": 2000, "draft_number": 199},
    "00-0020531": {"name": "Drew Brees", "draft_year": 2001, "draft_number": 32},
    "00-0020679": {"name": "Shaun Hill", "draft_year": 2002, "draft_number": 263},
    "00-0021206": {"name": "Josh McCown", "draft_year": 2002, "draft_number": 81},
    "00-0021429": {"name": "Carson Palmer", "draft_year": 2003, "draft_number": 1},
    "00-0022787": {"name": "Matt Schaub", "draft_year": 2004, "draft_number": 90},
    "00-0022803": {"name": "Eli Manning", "draft_year": 2004, "draft_number": 1},
    "00-0022924": {"name": "Ben Roethlisberger", "draft_year": 2004, "draft_number": 11},
    "00-0022942": {"name": "Philip Rivers", "draft_year": 2004, "draft_number": 4},
    "00-0023436": {"name": "Alex Smith", "draft_year": 2005, "draft_number": 1},
    "00-0023459": {"name": "Aaron Rodgers", "draft_year": 2005, "draft_number": 24},
    "00-0023645": {"name": "Derek Anderson", "draft_year": 2005, "draft_number": 213},
    "00-0023662": {"name": "Matt Cassel", "draft_year": 2005, "draft_number": 230},
    "00-0023682": {"name": "Ryan Fitzpatrick", "draft_year": 2005, "draft_number": 250},
    "00-0024226": {"name": "Jay Cutler", "draft_year": 2006, "draft_number": 11},
    "00-0025430": {"name": "Drew Stanton", "draft_year": 2007, "draft_number": 43},
    "00-0025708": {"name": "Matt Moore", "draft_year": 2007, "draft_number": 263},
    "00-0026143": {"name": "Matt Ryan", "draft_year": 2008, "draft_number": 3},
    "00-0026158": {"name": "Joe Flacco", "draft_year": 2008, "draft_number": 18},
    "00-0026197": {"name": "Chad Henne", "draft_year": 2008, "draft_number": 57},
    "00-0026300": {"name": "Josh Johnson", "draft_year": 2008, "draft_number": 160},
    "00-0026498": {"name": "Matthew Stafford", "draft_year": 2009, "draft_number": 1},
    "00-0026544": {"name": "Chase Daniel", "draft_year": 2009, "draft_number": 263},
    "00-0026625": {"name": "Brian Hoyer", "draft_year": 2009, "draft_number": 263},
    "00-0026898": {"name": "Mark Sanchez", "draft_year": 2009, "draft_number": 5},
    "00-0027688": {"name": "Colt McCoy", "draft_year": 2010, "draft_number": 85},
    "00-0027854": {"name": "Sam Bradford", "draft_year": 2010, "draft_number": 1},
    "00-0027939": {"name": "Cam Newton", "draft_year": 2011, "draft_number": 1},
    "00-0027948": {"name": "Blaine Gabbert", "draft_year": 2011, "draft_number": 10},
    "00-0027973": {"name": "Andy Dalton", "draft_year": 2011, "draft_number": 35},
    "00-0027974": {"name": "Colin Kaepernick", "draft_year": 2011, "draft_number": 36},
    "00-0028090": {"name": "T.J. Yates", "draft_year": 2011, "draft_number": 152},
    "00-0028118": {"name": "Tyrod Taylor", "draft_year": 2011, "draft_number": 180},
    "00-0028595": {"name": "Scott Tolzien", "draft_year": 2011, "draft_number": 263},
    "00-0028986": {"name": "Case Keenum", "draft_year": 2012, "draft_number": 263},
    "00-0029263": {"name": "Russell Wilson", "draft_year": 2012, "draft_number": 75},
    "00-0029567": {"name": "Nick Foles", "draft_year": 2012, "draft_number": 88},
    "00-0029604": {"name": "Kirk Cousins", "draft_year": 2012, "draft_number": 102},
    "00-0029665": {"name": "Robert Griffin", "draft_year": 2012, "draft_number": 2},
    "00-0029668": {"name": "Andrew Luck", "draft_year": 2012, "draft_number": 1},
    "00-0029682": {"name": "Brock Osweiler", "draft_year": 2012, "draft_number": 57},
    "00-0029701": {"name": "Ryan Tannehill", "draft_year": 2012, "draft_number": 8},
    "00-0030419": {"name": "Matt McGloin", "draft_year": 2013, "draft_number": 263},
    "00-0030520": {"name": "Mike Glennon", "draft_year": 2013, "draft_number": 73},
    "00-0030524": {"name": "Landry Jones", "draft_year": 2013, "draft_number": 115},
    "00-0030526": {"name": "EJ Manuel", "draft_year": 2013, "draft_number": 16},
    "00-0030533": {"name": "Matt Barkley", "draft_year": 2013, "draft_number": 98},
    "00-0030565": {"name": "Geno Smith", "draft_year": 2013, "draft_number": 39},
    "00-0031064": {"name": "Tom Savage", "draft_year": 2014, "draft_number": 135},
    "00-0031237": {"name": "Teddy Bridgewater", "draft_year": 2014, "draft_number": 32},
    "00-0031260": {"name": "Logan Thomas", "draft_year": 2014, "draft_number": 120},
    "00-0031280": {"name": "Derek Carr", "draft_year": 2014, "draft_number": 36},
    "00-0031288": {"name": "A.J. McCarron", "draft_year": 2014, "draft_number": 164},
    "00-0031345": {"name": "Jimmy Garoppolo", "draft_year": 2014, "draft_number": 62},
    "00-0031395": {"name": "Garrett Gilbert", "draft_year": 2014, "draft_number": 263},
    "00-0031407": {"name": "Blake Bortles", "draft_year": 2014, "draft_number": 3},
    "00-0031503": {"name": "Jameis Winston", "draft_year": 2015, "draft_number": 1},
    "00-0031568": {"name": "Bryce Petty", "draft_year": 2015, "draft_number": 103},
    "00-0031589": {"name": "Brett Hundley", "draft_year": 2015, "draft_number": 147},
    "00-0031800": {"name": "Taylor Heinicke", "draft_year": 2015, "draft_number": 263},
    "00-0032156": {"name": "Trevor Siemian", "draft_year": 2015, "draft_number": 250},
    "00-0032245": {"name": "Sean Mannion", "draft_year": 2015, "draft_number": 89},
    "00-0032268": {"name": "Marcus Mariota", "draft_year": 2015, "draft_number": 2},
    "00-0032434": {"name": "Brandon Allen", "draft_year": 2016, "draft_number": 263},
    "00-0032436": {"name": "Jeff Driskel", "draft_year": 2016, "draft_number": 207},
    "00-0032446": {"name": "Brandon Doughty", "draft_year": 2016, "draft_number": 223},
    "00-0032462": {"name": "Trevone Boykin", "draft_year": 2016, "draft_number": 263},
    "00-0032614": {"name": "Joel Stave", "draft_year": 2016, "draft_number": 263},
    "00-0032630": {"name": "Joe Callahan", "draft_year": 2016, "draft_number": 263},
    "00-0032784": {"name": "Kevin Hogan", "draft_year": 2016, "draft_number": 263},
    "00-0032792": {"name": "Nate Sudfeld", "draft_year": 2016, "draft_number": 187},
    "00-0032893": {"name": "Connor Cook", "draft_year": 2016, "draft_number": 263},
    "00-0032950": {"name": "Carson Wentz", "draft_year": 2016, "draft_number": 2},
    "00-0033077": {"name": "Dak Prescott", "draft_year": 2016, "draft_number": 135},
    "00-0033104": {"name": "Cody Kessler", "draft_year": 2016, "draft_number": 263},
    "00-0033106": {"name": "Jared Goff", "draft_year": 2016, "draft_number": 1},
    "00-0033108": {"name": "Paxton Lynch", "draft_year": 2016, "draft_number": 26},
    "00-0033119": {"name": "Jacoby Brissett", "draft_year": 2016, "draft_number": 91},
    "00-0033238": {"name": "Alek Torgersen", "draft_year": 2017, "draft_number": 263},
    "00-0033275": {"name": "P.J. Walker", "draft_year": 2017, "draft_number": 263},
    "00-0033319": {"name": "Nick Mullens", "draft_year": 2017, "draft_number": 263},
    "00-0033357": {"name": "Taysom Hill", "draft_year": 2017, "draft_number": 263},
    "00-0033537": {"name": "Deshaun Watson", "draft_year": 2017, "draft_number": 12},
    "00-0033550": {"name": "Davis Webb", "draft_year": 2017, "draft_number": 87},
    "00-0033662": {"name": "Cooper Rush", "draft_year": 2017, "draft_number": 263},
    "00-0033869": {"name": "Mitchell Trubisky", "draft_year": 2017, "draft_number": 2},
    "00-0033873": {"name": "Patrick Mahomes", "draft_year": 2017, "draft_number": 10},
    "00-0033899": {"name": "DeShone Kizer", "draft_year": 2017, "draft_number": 52},
    "00-0033936": {"name": "C.J. Beathard", "draft_year": 2017, "draft_number": 104},
    "00-0033949": {"name": "Joshua Dobbs", "draft_year": 2017, "draft_number": 135},
    "00-0033958": {"name": "Nathan Peterman", "draft_year": 2017, "draft_number": 171},
    "00-0034126": {"name": "J.T. Barrett", "draft_year": 2018, "draft_number": 263},
    "00-0034177": {"name": "Tim Boyle", "draft_year": 2018, "draft_number": 263},
    "00-0034343": {"name": "Josh Rosen", "draft_year": 2018, "draft_number": 10},
    "00-0034401": {"name": "Mike White", "draft_year": 2018, "draft_number": 171},
    "00-0034412": {"name": "Luke Falk", "draft_year": 2018, "draft_number": 263},
    "00-0034438": {"name": "Logan Woodside", "draft_year": 2018, "draft_number": 263},
    "00-0034478": {"name": "Chad Kanoff", "draft_year": 2018, "draft_number": 263},
    "00-0034577": {"name": "Kyle Allen", "draft_year": 2018, "draft_number": 263},
    "00-0034732": {"name": "Alex McGough", "draft_year": 2018, "draft_number": 220},
    "00-0034757": {"name": "Nick Stevens", "draft_year": 2018, "draft_number": 263},
    "00-0034771": {"name": "Mason Rudolph", "draft_year": 2018, "draft_number": 76},
    "00-0034796": {"name": "Lamar Jackson", "draft_year": 2018, "draft_number": 32},
    "00-0034855": {"name": "Baker Mayfield", "draft_year": 2018, "draft_number": 1},
    "00-0034857": {"name": "Josh Allen", "draft_year": 2018, "draft_number": 7},
    "00-0034869": {"name": "Sam Darnold", "draft_year": 2018, "draft_number": 3},
    "00-0034899": {"name": "John Wolford", "draft_year": 2018, "draft_number": 263},
    "00-0034955": {"name": "Brett Rypien", "draft_year": 2019, "draft_number": 263},
    "00-0035040": {"name": "David Blough", "draft_year": 2019, "draft_number": 263},
    "00-0035077": {"name": "Manny Wilkins", "draft_year": 2019, "draft_number": 263},
    "00-0035100": {"name": "Jake Browning", "draft_year": 2019, "draft_number": 263},
    "00-0035146": {"name": "Trace McSorley", "draft_year": 2019, "draft_number": 197},
    "00-0035228": {"name": "Kyler Murray", "draft_year": 2019, "draft_number": 1},
    "00-0035232": {"name": "Dwayne Haskins", "draft_year": 2019, "draft_number": 15},
    "00-0035251": {"name": "Will Grier", "draft_year": 2019, "draft_number": 100},
    "00-0035264": {"name": "Jarrett Stidham", "draft_year": 2019, "draft_number": 133},
    "00-0035282": {"name": "Easton Stick", "draft_year": 2019, "draft_number": 166},
    "00-0035283": {"name": "Clayton Thorson", "draft_year": 2019, "draft_number": 167},
    "00-0035289": {"name": "Gardner Minshew", "draft_year": 2019, "draft_number": 178},
    "00-0035483": {"name": "Drew Anderson", "draft_year": 2019, "draft_number": 263},
    "00-0035577": {"name": "Devlin Hodges", "draft_year": 2019, "draft_number": 263},
    "00-0035652": {"name": "Ryan Finley", "draft_year": 2019, "draft_number": 104},
    "00-0035704": {"name": "Drew Lock", "draft_year": 2019, "draft_number": 42},
    "00-0035710": {"name": "Daniel Jones", "draft_year": 2019, "draft_number": 6},
    "00-0035812": {"name": "Case Cookus", "draft_year": 2020, "draft_number": 263},
    "00-0035864": {"name": "Kendall Hinton", "draft_year": 2020, "draft_number": 263},
    "00-0035937": {"name": "Josh Love", "draft_year": 2020, "draft_number": 263},
    "00-0035939": {"name": "Bryce Perkins", "draft_year": 2020, "draft_number": 263},
    "00-0035968": {"name": "Jalen Morton", "draft_year": 2020, "draft_number": 263},
    "00-0035988": {"name": "Anthony Gordon", "draft_year": 2020, "draft_number": 263},
    "00-0035993": {"name": "Tyler Huntley", "draft_year": 2020, "draft_number": 263},
    "00-0036022": {"name": "Steven Montez", "draft_year": 2020, "draft_number": 263},
    "00-0036052": {"name": "Reid Sinnett", "draft_year": 2020, "draft_number": 263},
    "00-0036092": {"name": "Brian Lewerke", "draft_year": 2020, "draft_number": 263},
    "00-0036197": {"name": "Jake Fromm", "draft_year": 2020, "draft_number": 167},
    "00-0036212": {"name": "Tua Tagovailoa", "draft_year": 2020, "draft_number": 5},
    "00-0036257": {"name": "Cole McDonald", "draft_year": 2020, "draft_number": 224},
    "00-0036264": {"name": "Jordan Love", "draft_year": 2020, "draft_number": 26},
    "00-0036312": {"name": "Jake Luton", "draft_year": 2020, "draft_number": 189},
    "00-0036355": {"name": "Justin Herbert", "draft_year": 2020, "draft_number": 6},
    "00-0036384": {"name": "Ben DiNucci", "draft_year": 2020, "draft_number": 231},
    "00-0036389": {"name": "Jalen Hurts", "draft_year": 2020, "draft_number": 53},
    "00-0036442": {"name": "Joe Burrow", "draft_year": 2020, "draft_number": 1},
    "00-0036468": {"name": "Kai Locksley", "draft_year": 2021, "draft_number": 263},
    "00-0036679": {"name": "Shane Buechele", "draft_year": 2021, "draft_number": 263},
    "00-0036825": {"name": "Feleipe Franks", "draft_year": 2021, "draft_number": 263},
    "00-0036879": {"name": "Sam Ehlinger", "draft_year": 2021, "draft_number": 218},
    "00-0036898": {"name": "Davis Mills", "draft_year": 2021, "draft_number": 67},
    "00-0036928": {"name": "Kyle Trask", "draft_year": 2021, "draft_number": 64},
    "00-0036929": {"name": "Ian Book", "draft_year": 2021, "draft_number": 133},
    "00-0036945": {"name": "Justin Fields", "draft_year": 2021, "draft_number": 11},
    "00-0036971": {"name": "Trevor Lawrence", "draft_year": 2021, "draft_number": 1},
    "00-0036972": {"name": "Mac Jones", "draft_year": 2021, "draft_number": 15},
    "00-0037012": {"name": "Trey Lance", "draft_year": 2021, "draft_number": 3},
    "00-0037013": {"name": "Zach Wilson", "draft_year": 2021, "draft_number": 2},
    "00-0037077": {"name": "Sam Howell", "draft_year": 2022, "draft_number": 144},
    "00-0037175": {"name": "Anthony Brown", "draft_year": 2022, "draft_number": 263},
    "00-0037324": {"name": "Chris Oladokun", "draft_year": 2022, "draft_number": 241},
    "00-0037327": {"name": "Skylar Thompson", "draft_year": 2022, "draft_number": 247},
    "00-0037360": {"name": "Davis Cheek", "draft_year": 2022, "draft_number": 263},
    "00-0037834": {"name": "Brock Purdy", "draft_year": 2022, "draft_number": 262},
    "00-0038102": {"name": "Kenny Pickett", "draft_year": 2022, "draft_number": 20},
    "00-0038108": {"name": "Bailey Zappe", "draft_year": 2022, "draft_number": 137},
    "00-0038122": {"name": "Desmond Ridder", "draft_year": 2022, "draft_number": 74},
    "00-0038128": {"name": "Malik Willis", "draft_year": 2022, "draft_number": 86},
    "00-0038150": {"name": "Nathan Rourke", "draft_year": 2020, "draft_number": 263},
    "00-0038391": {"name": "Sean Clifford", "draft_year": 2023, "draft_number": 149},
    "00-0038400": {"name": "Tanner McKee", "draft_year": 2023, "draft_number": 188},
    "00-0038416": {"name": "Tyson Bagent", "draft_year": 2023, "draft_number": 263},
    "00-0038476": {"name": "Tommy DeVito", "draft_year": 2023, "draft_number": 263},
    "00-0038550": {"name": "Hendon Hooker", "draft_year": 2023, "draft_number": 68},
    "00-0038579": {"name": "Aidan O'Connell", "draft_year": 2023, "draft_number": 135},
    "00-0038582": {"name": "Clayton Tune", "draft_year": 2023, "draft_number": 139},
    "00-0038583": {"name": "Dorian Thompson-Robinson", "draft_year": 2023, "draft_number": 140},
    "00-0038598": {"name": "Jaren Hall", "draft_year": 2023, "draft_number": 164},
    "00-0038637": {"name": "Max Duggan", "draft_year": 2023, "draft_number": 239},
    "00-0038911": {"name": "Malik Cunningham", "draft_year": 2023, "draft_number": 263},
    "00-0038998": {"name": "Jake Haener", "draft_year": 2023, "draft_number": 127},
    "00-0039107": {"name": "Stetson Bennett", "draft_year": 2023, "draft_number": 128},
    "00-0039150": {"name": "Bryce Young", "draft_year": 2023, "draft_number": 1},
    "00-0039152": {"name": "Will Levis", "draft_year": 2023, "draft_number": 33},
    "00-0039163": {"name": "C.J. Stroud", "draft_year": 2023, "draft_number": 2},
    "00-0039164": {"name": "Anthony Richardson", "draft_year": 2023, "draft_number": 4},
}
