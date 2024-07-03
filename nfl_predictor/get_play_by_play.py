import pandas as pd

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.utils import nested_dict_to_df, read_write_data

START_YEAR = 2023
END_YEAR = 2023
ACTIVE_QB_IDS = constants.ACTIVE_QB_IDS
PBP_TEAMS = constants.PBP_TEAMS
TEAMS = constants.TEAMS


def clean_pbp_data(raw_pbp_df: pd.DataFrame):
    cleaned_pbp_df = raw_pbp_df.copy()
    del raw_pbp_df
    cleaned_pbp_df = cleaned_pbp_df.rename(columns={"id": "player_id2"})
    cleaned_pbp_df = cleaned_pbp_df.loc[
        (cleaned_pbp_df["play_type"].isin(["pass", "run", "no_play"]))
        & ~(cleaned_pbp_df["epa"].isna())
    ]
    cleaned_pbp_df.loc[cleaned_pbp_df["pass"] == 1, "play_type"] = "pass"
    cleaned_pbp_df.loc[cleaned_pbp_df["rush"] == 1, "play_type"] = "rush"
    cleaned_pbp_df.reset_index(drop=True, inplace=True)
    return cleaned_pbp_df


def get_season_pbp(year: int):
    season_pbp_df = pd.DataFrame()
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.csv"
    season_pbp_df = pd.read_csv(url, low_memory=False)
    season_pbp_df = clean_pbp_data(season_pbp_df)
    return season_pbp_df


def get_all_pbp_data():
    play_by_play_df = pd.DataFrame()
    for year in range(START_YEAR, END_YEAR + 1):
        season_pbp_df_name = f"{year}_pbp"
        season_pbp_df = read_write_data(season_pbp_df_name, get_season_pbp, year)
        play_by_play_df = pd.concat([play_by_play_df, season_pbp_df], sort=True)
        del season_pbp_df
    return play_by_play_df


def get_pbp_data():
    # play_by_play_df_name = "all_pbp"
    play_by_play_df_name = "2023_all_pbp"
    play_by_play_df = read_write_data(play_by_play_df_name, get_all_pbp_data)
    return play_by_play_df


def parse_qb_data(play_by_play_df: pd.DataFrame):
    qb_df = play_by_play_df.groupby(
        ["passer", "passer_id", "posteam", "season", "week"], as_index=False
    ).agg({"qb_epa": "mean", "cpoe": "mean", "qb_dropback": "count"})
    # Filter for active QBs only
    qb_df = qb_df[qb_df["passer_id"].isin(ACTIVE_QB_IDS.keys())]
    # log.debug("qb_df:\n%s", qb_df)
    qb_df = qb_df.set_index("passer_id")
    # log.debug("qb_df:\n%s", qb_df)
    active_qbs_df = nested_dict_to_df(ACTIVE_QB_IDS)
    merged_qb_df = qb_df.join(active_qbs_df)
    # log.debug("merged_qb_df:\n%s", merged_qb_df)
    # Rename columns
    merged_qb_df = merged_qb_df[
        [
            "name",
            "posteam",
            "draft_year",
            "draft_number",
            "season",
            "week",
            "qb_dropback",
            "qb_epa",
            "cpoe",
        ]
    ]
    merged_qb_df.columns = [
        "Player",
        "Team",
        "Draft Year",
        "Draft Number",
        "Season",
        "Week",
        "Dropbacks",
        "EPA/db",
        "CPOE",
    ]
    # Sort in descending order by EPA/dropback
    merged_qb_df.sort_values(
        by=["Season", "Week", "EPA/db"], ascending=[True, True, False], inplace=True
    )
    # log.debug("merged_qb_df:\n%s", merged_qb_df)
    return merged_qb_df


def parse_team_data(play_by_play_df: pd.DataFrame):
    play_by_play_df = play_by_play_df.loc[
        (play_by_play_df["play_type"].isin(["pass", "rush"]))
        & ~(play_by_play_df["posteam"].isna())
        & (play_by_play_df["season"] >= 2014)
    ]
    offense_epa_df = play_by_play_df.groupby(["posteam", "season", "week"], as_index=False)[
        ["epa"]
    ].mean()
    defense_epa_df = play_by_play_df.groupby(["defteam", "season", "week"], as_index=False)[
        ["epa"]
    ].mean()
    log.debug("offense_epa_df:\n%s", offense_epa_df)
    log.debug("defense_epa_df:\n%s", defense_epa_df)
    offense_epa_df.columns = [
        "Team",
        "Season",
        "Week",
        "Offense EPA",
    ]
    defense_epa_df.columns = [
        "Team",
        "Season",
        "Week",
        "Defense EPA",
    ]
    team_epa_df = offense_epa_df.merge(defense_epa_df)
    team_epa_df["Total EPA"] = team_epa_df["Offense EPA"] - team_epa_df["Defense EPA"]
    team_epa_df["Team"] = team_epa_df["Team"].replace(PBP_TEAMS, TEAMS)
    # Sort in descending order by Team Name, then Week, then Season
    team_epa_df.sort_values(
        by=["Season", "Week", "Team"], ascending=[True, True, True], inplace=True
    )
    team_epa_df = team_epa_df.reset_index(drop=True)
    log.debug("team_epa_df:\n%s", team_epa_df)
    return team_epa_df


def main():
    # Get play-by-play data
    play_by_play_df = get_pbp_data()
    # Parse weekly QB data
    # qb_df_name = "all_qbs"
    qb_df_name = "2023_qbs"
    qb_df = read_write_data(qb_df_name, parse_qb_data, play_by_play_df)
    log.debug("qb_df:\n%s", qb_df)
    # Parse weekly Team data
    # team_df_name = "all_teams"
    team_df_name = "2023_teams"
    team_df = read_write_data(team_df_name, parse_team_data, play_by_play_df)
    log.debug("team_df:\n%s", team_df)


if __name__ == "__main__":
    main()
