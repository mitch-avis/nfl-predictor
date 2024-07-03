from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sportsipy.nfl.boxscore import Boxscore, Boxscores

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.utils import read_write_data

STARTING_SEASON = 2023
NUM_SEASONS = 1
CURRENT_WEEK = 1

REFRESH_ELO = False
REFRESH_SEASON = False
REFRESH_WEEKS = False
REFRESH_SCHEDULE = False

ELO_DATA_URL = constants.ELO_DATA_URL
AWAY_STATS = constants.AWAY_STATS
AWAY_STATS_RENAME = constants.AWAY_STATS_RENAME
HOME_STATS = constants.HOME_STATS
HOME_STATS_RENAME = constants.HOME_STATS_RENAME
AGG_RENAME_AWAY = constants.AGG_RENAME_AWAY
AGG_RENAME_HOME = constants.AGG_RENAME_HOME
AGG_MERGE_ON = constants.AGG_MERGE_ON
AGG_DROP_COLS = constants.AGG_DROP_COLS
ELO_DROP_COLS = constants.ELO_DROP_COLS
ELO_TEAMS = constants.ELO_TEAMS
STD_TEAMS = constants.STD_TEAMS
TEAMS = constants.TEAMS


def main() -> None:
    # Get and save latest ELO spreadsheet
    elo_df_name = "nfl_elo"
    elo_df = read_write_data(
        elo_df_name,
        get_elo,
        force_refresh=REFRESH_ELO,
    )
    # Create Date objects for start and end dates, using set number of seasons
    start_date = date(STARTING_SEASON, 9, 1)
    end_date = start_date + relativedelta(years=NUM_SEASONS - 1) + relativedelta(months=6)
    start_season = start_date.year
    end_season = end_date.year
    today = date.today()
    current_season = today.year - 1 if today.month <= 3 else today.year
    log.info(
        "Collecting NFL data for the following [%s] season(s): %s",
        end_season - start_season,
        list(range(start_season, end_season)),
    )
    for season in range(start_season, end_season):
        # Create list of weeks to scrape; use all weeks for past seasons, or only up to the current
        # week if scraping the current season
        if season < current_season:
            if season < 2021:
                weeks = list(range(1, 18))
            else:
                weeks = list(range(1, 19))
        else:
            weeks = list(range(1, CURRENT_WEEK + 1))
        # Get and save season's game data
        log.info("Collecting game data for the %s season...", season)
        season_games_df_name = f"{season}/season_games"
        season_games_df = read_write_data(
            season_games_df_name,
            get_season_data,
            season,
            weeks,
            force_refresh=REFRESH_SEASON,
        )
        # Get and save aggregate game data
        agg_games_df_name = f"{season}/agg_games"
        agg_games_df = read_write_data(
            agg_games_df_name,
            agg_weekly_data,
            season_games_df,
            season,
            weeks,
            force_refresh=True,
        )
        # Get and save year's ELO ratings
        yearly_elo_df_name = f"{season}/elo"
        yearly_elo_df = read_write_data(
            yearly_elo_df_name,
            get_yearly_elo,
            elo_df,
            season,
            force_refresh=True,
        )
        # Get and save combined data
        combined_data_df_name = f"{season}/combined_data"
        combined_data_df = read_write_data(
            combined_data_df_name,
            combine_data,
            agg_games_df,
            yearly_elo_df,
            force_refresh=True,
        )
        # Get and save completed games
        comp_games_df_name = f"{season}/completed_games"
        read_write_data(
            comp_games_df_name,
            parse_completed_games,
            combined_data_df,
            force_refresh=True,
        )
        # # Get and save games to predict
        # pred_games_df_name = f"{season}/week_{CURRENT_WEEK:>02}_games_to_predict"
        # read_write_data(
        #     pred_games_df_name,
        #     parse_games_to_predict,
        #     combined_data_df,
        #     force_refresh=True,
        # )


def get_elo() -> pd.DataFrame:
    elo_df = pd.read_csv(ELO_DATA_URL)
    return elo_df


def get_season_data(year: int, weeks: list) -> pd.DataFrame:
    season_games_df = pd.DataFrame()
    # Skip scraping game data if it's a new season
    if len(weeks) == 1:
        return season_games_df
    # Step through each week of current season
    log.info("Scraping game data for %s...", year)
    for week in weeks:
        log.info("Collecting game data for Week %s...", week)
        # Get and save week's game data
        week_games_name = f"{year}/week_{week:>02}_game_data"
        week_games_df = read_write_data(
            week_games_name,
            get_game_data_by_week,
            year,
            week,
            force_refresh=REFRESH_WEEKS,
        )
        # Concatenate week's game data into season
        season_games_df = pd.concat([season_games_df, week_games_df])
    return season_games_df


def get_game_data_by_week(year: int, week: int) -> pd.DataFrame:
    date_string = f"{week}-{year}"
    week_scores = Boxscores(week, year)
    week_games_df = pd.DataFrame()
    log.info("Scraping game data for Week %s...", week)
    for game in range(len(week_scores.games[date_string])):
        game_str = week_scores.games[date_string][game]["boxscore"]
        log.info("Scraping game data for %s...", game_str)
        game_stats = Boxscore(game_str)
        game_df = pd.DataFrame(week_scores.games[date_string][game], index=[0])
        away_team_df, home_team_df = parse_game_data(game_df, game_stats)
        away_team_df["week"] = week
        home_team_df["week"] = week
        away_team_df["year"] = year
        home_team_df["year"] = year
        week_games_df = pd.concat([week_games_df, away_team_df], ignore_index=True)
        week_games_df = pd.concat([week_games_df, home_team_df], ignore_index=True)
    return week_games_df


def parse_game_data(
    game_df: pd.DataFrame, game_stats: Boxscore
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        away_team_df = game_df[["away_name", "away_abbr", "away_score"]].rename(
            columns={
                "away_name": "team_name",
                "away_abbr": "team_abbr",
                "away_score": "score",
            }
        )
        home_team_df = game_df[["home_name", "home_abbr", "home_score"]].rename(
            columns={
                "home_name": "team_name",
                "home_abbr": "team_abbr",
                "home_score": "score",
            }
        )
        try:
            # Away team won
            if game_df.loc[0, "away_score"] > game_df.loc[0, "home_score"]:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [float(1.0)], "game_lost": [float(0.0)]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [float(0.0)], "game_lost": [float(1.0)]}),
                    left_index=True,
                    right_index=True,
                )
            # Home team won
            elif game_df.loc[0, "away_score"] < game_df.loc[0, "home_score"]:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [float(0.0)], "game_lost": [float(1.0)]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [float(1.0)], "game_lost": [float(0.0)]}),
                    left_index=True,
                    right_index=True,
                )
            # Tie game
            else:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [float(0.5)], "game_lost": [float(0.5)]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [float(0.5)], "game_lost": [float(0.5)]}),
                    left_index=True,
                    right_index=True,
                )
        except TypeError:
            away_team_df = pd.merge(
                away_team_df,
                pd.DataFrame({"game_won": [float(0.0)], "game_lost": [float(0.0)]}),
                left_index=True,
                right_index=True,
            )
            home_team_df = pd.merge(
                home_team_df,
                pd.DataFrame({"game_won": [float(0.0)], "game_lost": [float(0.0)]}),
                left_index=True,
                right_index=True,
            )
        away_stats_df = (
            game_stats.dataframe[AWAY_STATS]
            .reset_index()
            .drop(columns="index")
            .rename(columns=AWAY_STATS_RENAME)
        )
        home_stats_df = (
            game_stats.dataframe[HOME_STATS]
            .reset_index()
            .drop(columns="index")
            .rename(columns=HOME_STATS_RENAME)
        )
        away_team_df = pd.merge(away_team_df, away_stats_df, left_index=True, right_index=True)
        home_team_df = pd.merge(home_team_df, home_stats_df, left_index=True, right_index=True)
        try:
            # Convert ToP from mm:ss to total seconds
            away_team_df["time_of_possession"] = (
                int(away_team_df["time_of_possession"].loc[0][0:2]) * 60
            ) + int(away_team_df["time_of_possession"].loc[0][3:5])
            home_team_df["time_of_possession"] = (
                int(home_team_df["time_of_possession"].loc[0][0:2]) * 60
            ) + int(home_team_df["time_of_possession"].loc[0][3:5])
        except TypeError:
            away_team_df["time_of_possession"] = np.nan
            home_team_df["time_of_possession"] = np.nan
    except TypeError:
        away_team_df = pd.DataFrame()
        home_team_df = pd.DataFrame()
    return away_team_df, home_team_df


def agg_weekly_data(season_games_df: pd.DataFrame, year: int, weeks: list) -> pd.DataFrame:
    # Get and save season's schedule
    log.info("Collecting schedule for %s...", year)
    schedule_df_name = f"{year}/schedule"
    schedule_df = read_write_data(
        schedule_df_name,
        get_schedule,
        year,
        force_refresh=REFRESH_SCHEDULE,
    )
    # TODO: Create empty df for first week of new season and skip all this
    agg_games_df = pd.DataFrame()
    for week in weeks:
        games_df = schedule_df[schedule_df.week == week]
        agg_weekly_df = (
            season_games_df[season_games_df.week < week]
            .drop(columns=["score", "week", "year", "game_won", "game_lost"])
            .groupby(by=["team_name", "team_abbr"])
            .mean()
            .reset_index()
        )
        win_loss_df = (
            season_games_df[season_games_df.week < week][
                ["team_name", "team_abbr", "game_won", "game_lost"]
            ]
            .groupby(by=["team_name", "team_abbr"])
            .sum()
            .reset_index()
        )
        win_loss_df["win_perc"] = win_loss_df["game_won"].div(
            win_loss_df["game_won"] + win_loss_df["game_lost"]
        )
        win_loss_df.loc[~np.isfinite(win_loss_df["win_perc"]), "win_perc"] = 0
        win_loss_df = win_loss_df.drop(columns=["game_won", "game_lost"])
        agg_weekly_df["fourth_down_perc"] = agg_weekly_df["fourth_down_conversions"].div(
            agg_weekly_df["fourth_down_attempts"]
        )
        agg_weekly_df.loc[~np.isfinite(agg_weekly_df["fourth_down_perc"]), "fourth_down_perc"] = 0
        agg_weekly_df["fourth_down_perc"] = agg_weekly_df["fourth_down_perc"].fillna(0)
        agg_weekly_df["third_down_perc"] = agg_weekly_df["third_down_conversions"].div(
            agg_weekly_df["third_down_attempts"]
        )
        agg_weekly_df.loc[~np.isfinite(agg_weekly_df["third_down_perc"]), "third_down_perc"] = 0
        agg_weekly_df["third_down_perc"] = agg_weekly_df["third_down_perc"].fillna(0)
        agg_weekly_df = agg_weekly_df.drop(
            columns=[
                "fourth_down_attempts",
                "fourth_down_conversions",
                "third_down_attempts",
                "third_down_conversions",
            ]
        )
        agg_weekly_df = pd.merge(
            win_loss_df,
            agg_weekly_df,
            left_on=["team_name", "team_abbr"],
            right_on=["team_name", "team_abbr"],
        )
        away_df = (
            pd.merge(
                games_df,
                agg_weekly_df,
                how="inner",
                left_on=["away_name", "away_abbr"],
                right_on=["team_name", "team_abbr"],
            )
            .drop(columns=["team_name", "team_abbr"])
            .rename(columns=AGG_RENAME_AWAY)
        )
        home_df = (
            pd.merge(
                games_df,
                agg_weekly_df,
                how="inner",
                left_on=["home_name", "home_abbr"],
                right_on=["team_name", "team_abbr"],
            )
            .drop(columns=["team_name", "team_abbr"])
            .rename(columns=AGG_RENAME_HOME)
        )
        agg_weekly_df = pd.merge(away_df, home_df, left_on=AGG_MERGE_ON, right_on=AGG_MERGE_ON)
        agg_weekly_df["win_perc_dif"] = (
            agg_weekly_df["away_win_perc"] - agg_weekly_df["home_win_perc"]
        )
        agg_weekly_df["first_downs_dif"] = (
            agg_weekly_df["away_first_downs"] - agg_weekly_df["home_first_downs"]
        )
        agg_weekly_df["fumbles_dif"] = agg_weekly_df["away_fumbles"] - agg_weekly_df["home_fumbles"]
        agg_weekly_df["interceptions_dif"] = (
            agg_weekly_df["away_interceptions"] - agg_weekly_df["home_interceptions"]
        )
        agg_weekly_df["net_pass_yards_dif"] = (
            agg_weekly_df["away_net_pass_yards"] - agg_weekly_df["home_net_pass_yards"]
        )
        agg_weekly_df["pass_attempts_dif"] = (
            agg_weekly_df["away_pass_attempts"] - agg_weekly_df["home_pass_attempts"]
        )
        agg_weekly_df["pass_completions_dif"] = (
            agg_weekly_df["away_pass_completions"] - agg_weekly_df["home_pass_completions"]
        )
        agg_weekly_df["pass_touchdowns_dif"] = (
            agg_weekly_df["away_pass_touchdowns"] - agg_weekly_df["home_pass_touchdowns"]
        )
        agg_weekly_df["pass_yards_dif"] = (
            agg_weekly_df["away_pass_yards"] - agg_weekly_df["home_pass_yards"]
        )
        agg_weekly_df["penalties_dif"] = (
            agg_weekly_df["away_penalties"] - agg_weekly_df["home_penalties"]
        )
        agg_weekly_df["points_dif"] = agg_weekly_df["away_points"] - agg_weekly_df["home_points"]
        agg_weekly_df["rush_attempts_dif"] = (
            agg_weekly_df["away_rush_attempts"] - agg_weekly_df["home_rush_attempts"]
        )
        agg_weekly_df["rush_touchdowns_dif"] = (
            agg_weekly_df["away_rush_touchdowns"] - agg_weekly_df["home_rush_touchdowns"]
        )
        agg_weekly_df["rush_yards_dif"] = (
            agg_weekly_df["away_rush_yards"] - agg_weekly_df["home_rush_yards"]
        )
        agg_weekly_df["time_of_possession_dif"] = (
            agg_weekly_df["away_time_of_possession"] - agg_weekly_df["home_time_of_possession"]
        )
        agg_weekly_df["times_sacked_dif"] = (
            agg_weekly_df["away_times_sacked"] - agg_weekly_df["home_times_sacked"]
        )
        agg_weekly_df["total_yards_dif"] = (
            agg_weekly_df["away_total_yards"] - agg_weekly_df["home_total_yards"]
        )
        agg_weekly_df["turnovers_dif"] = (
            agg_weekly_df["away_turnovers"] - agg_weekly_df["home_turnovers"]
        )
        agg_weekly_df["yards_from_penalties_dif"] = (
            agg_weekly_df["away_yards_from_penalties"] - agg_weekly_df["home_yards_from_penalties"]
        )
        agg_weekly_df["yards_lost_from_sacks_dif"] = (
            agg_weekly_df["away_yards_lost_from_sacks"]
            - agg_weekly_df["home_yards_lost_from_sacks"]
        )
        agg_weekly_df["fourth_down_perc_dif"] = (
            agg_weekly_df["away_fourth_down_perc"] - agg_weekly_df["home_fourth_down_perc"]
        )
        agg_weekly_df["third_down_perc_dif"] = (
            agg_weekly_df["away_third_down_perc"] - agg_weekly_df["home_third_down_perc"]
        )
        agg_weekly_df = agg_weekly_df.drop(columns=AGG_DROP_COLS, errors="ignore")
        if agg_weekly_df["winning_name"].isna().values.any():
            if agg_weekly_df["winning_name"].isna().sum() == agg_weekly_df.shape[0]:
                agg_weekly_df["result"] = np.nan
                log.info("Week %s games have not finished yet.", week)
            else:
                agg_weekly_df.loc[agg_weekly_df["winning_name"].isna(), "result"] = 0
                agg_weekly_df["result"] = agg_weekly_df["result"].astype("float")
        else:
            agg_weekly_df["result"] = agg_weekly_df["winning_name"] == agg_weekly_df["away_name"]
            agg_weekly_df["result"] = agg_weekly_df["result"].astype("float")
        agg_weekly_df = agg_weekly_df.drop(columns=["winning_name", "winning_abbr"])
        agg_games_df = pd.concat([agg_games_df, agg_weekly_df])
    agg_games_df = agg_games_df.reset_index(drop=True)
    return agg_games_df


def get_schedule(year: int) -> pd.DataFrame:
    # Create list of weeks to scrape
    if year < 2021:
        weeks = list(range(1, 18))
    else:
        weeks = list(range(1, 19))
    schedule_df = pd.DataFrame()
    log.info("Scraping %s schedule...", year)
    for week in weeks:
        log.info("Week %s...", week)
        date_string = f"{week}-{year}"
        week_scores = Boxscores(week, year)
        week_games_df = pd.DataFrame()
        for game in range(len(week_scores.games[date_string])):
            game_df = pd.DataFrame(week_scores.games[date_string][game], index=[0])[
                [
                    "away_name",
                    "away_abbr",
                    "home_name",
                    "home_abbr",
                    "winning_name",
                    "winning_abbr",
                ]
            ]
            game_df["week"] = week
            week_games_df = pd.concat([week_games_df, game_df])
        schedule_df = pd.concat([schedule_df, week_games_df]).reset_index(drop=True)
    return schedule_df


def get_yearly_elo(elo_df: pd.DataFrame, year: int) -> pd.DataFrame:
    yearly_elo_df = elo_df.copy()
    yearly_elo_df = yearly_elo_df.drop(columns=ELO_DROP_COLS)
    yearly_elo_df["date"] = pd.to_datetime(yearly_elo_df["date"])
    start_date = datetime(year, 9, 1)
    end_date = datetime(year + 1, 3, 1)
    mask = (yearly_elo_df["date"] >= start_date) & (yearly_elo_df["date"] <= end_date)
    yearly_elo_df = yearly_elo_df.loc[mask]
    yearly_elo_df["team1"] = yearly_elo_df["team1"].replace(ELO_TEAMS, STD_TEAMS)
    yearly_elo_df["team2"] = yearly_elo_df["team2"].replace(ELO_TEAMS, STD_TEAMS)
    return yearly_elo_df


def combine_data(agg_games_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    agg_games_df = pd.merge(
        agg_games_df,
        elo_df,
        how="inner",
        left_on=["home_abbr", "away_abbr"],
        right_on=["team1", "team2"],
    ).drop(columns=["date", "team1", "team2"])
    agg_games_df["elo_dif"] = agg_games_df["elo2_pre"] - agg_games_df["elo1_pre"]
    agg_games_df["qb_dif"] = agg_games_df["qb2_value_pre"] - agg_games_df["qb1_value_pre"]
    agg_games_df = agg_games_df.drop(
        columns=["elo1_pre", "elo2_pre", "qb1_value_pre", "qb2_value_pre"]
    )
    # Set team abbreviations back to normal ones
    agg_games_df["home_abbr"] = agg_games_df["home_abbr"].replace(STD_TEAMS, TEAMS)
    agg_games_df["away_abbr"] = agg_games_df["away_abbr"].replace(STD_TEAMS, TEAMS)
    # Move "result" column to the end
    agg_games_df = agg_games_df[
        [col for col in agg_games_df if col not in ["result"]]
        + [col for col in ["result"] if col in agg_games_df]
    ]
    return agg_games_df


def parse_completed_games(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    comp_games_df = combined_data_df[combined_data_df["result"].notna()]
    return comp_games_df


def parse_games_to_predict(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    pred_games_df = pd.DataFrame(combined_data_df[combined_data_df["week"] == CURRENT_WEEK])
    return pred_games_df


if __name__ == "__main__":
    main()
