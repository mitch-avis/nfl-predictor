#!/bin/env python

import time
from datetime import datetime

import numpy as np
import pandas as pd
from sportsipy.nfl.boxscore import Boxscore, Boxscores

from definitions import (
    AGG_DROP_COLS,
    AGG_MERGE_ON,
    AGG_RENAME_AWAY,
    AGG_RENAME_HOME,
    AWAY_STATS,
    AWAY_STATS_DROP,
    ELO_DATA_URL,
    ELO_DROP_COLS,
    ELO_TEAMS,
    HOME_STATS,
    HOME_STATS_DROP,
    STD_TEAMS,
)
from utils.logger import log
from utils.utils import read_write_data


def main():
    start_date = "2022-09-08"
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = "2023-01-11"
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    current_week = 18
    collect_data(start_date_dt, end_date_dt, current_week)


def collect_data(start_date: datetime, end_date: datetime, current_week: int) -> None:
    # Get start year from start date
    start_year = int(start_date.year)
    # Get end year from end date
    if end_date.year == start_date.year + 1:
        end_year = int(end_date.year - 1)
    else:
        end_year = int(end_date.year)
    # Create list of weeks to scrape
    weeks = list(range(1, current_week + 1))
    for year in range(start_year, end_year + 1):
        log.info(f"Collecting game data for {year}...")
        # Get and save season's game data
        season_games_name = f"{year}_season_games"
        season_games_df = read_write_data(season_games_name, get_season_data, year, current_week)
        # Get and save season's schedule
        schedule_name = f"{year}_schedule"
        schedule_df = read_write_data(schedule_name, get_schedule, year)
        # Get and save aggregate game data
        agg_games_name = f"{year}_agg_games"
        agg_games_df = read_write_data(
            agg_games_name,
            agg_weekly_data,
            schedule_df,
            season_games_df,
            current_week,
            weeks,
        )
        # Get and save ELO ratings
        elo_name = f"{year}_elo"
        elo_df = read_write_data(elo_name, get_elo, start_date, end_date)
        # Get and save combined data
        combined_data_name = f"{year}_combined_data"
        combined_data_df = read_write_data(combined_data_name, combine_data, agg_games_df, elo_df)
        # Get and save completed games
        comp_games_name = f"{year}_comp_games"
        _ = read_write_data(comp_games_name, parse_completed_games, combined_data_df)
        # Get and save games to predict
        pred_games_name = f"{year}_{current_week}_pred_games"
        _ = read_write_data(pred_games_name, parse_games_to_predict, combined_data_df, current_week)


def get_season_data(year: int, current_week: int) -> pd.DataFrame:
    season_games_df = pd.DataFrame()
    # Step through each week of current season
    for week in range(1, current_week + 1):
        log.info(f"Scraping Week {week} game data...")
        # Get and save week's game data
        week_games_name = f"{year}_{week}"
        week_games_df = read_write_data(week_games_name, get_game_data_by_week, year, week)
        # Concatenate week's game data into season
        season_games_df = pd.concat([season_games_df, week_games_df])
    return season_games_df


def get_game_data_by_week(year: int, week: int) -> pd.DataFrame:
    date_string = f"{week}-{year}"
    week_scores = Boxscores(week, year)
    time.sleep(3)
    week_games_df = pd.DataFrame()
    for game in range(len(week_scores.games[date_string])):
        game_str = week_scores.games[date_string][game]["boxscore"]
        game_stats = Boxscore(game_str)
        time.sleep(3)
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
            if game_df.loc[0, "away_score"] > game_df.loc[0, "home_score"]:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [1], "game_lost": [0]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [0], "game_lost": [1]}),
                    left_index=True,
                    right_index=True,
                )
            elif game_df.loc[0, "away_score"] < game_df.loc[0, "home_score"]:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [0], "game_lost": [1]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [1], "game_lost": [0]}),
                    left_index=True,
                    right_index=True,
                )
            else:
                away_team_df = pd.merge(
                    away_team_df,
                    pd.DataFrame({"game_won": [0], "game_lost": [0]}),
                    left_index=True,
                    right_index=True,
                )
                home_team_df = pd.merge(
                    home_team_df,
                    pd.DataFrame({"game_won": [0], "game_lost": [0]}),
                    left_index=True,
                    right_index=True,
                )
        except TypeError:
            away_team_df = pd.merge(
                away_team_df,
                pd.DataFrame({"game_won": [np.nan], "game_lost": [np.nan]}),
                left_index=True,
                right_index=True,
            )
            home_team_df = pd.merge(
                home_team_df,
                pd.DataFrame({"game_won": [np.nan], "game_lost": [np.nan]}),
                left_index=True,
                right_index=True,
            )
        away_stats_df = (
            game_stats.dataframe[AWAY_STATS]
            .reset_index()
            .drop(columns="index")
            .rename(columns=AWAY_STATS_DROP)
        )
        home_stats_df = (
            game_stats.dataframe[HOME_STATS]
            .reset_index()
            .drop(columns="index")
            .rename(columns=HOME_STATS_DROP)
        )
        away_team_df = pd.merge(away_team_df, away_stats_df, left_index=True, right_index=True)
        home_team_df = pd.merge(home_team_df, home_stats_df, left_index=True, right_index=True)
        try:
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


def get_schedule(year: int) -> pd.DataFrame:
    weeks = list(range(1, 19))
    schedule_df = pd.DataFrame()
    log.info(f"Getting {year} schedule...")
    for week in weeks:
        log.info(f"Week {week}...")
        date_string = f"{week}-{year}"
        week_scores = Boxscores(week, year)
        time.sleep(3)
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


def agg_weekly_data(
    schedule_df: pd.DataFrame, season_games_df: pd.DataFrame, current_week: int, weeks: list
) -> pd.DataFrame:
    schedule_df = schedule_df[schedule_df.week < current_week + 1]
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
        agg_weekly_df = agg_weekly_df.drop(columns=AGG_DROP_COLS)
        if agg_weekly_df["winning_name"].isna().values.any():
            if agg_weekly_df["winning_name"].isna().sum() == agg_weekly_df.shape[0]:
                agg_weekly_df["result"] = np.nan
                log.info(f"Week {week} games have not finished yet.")
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


def get_elo(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    elo_df = pd.read_csv(ELO_DATA_URL)
    # elo_df = pd.read_csv("./utils/nfl_elo.csv")
    elo_df = elo_df.drop(columns=ELO_DROP_COLS)
    elo_df["date"] = pd.to_datetime(elo_df["date"])
    mask = (elo_df["date"] >= start_date) & (elo_df["date"] <= end_date)
    elo_df = elo_df.loc[mask]
    elo_df["team1"] = elo_df["team1"].replace(ELO_TEAMS, STD_TEAMS)
    elo_df["team2"] = elo_df["team2"].replace(ELO_TEAMS, STD_TEAMS)
    return elo_df


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
    return agg_games_df


def parse_completed_games(combined_data_df: pd.DataFrame) -> pd.DataFrame:
    comp_games_df = combined_data_df[combined_data_df["result"].notna()]
    return comp_games_df


def parse_games_to_predict(combined_data_df: pd.DataFrame, current_week: int) -> pd.DataFrame:
    pred_games_df = pd.DataFrame(combined_data_df[combined_data_df["week"] == current_week])
    return pred_games_df


if __name__ == "__main__":
    main()
