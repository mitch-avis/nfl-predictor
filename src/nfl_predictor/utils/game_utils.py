"""Game-related utilities for NFL data processing.

This module provides functions for handling game-specific data:
    - Spread to moneyline conversion
    - Missing moneyline calculation
    - QB data lookups and fills
    - Future game line fills
"""

import math
from typing import Any

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.scraping_utils import scrape_survivor_grid_spreads


def spread_to_moneyline(spread: float, vig: float = 0.05) -> int:
    """Convert an NFL point spread to a moneyline, including the effect of vig.

    Uses the normal distribution to model score differentials and convert
    spreads to implied probabilities, then to moneyline odds.

    Args:
        spread: The point spread (negative for favorites, positive for underdogs)
        vig: The vig percentage as a decimal (default is 0.05 for 5%)

    Returns:
        The moneyline corresponding to the given spread

    """
    # Use the standard deviation of NFL score differences
    std_dev = constants.SCORE_DIFF_STD_DEV

    # Calculate the implied probability using normal CDF
    # This is equivalent to: stats.norm.cdf(-spread, 0, std_dev)
    z_score = -spread / std_dev
    implied_probability = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))

    # Scale probabilities to include the vig
    adjusted_implied_probability = implied_probability * (1 + vig)

    # Clamp probability to avoid division by zero
    adjusted_implied_probability = max(0.01, min(0.99, adjusted_implied_probability))

    # Convert probabilities back to moneyline odds
    if spread < 0:
        # Favorite moneyline (negative)
        moneyline = -100 * (adjusted_implied_probability / (1 - adjusted_implied_probability))
    elif spread > 0:
        # Underdog moneyline (positive)
        moneyline = 100 * ((1 - adjusted_implied_probability) / adjusted_implied_probability)
    else:
        # Pick'em (spread = 0): slight underdog due to vig
        moneyline = 100 * ((1 - adjusted_implied_probability) / adjusted_implied_probability)

    return round(moneyline)


def fill_missing_moneylines(df: pl.DataFrame) -> pl.DataFrame:
    """Fill in missing moneylines by calculating them from spreads.

    For games where moneyline is missing but spread is available,
    calculates the moneyline using spread_to_moneyline conversion.

    Args:
        df: DataFrame with home_spread, away_spread, home_moneyline, away_moneyline columns

    Returns:
        DataFrame with moneylines filled in where possible

    """
    if "home_spread" not in df.columns:
        return df

    # Check if moneyline columns exist, create them if not
    if "home_moneyline" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("home_moneyline"))
    if "away_moneyline" not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("away_moneyline"))

    # Find rows where spread exists but moneyline doesn't
    needs_home_ml = df.filter(
        pl.col("home_spread").is_not_null() & pl.col("home_moneyline").is_null()
    )
    needs_away_ml = df.filter(
        pl.col("away_spread").is_not_null() & pl.col("away_moneyline").is_null()
    )

    if needs_home_ml.height == 0 and needs_away_ml.height == 0:
        return df

    log.debug(
        "Calculating moneylines for %d home/%d away missing values",
        needs_home_ml.height,
        needs_away_ml.height,
    )

    # Calculate moneylines from spreads using map_elements
    # For home team: use home_spread directly
    # For away team: use away_spread (which is -home_spread)
    return df.with_columns(
        [
            pl.when(pl.col("home_spread").is_not_null() & pl.col("home_moneyline").is_null())
            .then(
                pl.col("home_spread").map_elements(
                    lambda s: spread_to_moneyline(s) if s is not None else None,
                    return_dtype=pl.Int64,
                )
            )
            .otherwise(pl.col("home_moneyline"))
            .alias("home_moneyline"),
            pl.when(pl.col("away_spread").is_not_null() & pl.col("away_moneyline").is_null())
            .then(
                pl.col("away_spread").map_elements(
                    lambda s: spread_to_moneyline(s) if s is not None else None,
                    return_dtype=pl.Int64,
                )
            )
            .otherwise(pl.col("away_moneyline"))
            .alias("away_moneyline"),
        ]
    )


def get_latest_qb_by_team(elo_df: pl.DataFrame) -> pl.DataFrame:
    """Get the most recent starting QB for each team from ELO data.

    Uses the qb_elos.csv data to find the most recent game where each team
    had a recorded starting QB.

    Args:
        elo_df: Raw ELO DataFrame with qb1, qb2, team1, team2 columns

    Returns:
        DataFrame with columns: team_abbr, qb_name, qb_value_pre, qb_elo_pre

    """
    required_cols = ["qb1", "qb2", "team1", "team2"]
    if not all(c in elo_df.columns for c in required_cols):
        return pl.DataFrame()

    # Normalize team abbreviations if needed
    elo_df = elo_df.with_columns(
        [
            pl.col("team1").replace(constants.ALIAS_TO_CANONICAL).alias("team1"),
            pl.col("team2").replace(constants.ALIAS_TO_CANONICAL).alias("team2"),
        ]
    )

    # Get home team QBs (team1 = home, qb1 = home QB)
    home_qbs = elo_df.filter(pl.col("qb1").is_not_null() & (pl.col("qb1") != "")).select(
        [
            pl.col("date"),
            pl.col("team1").alias("team_abbr"),
            pl.col("qb1").alias("qb_name"),
            pl.col("qb1_value_pre").alias("qb_value_pre"),
            pl.col("qbelo1_pre").alias("qb_elo_pre"),
        ]
    )

    # Get away team QBs (team2 = away, qb2 = away QB)
    away_qbs = elo_df.filter(pl.col("qb2").is_not_null() & (pl.col("qb2") != "")).select(
        [
            pl.col("date"),
            pl.col("team2").alias("team_abbr"),
            pl.col("qb2").alias("qb_name"),
            pl.col("qb2_value_pre").alias("qb_value_pre"),
            pl.col("qbelo2_pre").alias("qb_elo_pre"),
        ]
    )

    # Combine and sort by date descending
    all_qbs = pl.concat([home_qbs, away_qbs])
    all_qbs = all_qbs.sort("date", descending=True)

    # Get most recent QB per team
    return all_qbs.group_by("team_abbr").agg(
        [
            pl.col("qb_name").first(),
            pl.col("qb_value_pre").first(),
            pl.col("qb_elo_pre").first(),
        ]
    )


def get_qb_elo_by_name(elo_df: pl.DataFrame, qb_name: str) -> dict:
    """Get the most recent ELO values for a specific QB by name.

    Searches both qb1 and qb2 columns to find the most recent game
    the QB started, then returns their pre-game ELO values.

    Args:
        elo_df: Full ELO DataFrame
        qb_name: Name of the QB to search for

    Returns:
        Dict with qb_value_pre and qb_elo_pre, or empty dict if not found

    """
    if elo_df.height == 0 or not qb_name:
        return {}

    # Search in qb1 column (home QB)
    qb1_games = elo_df.filter(pl.col("qb1") == qb_name)
    if qb1_games.height > 0:
        # Sort by date descending, take first
        latest = qb1_games.sort("date", descending=True).head(1)
        qb_value = latest.select("qb1_value_pre").item()
        qb_elo = latest.select("qbelo1_pre").item()
        return {"qb_value_pre": qb_value, "qb_elo_pre": qb_elo}

    # Search in qb2 column (away QB)
    qb2_games = elo_df.filter(pl.col("qb2") == qb_name)
    if qb2_games.height > 0:
        latest = qb2_games.sort("date", descending=True).head(1)
        qb_value = latest.select("qb2_value_pre").item()
        qb_elo = latest.select("qbelo2_pre").item()
        return {"qb_value_pre": qb_value, "qb_elo_pre": qb_elo}

    return {}


def fill_future_qb_data(
    df: pl.DataFrame,
    elo_df: pl.DataFrame,
) -> pl.DataFrame:
    """Fill in QB data for future games using each team's most recent starter.

    For games where away_qb/home_qb are null, looks up the most recent QB
    for each team from the ELO data and fills in their name and ELO values.

    Args:
        df: Combined data DataFrame with potential null QBs
        elo_df: Raw ELO DataFrame (to look up QB-specific ELO values)

    Returns:
        DataFrame with QB data filled in for future games

    """
    required_cols = ["away_abbr", "home_abbr"]
    if not all(c in df.columns for c in required_cols):
        return df

    # Check if away_qb column exists; if not, all games need QB data
    has_away_qb = "away_qb" in df.columns
    has_home_qb = "home_qb" in df.columns

    null_away = df.filter(pl.col("away_qb").is_null()) if has_away_qb else df

    null_home = df.filter(pl.col("home_qb").is_null()) if has_home_qb else df

    if null_away.height == 0 and null_home.height == 0:
        return df

    log.debug(
        "Filling QB data for %d away nulls and %d home nulls",
        null_away.height,
        null_home.height,
    )

    # Get latest QB per team from ELO data
    latest_qbs = get_latest_qb_by_team(elo_df)

    if latest_qbs.height == 0:
        return df

    # Build lookup dict for QBs
    qb_lookup = {}
    for row in latest_qbs.iter_rows(named=True):
        team = row["team_abbr"]
        qb_lookup[team] = {
            "qb_name": row["qb_name"],
            "qb_value_pre": row["qb_value_pre"],
            "qb_elo_pre": row["qb_elo_pre"],
        }

    if null_away.height > 0:
        df = _fill_side_qb(df, "away", qb_lookup, has_qb_column=has_away_qb)
    if null_home.height > 0:
        df = _fill_side_qb(df, "home", qb_lookup, has_qb_column=has_home_qb)
    return df


def _fill_side_qb(
    df: pl.DataFrame,
    side: str,
    qb_lookup: dict[str, dict[str, Any]],
    *,
    has_qb_column: bool,
) -> pl.DataFrame:
    """Fill one side's missing quarterback name and ELO values from the latest starters."""

    def looked_up(key: str, dtype: type[pl.DataType]) -> pl.Expr:
        return pl.col(f"{side}_abbr").map_elements(
            lambda t: qb_lookup.get(t, {}).get(key), return_dtype=dtype
        )

    qb_col = f"{side}_qb"
    new_cols = [
        pl.when(pl.col(qb_col).is_null() if has_qb_column else pl.lit(True))
        .then(looked_up("qb_name", pl.Utf8))
        .otherwise(pl.col(qb_col) if has_qb_column else pl.lit(None))
        .alias(qb_col),
    ]
    # Only fill ELO values if columns exist and are null
    for key in ("qb_value_pre", "qb_elo_pre"):
        column = f"{side}_{key}"
        if column in df.columns:
            new_cols.append(
                pl.when(pl.col(column).is_null())
                .then(looked_up(key, pl.Float64))
                .otherwise(pl.col(column))
                .alias(column)
            )
    return df.with_columns(new_cols)


def fill_future_game_lines(df: pl.DataFrame) -> pl.DataFrame:
    """Fill in lines (spreads, moneylines, totals) for future games using SurvivorGrid data.

    Scrapes current spreads from SurvivorGrid.com and applies them to future games
    that don't have lines data. Also calculates moneylines from spreads and uses
    the historical average for total lines.

    Args:
        df: DataFrame with games, some of which may be missing lines data

    Returns:
        DataFrame with lines filled in for future games

    """
    # Only process if we have the required columns
    required_cols = ["week", "away_abbr", "home_abbr"]
    if not all(c in df.columns for c in required_cols):
        return df

    # Identify future games (no score data)
    if "away_score" in df.columns:
        future_mask = pl.col("away_score").is_null()
    # Fall back to checking if lines are missing
    elif "home_spread" in df.columns:
        future_mask = pl.col("home_spread").is_null()
    else:
        return df

    future_games = df.filter(future_mask)
    if future_games.height == 0:
        return df

    log.debug("Found %d future games to fill lines for", future_games.height)

    # Scrape current spreads
    spreads_data = scrape_survivor_grid_spreads()

    if not spreads_data:
        log.warning("No spreads data available from SurvivorGrid")
        return df

    updates = _future_line_updates(df, spreads_data)
    if not updates:
        return df

    log.info("Filling lines for %d future games from SurvivorGrid", len(updates))
    df = _apply_line_updates(df, pl.DataFrame(updates))

    # Now calculate moneylines from the new spreads
    return fill_missing_moneylines(df)


def _home_spread(
    spreads_data: dict[str, dict[int, float]], week: int, home_abbr: str, away_abbr: str
) -> float | None:
    """Return a game's home spread from SurvivorGrid data, negating the away side's if needed.

    SurvivorGrid shows the spread from the perspective of the team listed: a positive spread
    marks the underdog, a negative one the favorite.
    """
    home_spread = spreads_data.get(home_abbr, {}).get(week)
    if home_spread is not None:
        return home_spread
    # Try getting from away perspective (negate)
    away_spread = spreads_data.get(away_abbr, {}).get(week)
    if away_spread is not None:
        return -away_spread
    return None


def _future_line_updates(
    df: pl.DataFrame, spreads_data: dict[str, dict[int, float]]
) -> list[dict[str, Any]]:
    """Return new lines for each unplayed game that has no spread yet."""
    updates = []
    for row in df.iter_rows(named=True):
        week = row.get("week")
        home_abbr = row.get("home_abbr")
        away_abbr = row.get("away_abbr")
        # Only update future games (no score) with missing spreads
        is_future = row.get("away_score") is None
        has_spread = row.get("home_spread") is not None
        if not (is_future and not has_spread and week and home_abbr and away_abbr):
            continue
        home_spread = _home_spread(spreads_data, week, home_abbr, away_abbr)
        if home_spread is not None:
            updates.append(
                {
                    "game_id": row.get("game_id"),
                    "new_home_spread": home_spread,
                    "new_away_spread": -home_spread,
                    "new_total_line": constants.DEFAULT_TOTAL_LINE,
                }
            )
    return updates


def _apply_line_updates(df: pl.DataFrame, updates_df: pl.DataFrame) -> pl.DataFrame:
    """Write the new spreads, and totals where none exist, onto the matching games."""
    df = df.join(updates_df, on="game_id", how="left")
    if "new_home_spread" not in df.columns:
        return df
    df = df.with_columns(
        [
            pl.when(pl.col("new_home_spread").is_not_null())
            .then(pl.col("new_home_spread"))
            .otherwise(pl.col("home_spread") if "home_spread" in df.columns else pl.lit(None))
            .alias("home_spread"),
            pl.when(pl.col("new_away_spread").is_not_null())
            .then(pl.col("new_away_spread"))
            .otherwise(pl.col("away_spread") if "away_spread" in df.columns else pl.lit(None))
            .alias("away_spread"),
            pl.when(
                pl.col("new_total_line").is_not_null()
                & (pl.col("total_line").is_null() if "total_line" in df.columns else pl.lit(True))
            )
            .then(pl.col("new_total_line"))
            .otherwise(pl.col("total_line") if "total_line" in df.columns else pl.lit(None))
            .alias("total_line"),
        ]
    )
    # Drop temporary columns
    return df.drop(["new_home_spread", "new_away_spread", "new_total_line"])
