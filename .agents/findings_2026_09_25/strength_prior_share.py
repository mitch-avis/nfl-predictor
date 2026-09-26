"""Split one week's published strength composite into its prior and in-season parts.

The strength snapshot for week ``N`` blends two things per component (``strength_snapshot``):
the in-season ridge solve over weeks before ``N`` and the previous season's full-season solve
regressed by ``WEEK1_REGRESSION_FACTOR``, weighted ``games / (games + PRIOR_BLEND_GAMES)``. This
script rebuilds all three from the cached play-by-play (no ETL run, nothing written under
``data/``), checks that the blend reproduces the published ``data/strength_snapshots.csv`` row for
row, and reports how the published rank relates to each part: rank per team, the correlations,
and each component's spread contributed by the prior and by the in-season solve.

Run from the repository root:

    .venv/bin/python .agents/findings_2026_09_25/strength_prior_share.py \
        > .agents/findings_2026_09_25/strength_prior_share_output.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars.pbp import aggregate_pbp_team_game_stats
from nfl_predictor.utils.polars.strength_snapshot import (
    COMPOSITE_WEIGHTS,
    build_strength_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "nflreadpy"
SEASON = 2026
WEEK = 3
COMPOSITE = "adj_strength_composite"


def _team_games() -> pl.DataFrame:
    """Aggregate the cached play-by-play of the season and the one before it to team-games."""
    frames = [
        aggregate_pbp_team_game_stats(pl.read_parquet(CACHE / f"pbp_{season}_reg.parquet"))
        for season in (SEASON - 1, SEASON)
    ]
    return pl.concat(frames, how="diagonal")


def _results() -> pl.DataFrame:
    """Return each team's wins and point margin before ``WEEK`` from the cached schedule."""
    schedule = pl.read_parquet(CACHE / f"schedule_{SEASON}.parquet").filter(
        (pl.col("game_type") == "REG") & (pl.col("week") < WEEK)
    )
    margin = pl.col("home_score") - pl.col("away_score")
    sides = pl.concat(
        [
            schedule.select(pl.col("home_abbr").alias("team_abbr"), margin.alias("margin")),
            schedule.select(pl.col("away_abbr").alias("team_abbr"), (-margin).alias("margin")),
        ]
    )
    return sides.group_by("team_abbr").agg(
        (pl.col("margin") > 0).sum().alias("wins"), pl.col("margin").sum().alias("point_margin")
    )


def main() -> int:
    """Print the decomposition; exit 1 when the rebuilt blend does not match the published file."""
    team_games = _team_games()
    teams = sorted(team_games.filter(pl.col("season") == SEASON)["team_abbr"].unique())
    last_regular = constants.get_regular_season_weeks(SEASON - 1) + 1
    prior = build_strength_snapshot(
        team_games, season=SEASON - 1, week=last_regular, blend_prior=False
    )
    in_season = build_strength_snapshot(
        team_games, season=SEASON, week=WEEK, blend_prior=False, teams=teams
    )
    blended = build_strength_snapshot(
        team_games, season=SEASON, week=WEEK, prior_snapshot=prior, teams=teams
    )
    published = pl.read_csv(ROOT / "data" / "strength_snapshots.csv").filter(
        (pl.col("season") == SEASON) & (pl.col("week") == WEEK)
    )

    table = (
        blended.select("team_abbr", pl.col(COMPOSITE).alias("published_blend"))
        .join(published.select("team_abbr", pl.col(COMPOSITE).alias("file")), on="team_abbr")
        .join(prior.select("team_abbr", pl.col(COMPOSITE).alias("prior")), on="team_abbr")
        .join(in_season.select("team_abbr", pl.col(COMPOSITE).alias("in_season")), on="team_abbr")
        .join(_results(), on="team_abbr")
        .with_columns(
            pl.col(column).rank(descending=True).cast(pl.Int64).alias(f"{column}_rank")
            for column in ("published_blend", "prior", "in_season")
        )
        .sort("published_blend", descending=True)
    )
    worst = float((table["published_blend"] - table["file"]).abs().max())
    games = constants.PRIOR_BLEND_GAMES
    played = float(in_season["strength_games_played"].max())
    weight = played / (played + games)
    keep = 1.0 - constants.WEEK1_REGRESSION_FACTOR

    print(f"season {SEASON}, snapshot week {WEEK}: {played:.0f} games per team")
    print(f"rebuilt blend vs data/strength_snapshots.csv, max abs difference: {worst}")
    print(f"in-season weight {weight:.4f}; prior weight {1 - weight:.4f} x regression {keep:.4f}")
    print()
    print("team  W  pts  published  prior  in-season")
    for row in table.iter_rows(named=True):
        print(
            f"{row['team_abbr']:<4} {row['wins']:>2} {row['point_margin']:>4}"
            f" {row['published_blend_rank']:>10} {row['prior_rank']:>6}"
            f" {row['in_season_rank']:>10}"
        )
    print()
    for other in ("prior", "in_season"):
        corr = table.select(pl.corr("published_blend", other)).item()
        print(f"correlation of the published composite with the {other} composite: {corr:.4f}")
    print()
    print("component spread (population std) each part contributes to the blend:")
    for column in COMPOSITE_WEIGHTS:
        prior_part = float(prior[column].std(ddof=0)) * (1 - weight) * keep
        in_part = float(in_season[column].std(ddof=0)) * weight
        print(f"  {column}: prior {prior_part:.4f}, in-season {in_part:.4f}")
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
