"""Count, per regular season, the plays each play-by-play correction of the integration touches.

Reads the measurement cache ``integration_cache.py`` writes (read-only) and prints, for
1999-2025, one table per question:

0. Carries: plays with ``rush`` set but not ``rush_attempt`` (runs a penalty wiped out) and their
   EPA, and rush attempts (two-point tries, scrambles and kneels out) without ``rush``, with how
   many of those set no other snap flag (``qb_dropback``, ``qb_spike``).
1. Scrambles on run plays (``qb_scramble`` set, ``play_type`` ``run``): how many, how many of them
   carry ``qb_dropback`` 0 (nflverse's pre-2006 coding gap) and how many carry ``rush`` 1 (coded
   as runs); and, for comparison, the penalty-wiped scrambles (``play_type`` ``no_play``) without
   ``qb_dropback``, which the rule leaves alone.
2. Plays flagged both ``qb_dropback`` and ``rush``.
3. The games whose player team columns name a single team (``player_team_repair.one_team_games``
   over ``constants.PBP_PLAYER_TEAM_ID_COLUMNS``, the detection the ETL now runs).
4. Plays with a third-down, fourth-down or two-point flag whose ``play_type`` is not a run, pass,
   kneel or spike, by play type (null included).
5. Plays flagged both converted and failed on third or fourth down.
6. Two-point plays whose result is a success without the ``two_point_attempt`` flag.
7. Dropbacks with neither a passer nor a rusher id, which no quarterback can be credited with.

Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_play_facts.py \
        --cache-dir /tmp/step6_int/cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import loaders, player_team_repair

SITUATIONAL = ("run", "pass", "qb_kneel", "qb_spike")


def _flag(column: str) -> pl.Expr:
    """Return a 0/1 play flag as a boolean, null read as 0."""
    return pl.col(column).fill_null(0) > 0


def _print(title: str, frame: pl.DataFrame) -> None:
    """Print one titled table in full."""
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, tbl_hide_dataframe_shape=True):
        sys.stdout.write(f"\n## {title}\n{frame}\n")


def season_facts(plays: pl.DataFrame) -> dict[str, int]:
    """Return one season's counts for tables 1, 2, 5, 6 and 7."""
    run = (pl.col("play_type") == "run").fill_null(value=False)
    no_play = (pl.col("play_type") == "no_play").fill_null(value=False)
    scramble = _flag("qb_scramble")
    run_scramble = scramble & run
    dropback = (_flag("qb_dropback") | run_scramble) & ~_flag("two_point_attempt")
    dropback &= ~(_flag("qb_kneel") | _flag("qb_spike"))
    rush_attempt = _flag("rush_attempt") & ~_flag("two_point_attempt")
    rush_attempt &= ~(scramble | _flag("qb_kneel"))
    return plays.select(
        (_flag("rush") & ~_flag("rush_attempt")).sum().alias("rush_without_attempt"),
        pl.col("epa")
        .fill_null(0.0)
        .filter(_flag("rush") & ~_flag("rush_attempt"))
        .sum()
        .round(2)
        .alias("rush_without_attempt_epa"),
        (rush_attempt & ~_flag("rush")).sum().alias("attempt_without_rush"),
        (rush_attempt & ~_flag("rush") & ~_flag("qb_dropback") & ~_flag("qb_spike"))
        .sum()
        .alias("attempt_without_rush_not_snap"),
        run_scramble.sum().alias("run_scrambles"),
        (run_scramble & ~_flag("qb_dropback")).sum().alias("run_scrambles_no_dropback"),
        (run_scramble & _flag("rush")).sum().alias("run_scrambles_rush"),
        (scramble & no_play & ~_flag("qb_dropback")).sum().alias("wiped_scrambles_no_dropback"),
        (_flag("qb_dropback") & _flag("rush")).sum().alias("dropback_and_rush"),
        (_flag("third_down_converted") & _flag("third_down_failed")).sum().alias("third_both"),
        (_flag("fourth_down_converted") & _flag("fourth_down_failed")).sum().alias("fourth_both"),
        ((pl.col("two_point_conv_result") == "success").fill_null(value=False) & ~_flag(
            "two_point_attempt"
        ))
        .sum()
        .alias("two_point_success_without_flag"),
        (
            dropback
            & pl.col("passer_player_id").is_null()
            & pl.col("rusher_player_id").is_null()
        )
        .sum()
        .alias("dropbacks_without_ids"),
    ).row(0, named=True)


def situational_off_type(plays: pl.DataFrame) -> pl.DataFrame:
    """Return the flagged plays whose play type is outside the run, pass, kneel and spike."""
    flags = {
        "third_down": _flag("third_down_converted") | _flag("third_down_failed"),
        "fourth_down": _flag("fourth_down_converted") | _flag("fourth_down_failed"),
        "two_point": _flag("two_point_attempt"),
    }
    outside = ~pl.col("play_type").is_in(list(SITUATIONAL)).fill_null(value=False)
    return (
        plays.filter(outside)
        .group_by(pl.col("play_type").fill_null("(null)"), maintain_order=True)
        .agg(*(flag.sum().alias(name) for name, flag in flags.items()))
        .filter(pl.sum_horizontal(*flags) > 0)
        .sort("play_type")
    )


def main() -> None:
    """Print every table."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    rows: list[dict[str, int]] = []
    off_type: list[pl.DataFrame] = []
    one_team: list[dict[str, object]] = []
    for season in range(args.first_season, args.last_season + 1):
        plays = loaders.load_pbp(
            [season], cache_dir=args.cache_dir, current_season=args.last_season + 1
        )
        plays = plays.filter(pl.col("posteam").is_not_null() & (pl.col("posteam") != ""))
        rows.append({"season": season, **season_facts(plays)})
        off_type.append(situational_off_type(plays).with_columns(pl.lit(season).alias("season")))
        games = player_team_repair.one_team_games(
            plays, list(constants.PBP_PLAYER_TEAM_ID_COLUMNS)
        )
        one_team.extend({"season": season, "game_id": game} for game in games)
    facts = pl.DataFrame(rows)
    _print("Scrambles, both-flagged plays and conversion flags per season", facts)
    _print("Totals 1999-2025", facts.drop("season").sum())
    _print(
        "Third-down, fourth-down and two-point flags outside run, pass, kneel and spike plays",
        pl.concat(off_type).select("season", pl.exclude("season")),
    )
    _print("Games whose player team columns name one team", pl.DataFrame(one_team))


if __name__ == "__main__":
    main()
