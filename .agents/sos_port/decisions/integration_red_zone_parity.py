"""Compare this repository's red-zone counts with nfl-sos-ratings' published game logs.

The acceptance check of the red-zone trip definition: after the change, every 1999-2025
team-game's ``red_zone_trips`` from this repository's play-by-play
(``pbp.aggregate_pbp_team_game_stats``, as ``integration_feature_values.py`` writes them to
``team_games.parquet``) must equal ``red_zone_trips`` in ``nfl-sos-ratings``' published
``{season}_team_game_logs.parquet``. A team's allowed trips are its opponent's offense, so its
``red_zone_trips_allowed`` is compared with the opponent's published ``red_zone_trips``.

The touchdown drives are compared too. The source publishes a rate, not a count: its
``red_zone_td_pct`` is the mean of ``fixed_drive_result == "Touchdown"`` over its trips, and a
trip whose drive has no recorded result (null) drops out of that mean, so its denominator is the
trips with a recorded result. Its touchdown drives are therefore ``red_zone_td_pct`` times its
trips less those without a result, rounded (a game without such a trip gives 0); the trips
without a result are counted on the measurement cache, the plays the source's logs were built
from. The script also prints how many team-games differ when the rate is read over all trips,
the reading that misses this. The allowed touchdown drives use ``red_zone_td_pct_allowed`` and
the opponent's trips. Last, it compares the rate itself: this repository's touchdown drives over
its trips, per team-game, against the published ``red_zone_td_pct`` (both null without a trip,
equal within ``1e-9``), and names the team-games that differ.

It then counts, on the regular-season plays with both teams known (``pbp.regular_season_plays``)
of the measurement cache ``integration_cache.py`` writes:

- how nflverse's ``drive_inside20`` flag sits on the plays of a drive: the plays that leave it
  null, the drives where it is null on every play, the drives whose plays disagree on it (some
  flagged, some 0 or null), and how many trips a rule reading only a drive's first play would
  lose against the rule that reads any of its plays;
- the trips whose drive has no recorded result, by season;
- the drives whose only plays at or inside the 20 belong to the try, which the replaced rule
  (any play at ``yardline_100 <= 20``) counted as trips, by drive result, with how many of them
  nflverse flags and how many hold nothing but the try. The try is ``integration_red_zone.py``'s:
  an extra point, a two-point attempt, or a play a penalty wiped out in kick formation or on a
  two-point attempt.

Team-games are matched on season, week and team, with the source's team codes mapped through
``constants.ALIAS_TO_CANONICAL``. Every input is read only. Run from the repository root:

    .venv/bin/python -I .agents/sos_port/decisions/integration_red_zone_parity.py \
        --team-games /tmp/step6_int/values/<label>/team_games.parquet \
        --cache-dir /tmp/step6_int/cache \
        --sos-data /home/mitch/workspace/nfl-sos-ratings/data
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import pbp

_KEYS = ["season", "week", "team_abbr"]
# (this repository's count, the source's count) for each comparison.
_PAIRS = (
    ("red_zone_trips", "source_trips"),
    ("red_zone_td_drives", "source_td_drives"),
    ("red_zone_trips_allowed", "source_trips_allowed"),
    ("red_zone_td_drives_allowed", "source_td_drives_allowed"),
)
# The touchdown comparisons read over all trips, the reading that misses the rate's denominator.
_NAIVE_PAIRS = (
    ("red_zone_td_drives", "source_td_drives_all_trips"),
    ("red_zone_td_drives_allowed", "source_td_drives_allowed_all_trips"),
)
# The play-by-play columns the drive facts read.
_PLAY_COLUMNS = [
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "yardline_100",
    "two_point_attempt",
    "fixed_drive",
    "fixed_drive_result",
    "drive_inside20",
    "desc",
]
# The yard line of the replaced rule: a play at or inside it made its drive a trip.
_YARD_LINE = 20
# Rates closer than this are the same number.
_RATE_TOLERANCE = 1e-9
# (this repository's touchdown drives, its trips, the source's rate, the trips without a result).
_RATES = (
    ("red_zone_td_drives", "red_zone_trips", "td_pct", "no_result_trips"),
    (
        "red_zone_td_drives_allowed",
        "red_zone_trips_allowed",
        "td_pct_allowed",
        "opponent_no_result_trips",
    ),
)
_TABLES = pl.Config(
    tbl_rows=-1,
    tbl_cols=-1,
    tbl_width_chars=400,
    tbl_hide_dataframe_shape=True,
    fmt_str_lengths=40,
)


def _source_logs(data_dir: Path, seasons: range) -> tuple[pl.DataFrame, str]:
    """Return the source's red-zone trips and rates, and one hash over the files."""
    digest = hashlib.sha256()
    frames: list[pl.DataFrame] = []
    for season in seasons:
        path = data_dir / f"{season}_team_game_logs.parquet"
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        frames.append(
            pl.read_parquet(path).select(
                pl.lit(season, dtype=pl.Int64).alias("season"),
                pl.col("week").cast(pl.Int64),
                pl.col("team").replace(constants.ALIAS_TO_CANONICAL).alias("team_abbr"),
                pl.col("red_zone_trips").cast(pl.Int64).alias("source_trips"),
                pl.col("red_zone_td_pct").alias("td_pct"),
                pl.col("red_zone_td_pct_allowed").alias("td_pct_allowed"),
            )
        )
    return pl.concat(frames), digest.hexdigest()


def _cached_plays(cache_dir: Path, seasons: range) -> pl.DataFrame:
    """Return the regular-season plays with both teams known, from the measurement cache."""
    frames = [
        pl.read_parquet(cache_dir / f"pbp_{season}_reg.parquet", columns=_PLAY_COLUMNS)
        for season in seasons
    ]
    return pbp.regular_season_plays(pl.concat(frames, how="vertical_relaxed"))


def _drives(plays: pl.DataFrame) -> pl.DataFrame:
    """Return one row per team drive: its flag, its result and whether the result is missing."""
    flag = pl.col("drive_inside20").fill_null(0.0) > 0
    return plays.group_by("season", "week", "posteam", "fixed_drive", maintain_order=True).agg(
        flag.any().alias("flagged"),
        flag.first().alias("first_play"),
        flag.all().alias("every_play"),
        pl.col("drive_inside20").is_null().all().alias("all_null"),
        pl.col("fixed_drive_result").is_null().all().alias("no_result"),
    )


def _no_result_trips(drives: pl.DataFrame) -> pl.DataFrame:
    """Return, per team-game, the trips whose drive has no recorded result."""
    return (
        drives.filter(pl.col("flagged") & pl.col("no_result"))
        .group_by("season", "week", "posteam")
        .agg(pl.len().cast(pl.Int64).alias("no_result_trips"))
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("posteam").alias("team_abbr"),
            "no_result_trips",
        )
    )


def _touchdowns(rate: str, trips: str, without_result: str | None) -> pl.Expr:
    """Return the rounded touchdown drives a published rate implies over its trips."""
    rated = pl.col(trips) - pl.col(without_result) if without_result else pl.col(trips)
    return (pl.col(rate).fill_null(0.0) * rated).round(0).cast(pl.Int64)


def _with_source(ours: pl.DataFrame, source: pl.DataFrame, no_result: pl.DataFrame) -> pl.DataFrame:
    """Join each team-game with its own published row and its opponent's."""
    source = source.join(no_result, on=_KEYS, how="left").with_columns(
        pl.col("no_result_trips").fill_null(0)
    )
    opponent = source.select(
        "season",
        "week",
        pl.col("team_abbr").alias("opponent_abbr"),
        pl.col("source_trips").alias("source_trips_allowed"),
        pl.col("no_result_trips").alias("opponent_no_result_trips"),
    )
    return (
        ours.join(source, on=_KEYS, how="full", coalesce=True)
        .join(opponent, on=["season", "week", "opponent_abbr"], how="left")
        .with_columns(
            _touchdowns("td_pct", "source_trips", "no_result_trips").alias("source_td_drives"),
            _touchdowns("td_pct_allowed", "source_trips_allowed", "opponent_no_result_trips").alias(
                "source_td_drives_allowed"
            ),
            _touchdowns("td_pct", "source_trips", None).alias("source_td_drives_all_trips"),
            _touchdowns("td_pct_allowed", "source_trips_allowed", None).alias(
                "source_td_drives_allowed_all_trips"
            ),
        )
    )


def _drive_flag_facts(plays: pl.DataFrame, drives: pl.DataFrame) -> str:
    """Return how ``drive_inside20`` sits on the plays of each drive, as text."""
    disagree = drives.filter(pl.col("flagged") & ~pl.col("every_play"))
    lost = drives.filter(pl.col("flagged") & ~pl.col("first_play"))
    return (
        f"regular-season plays {plays.height}, of which drive_inside20 is null on "
        f"{plays['drive_inside20'].null_count()}; plays without fixed_drive "
        f"{plays['fixed_drive'].null_count()}\n"
        f"drives {drives.height}: flagged on any play {int(drives['flagged'].sum())}, "
        f"null on every play {int(drives['all_null'].sum())}, "
        f"flagged on some plays but not others {disagree.height}\n"
        f"trips a rule reading only the drive's first play would lose: {lost.height}\n"
    )


def _try_facts(plays: pl.DataFrame) -> str:
    """Return the drives the replaced rule counted only through their try, by drive result."""
    in_zone = (pl.col("yardline_100") <= _YARD_LINE).fill_null(value=False)
    wiped = (pl.col("play_type") == "no_play").fill_null(value=False)
    on_try = pl.col("desc").str.contains(r"Kick formation|TWO-POINT").fill_null(value=False)
    is_try = (
        (pl.col("play_type") == "extra_point").fill_null(value=False)
        | (pl.col("two_point_attempt").fill_null(0) > 0)
        | (wiped & on_try)
    )
    drives = plays.group_by("season", "week", "posteam", "fixed_drive", maintain_order=True).agg(
        pl.col("fixed_drive_result").first().alias("result"),
        (pl.col("drive_inside20").fill_null(0.0) > 0).any().alias("flagged"),
        (in_zone & is_try).any().alias("try_in_zone"),
        (in_zone & ~is_try).any().alias("other_in_zone"),
        is_try.all().alias("only_the_try"),
    )
    try_only = drives.filter(pl.col("try_in_zone") & ~pl.col("other_in_zone"))
    table = (
        try_only.group_by("result")
        .agg(
            pl.len().alias("drives"),
            pl.col("flagged").sum().alias("flagged"),
            (~pl.col("flagged")).sum().alias("not_flagged"),
            pl.col("only_the_try").sum().alias("only_the_try"),
        )
        .sort("drives", descending=True)
    )
    return (
        f"drives whose only plays at or inside the {_YARD_LINE} belong to the try: "
        f"{try_only.height}, of which nflverse flags {int(try_only['flagged'].sum())}\n"
        f"by drive result:\n{table}\n"
    )


def _comparison(both: pl.DataFrame, joined: pl.DataFrame, digest: str, span: str) -> str:
    """Return the per-team-game comparison with the published logs, as text."""
    differ = {name: both.filter(pl.col(name).ne_missing(pl.col(theirs))) for name, theirs in _PAIRS}
    naive = {
        name: both.filter(pl.col(name).ne_missing(pl.col(theirs))).height
        for name, theirs in _NAIVE_PAIRS
    }
    by_season = (
        both.group_by("season", maintain_order=True)
        .agg(
            pl.len().alias("team_games"),
            *(
                pl.col(name).ne_missing(pl.col(theirs)).sum().alias(f"{name}_differ")
                for name, theirs in _PAIRS
            ),
            pl.col("red_zone_trips").sum().alias("trips"),
            pl.col("source_trips").sum().alias("source_trips"),
            pl.col("red_zone_td_drives").sum().alias("td_drives"),
            pl.col("source_td_drives").sum().alias("source_td_drives"),
        )
        .sort("season")
    )
    totals = both.select(
        pl.col("red_zone_trips").sum().alias("trips"),
        pl.col("source_trips").sum().alias("source_trips"),
        pl.col("red_zone_td_drives").sum().alias("td_drives"),
        pl.col("source_td_drives").sum().alias("source_td_drives"),
        pl.col("no_result_trips").sum().alias("no_result_trips"),
    ).row(0, named=True)
    text = (
        f"source game logs {span} SHA-256 {digest}\n"
        f"team-games: both {both.height}, "
        f"only ours {joined.filter(pl.col('source_trips').is_null()).height}, "
        f"only the source {joined.filter(pl.col('red_zone_trips').is_null()).height}\n"
        + "".join(f"{name} differs in {frame.height}\n" for name, frame in differ.items())
        + "".join(
            f"{name} differs in {count} when the source's rate is read over all its trips\n"
            for name, count in naive.items()
        )
        + f"trips {totals['trips']} (source {totals['source_trips']}), of which without a "
        f"recorded drive result {totals['no_result_trips']}; touchdown drives "
        f"{totals['td_drives']} (source {totals['source_td_drives']}); league touchdown rate "
        f"per trip {totals['td_drives'] / totals['trips']:.4f}\n{by_season}\n"
    )
    for name, frame in differ.items():
        text += f"{name} that differ:\n{frame.sort(_KEYS)}\n"
    for touchdowns, trips, published, without_result in _RATES:
        ours = pl.when(pl.col(trips) > 0).then(pl.col(touchdowns) / pl.col(trips))
        rates = both.filter(
            ((ours - pl.col(published)).abs() > _RATE_TOLERANCE).fill_null(value=False)
            | (ours.is_null() != pl.col(published).is_null())
        ).select(*_KEYS, trips, touchdowns, without_result, ours.alias("rate"), published)
        held = rates.filter(pl.col(without_result) > 0).height
        text += (
            f"{touchdowns} / {trips} differs from the published {published} in {rates.height} "
            f"team-games, {held} of them holding a trip without a recorded result:\n"
            f"{rates.sort(_KEYS)}\n"
        )
    return text


def main() -> None:
    """Print the comparison and the drive facts."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--team-games", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sos-data", type=Path, required=True)
    parser.add_argument("--first-season", type=int, default=1999)
    parser.add_argument("--last-season", type=int, default=2025)
    args = parser.parse_args()
    seasons = range(args.first_season, args.last_season + 1)
    ours = (
        pl.read_parquet(args.team_games)
        .filter(pl.col("season").is_between(args.first_season, args.last_season))
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            "team_abbr",
            "opponent_abbr",
            *(pl.col(name).cast(pl.Int64) for name, _ in _PAIRS),
        )
    )
    source, digest = _source_logs(args.sos_data, seasons)
    plays = _cached_plays(args.cache_dir, seasons)
    drives = _drives(plays)
    no_result = _no_result_trips(drives)
    joined = _with_source(ours, source, no_result)
    both = joined.filter(
        pl.col("red_zone_trips").is_not_null() & pl.col("source_trips").is_not_null()
    )
    no_result_by_season = (
        no_result.group_by("season")
        .agg(
            pl.len().alias("team_games"),
            pl.col("no_result_trips").sum().alias("trips_without_a_result"),
        )
        .sort("season")
    )
    span = f"{args.first_season}-{args.last_season}"
    with _TABLES:
        sys.stdout.write(
            f"team-games: ours {ours.height}, source {source.height}\n"
            + _comparison(both, joined, digest, span)
            + f"\n## drive_inside20 on the plays of a drive\n{_drive_flag_facts(plays, drives)}"
            + f"\n## Trips whose drive has no recorded result, by season\n{no_result_by_season}\n"
            + f"\n## The try under the replaced rule\n{_try_facts(plays)}"
        )


if __name__ == "__main__":
    main()
