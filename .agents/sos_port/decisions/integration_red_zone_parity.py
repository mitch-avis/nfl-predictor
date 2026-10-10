"""Compare this repository's red-zone counts with nfl-sos-ratings' published game logs.

The acceptance check of the red-zone trip definition. This repository counts a drive group (the
plays of one team's ``fixed_drive`` in one game) as the team's trip when nflverse flags any of
its plays ``drive_inside20`` and the team ran an offensive snap in it; the source counts every
flagged group. So every 1999-2025 team-game's ``red_zone_trips`` from this repository's
play-by-play (``pbp.aggregate_pbp_team_game_stats``, as ``integration_feature_values.py``
writes them to ``team_games.parquet``) must equal ``red_zone_trips`` in ``nfl-sos-ratings``'
published ``{season}_team_game_logs.parquet`` less the team's flagged groups without a snap,
the one attributed cause; the script counts the team-games that differ, how many of them that
cause explains exactly, and lists the rest (none is expected) and every group without a snap.
A team's allowed trips are its opponent's offense, so its ``red_zone_trips_allowed`` is
compared with the opponent's published ``red_zone_trips``, less the opponent's groups.

A snap is an offensive scrimmage snap as ``offensive_snaps`` counts them, written out here
rather than imported: a play with ``qb_dropback``, ``rush``, ``qb_kneel`` or ``qb_spike`` set,
or a scramble filed as a run play (``qb_scramble`` set on a ``run`` play), that is not a
two-point try. The groups are counted on the measurement cache ``integration_cache.py`` writes,
the plays the source's logs were built from. For comparison it also counts the groups a
narrower test would leave without a snap, one that does not count a scramble filed as a run
play, and lists the groups whose only snap is such a scramble.

The touchdown drives are compared too, with the same attribution (a group without a snap whose
drive ended in a touchdown). The source publishes a rate, not a count: its ``red_zone_td_pct``
is the mean of ``fixed_drive_result == "Touchdown"`` over its trips, and a trip whose drive has
no recorded result (null) drops out of that mean, so its denominator is the trips with a
recorded result. Its touchdown drives are therefore ``red_zone_td_pct`` times its trips less
those without a result, rounded (a game without such a trip gives 0). The script also prints how
many team-games differ when the rate is read over all trips, the reading that misses this. The
allowed touchdown drives use ``red_zone_td_pct_allowed`` and the opponent's trips. Last, it
compares the rate itself: this repository's touchdown drives over its trips, per team-game,
against the published ``red_zone_td_pct`` (both null without a trip, equal within ``1e-9``),
and counts the team-games that differ without holding a group without a snap or a trip without
a recorded result.

It then counts, on the regular-season plays with both teams known (``pbp.regular_season_plays``)
of the measurement cache:

- how nflverse's ``drive_inside20`` flag sits on the plays of a drive: the plays that leave it
  null, the drives where it is null on every play, the drives whose plays disagree on it (some
  flagged, some 0 or null), and how many trips a rule reading only a drive's first play would
  lose against the rule that reads any of its plays;
- the trips whose drive has no recorded result, by season;
- the drives whose only plays at or inside the 20 belong to the try, which the replaced rule
  (any play at ``yardline_100 <= 20``) counted as trips, by drive result, with how many of them
  nflverse flags, how many of those hold no snap, and how many hold nothing but the try. The try
  is ``integration_red_zone.py``'s:
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
_DRIVE = ["season", "week", "posteam", "fixed_drive"]
# (this repository's count, the source's count, the source's groups without a snap) per check.
_PAIRS = (
    ("red_zone_trips", "source_trips", "no_snap_trips"),
    ("red_zone_td_drives", "source_td_drives", "no_snap_td_drives"),
    ("red_zone_trips_allowed", "source_trips_allowed", "opponent_no_snap_trips"),
    ("red_zone_td_drives_allowed", "source_td_drives_allowed", "opponent_no_snap_td_drives"),
)
# The touchdown comparisons read over all trips, the reading that misses the rate's denominator.
_NAIVE_PAIRS = (
    ("red_zone_td_drives", "source_td_drives_all_trips", "no_snap_td_drives"),
    (
        "red_zone_td_drives_allowed",
        "source_td_drives_allowed_all_trips",
        "opponent_no_snap_td_drives",
    ),
)
# (touchdown drives, trips, the published rate, its groups without a snap, its trips without
# a recorded result) for the offense and the allowed side.
_RATES = (
    ("red_zone_td_drives", "red_zone_trips", "td_pct", "no_snap_trips", "no_result_trips"),
    (
        "red_zone_td_drives_allowed",
        "red_zone_trips_allowed",
        "td_pct_allowed",
        "opponent_no_snap_trips",
        "opponent_no_result_trips",
    ),
)
# Per-team-game counts taken from the plays, zero where a team-game has none.
_GROUP_COUNTS = ("no_snap_trips", "no_snap_td_drives", "no_result_trips")
# The play-by-play columns the drive facts read.
_PLAY_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "play_type",
    "yardline_100",
    "qb_dropback",
    "qb_scramble",
    "rush",
    "qb_kneel",
    "qb_spike",
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
_TABLES = pl.Config(
    tbl_rows=-1,
    tbl_cols=-1,
    tbl_width_chars=400,
    tbl_hide_dataframe_shape=True,
    fmt_str_lengths=60,
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


def _set(name: str) -> pl.Expr:
    """Return whether a 0/1 play flag is set, a null read as 0."""
    return pl.col(name).fill_null(0.0) > 0


def _snap(*, run_play_scrambles: bool = True) -> pl.Expr:
    """Return an offensive scrimmage snap as ``offensive_snaps`` counts it, not a two-point try.

    With ``run_play_scrambles`` false, a scramble filed as a run play without the rush or dropback
    flag is no snap: the narrower test, printed for comparison only.
    """
    run_play_scramble = _set("qb_scramble") & (pl.col("play_type") == "run").fill_null(value=False)
    scrimmage = _set("qb_dropback") | _set("rush") | _set("qb_kneel") | _set("qb_spike")
    if run_play_scrambles:
        scrimmage |= run_play_scramble
    return scrimmage & ~_set("two_point_attempt")


def _drives(plays: pl.DataFrame) -> pl.DataFrame:
    """Return one row per drive group: its flag, its snap, its result and its plays' types."""
    flag = _set("drive_inside20")
    return plays.group_by(_DRIVE, maintain_order=True).agg(
        pl.col("game_id").first(),
        flag.any().alias("flagged"),
        flag.first().alias("first_play"),
        flag.all().alias("every_play"),
        pl.col("drive_inside20").is_null().all().alias("all_null"),
        _snap().any().alias("snap"),
        _snap(run_play_scrambles=False).any().alias("snap_without_scrambles"),
        pl.col("fixed_drive_result").first().alias("result"),
        pl.col("fixed_drive_result").is_null().all().alias("no_result"),
        pl.col("play_type").drop_nulls().unique().sort().str.join(",").alias("plays"),
    )


def _group_counts(drives: pl.DataFrame) -> pl.DataFrame:
    """Return, per team-game, its flagged groups without a snap and its trips without a result."""
    flagged = drives.filter(pl.col("flagged"))
    no_snap = ~pl.col("snap")
    touchdown = (pl.col("result") == "Touchdown").fill_null(value=False)
    return (
        flagged.group_by("season", "week", "posteam")
        .agg(
            no_snap.sum().cast(pl.Int64).alias("no_snap_trips"),
            (no_snap & touchdown).sum().cast(pl.Int64).alias("no_snap_td_drives"),
            pl.col("no_result").sum().cast(pl.Int64).alias("no_result_trips"),
        )
        .select(
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("posteam").alias("team_abbr"),
            *_GROUP_COUNTS,
        )
    )


def _touchdowns(rate: str, trips: str, without_result: str | None) -> pl.Expr:
    """Return the rounded touchdown drives a published rate implies over its trips."""
    rated = pl.col(trips) - pl.col(without_result) if without_result else pl.col(trips)
    return (pl.col(rate).fill_null(0.0) * rated).round(0).cast(pl.Int64)


def _with_source(ours: pl.DataFrame, source: pl.DataFrame, counts: pl.DataFrame) -> pl.DataFrame:
    """Join each team-game with its own published row and its opponent's."""
    source = source.join(counts, on=_KEYS, how="left").with_columns(
        pl.col(name).fill_null(0) for name in _GROUP_COUNTS
    )
    opponent = source.select(
        "season",
        "week",
        pl.col("team_abbr").alias("opponent_abbr"),
        pl.col("source_trips").alias("source_trips_allowed"),
        *(pl.col(name).alias(f"opponent_{name}") for name in _GROUP_COUNTS),
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
        f"flagged groups a rule reading only the group's first play would lose: {lost.height}\n"
    )


def _no_snap_facts(drives: pl.DataFrame) -> str:
    """Return the flagged groups without a snap: totals, by plays and season, and the list."""
    groups = drives.filter(pl.col("flagged") & ~pl.col("snap"))
    touchdown = (pl.col("result") == "Touchdown").fill_null(value=False)
    by_plays = (
        groups.group_by("plays")
        .agg(pl.len().alias("groups"), touchdown.sum().alias("touchdowns"))
        .sort("groups", "plays", descending=[True, False])
    )
    by_result = (
        groups.group_by("result")
        .agg(pl.len().alias("groups"))
        .sort(["groups", "result"], descending=[True, False], nulls_last=True)
    )
    by_season = groups.group_by("season").agg(pl.len().alias("groups")).sort("season")
    columns = ["season", "week", "game_id", "posteam", "fixed_drive", "result", "plays"]
    order = ["season", "week", "game_id", "posteam", "fixed_drive"]
    listing = groups.select(columns).sort(order)
    narrower = drives.filter(pl.col("flagged") & ~pl.col("snap_without_scrambles"))
    scramble_only = narrower.filter(pl.col("snap")).select(columns).sort(order)
    return (
        f"flagged groups without a snap: {groups.height} of {int(drives['flagged'].sum())}, "
        f"{groups.filter(touchdown).height} of them ending in a touchdown\n"
        f"by the play types they hold:\n{by_plays}\nby drive result:\n{by_result}\n"
        f"by season:\n{by_season}\nevery group:\n{listing}\n"
        f"with a snap test that leaves out a scramble filed as a run play: {narrower.height} "
        f"groups, the scramble being the only snap of these:\n{scramble_only}\n"
    )


def _try_facts(plays: pl.DataFrame) -> str:
    """Return the drives the replaced rule counted only through their try, by drive result."""
    in_zone = (pl.col("yardline_100") <= _YARD_LINE).fill_null(value=False)
    wiped = (pl.col("play_type") == "no_play").fill_null(value=False)
    on_try = pl.col("desc").str.contains(r"Kick formation|TWO-POINT").fill_null(value=False)
    is_try = (
        (pl.col("play_type") == "extra_point").fill_null(value=False)
        | _set("two_point_attempt")
        | (wiped & on_try)
    )
    drives = plays.group_by(_DRIVE, maintain_order=True).agg(
        pl.col("fixed_drive_result").first().alias("result"),
        _set("drive_inside20").any().alias("flagged"),
        (in_zone & is_try).any().alias("try_in_zone"),
        (in_zone & ~is_try).any().alias("other_in_zone"),
        is_try.all().alias("only_the_try"),
        _snap().any().alias("snap"),
    )
    try_only = drives.filter(pl.col("try_in_zone") & ~pl.col("other_in_zone"))
    table = (
        try_only.group_by("result")
        .agg(
            pl.len().alias("drives"),
            pl.col("flagged").sum().alias("flagged"),
            (pl.col("flagged") & ~pl.col("snap")).sum().alias("flagged_without_a_snap"),
            (~pl.col("flagged")).sum().alias("not_flagged"),
            pl.col("only_the_try").sum().alias("only_the_try"),
        )
        .sort(["drives", "result"], descending=[True, False], nulls_last=True)
    )
    return (
        f"drives whose only plays at or inside the {_YARD_LINE} belong to the try: "
        f"{try_only.height}, of which nflverse flags {int(try_only['flagged'].sum())}\n"
        f"by drive result:\n{table}\n"
    )


def _attributed(name: str, theirs: str, no_snap: str) -> tuple[pl.Expr, pl.Expr]:
    """Return whether a count differs from the source's, and whether its groups explain it."""
    differs = pl.col(name).ne_missing(pl.col(theirs))
    explained = differs & (pl.col(name) + pl.col(no_snap)).eq_missing(pl.col(theirs))
    return differs, explained


def _comparison(both: pl.DataFrame, joined: pl.DataFrame, digest: str, span: str) -> str:
    """Return the per-team-game comparison with the published logs, as text."""
    text = (
        f"source game logs {span} SHA-256 {digest}\n"
        f"team-games: both {both.height}, "
        f"only ours {joined.filter(pl.col('source_trips').is_null()).height}, "
        f"only the source {joined.filter(pl.col('red_zone_trips').is_null()).height}\n"
    )
    unexplained: dict[str, pl.DataFrame] = {}
    for name, theirs, no_snap in _PAIRS:
        differs, explained = _attributed(name, theirs, no_snap)
        counted = both.select(differs.sum().alias("differ"), explained.sum().alias("explained"))
        left = both.filter(differs & ~explained)
        unexplained[name] = left
        text += (
            f"{name} differs from the source in {counted['differ'].item()} team-games; the "
            f"groups without a snap explain {counted['explained'].item()} exactly, "
            f"{left.height} unexplained\n"
        )
    for name, theirs, no_snap in _NAIVE_PAIRS:
        differs, explained = _attributed(name, theirs, no_snap)
        left = both.filter(differs & ~explained).height
        text += (
            f"{name}, with the source's rate read over all its trips: {left} team-games "
            f"unexplained\n"
        )
    by_season = (
        both.group_by("season", maintain_order=True)
        .agg(
            pl.len().alias("team_games"),
            *(
                _attributed(name, theirs, no_snap)[0].sum().alias(f"{name}_differ")
                for name, theirs, no_snap in _PAIRS
            ),
            *(
                (_attributed(name, theirs, no_snap)[0] & ~_attributed(name, theirs, no_snap)[1])
                .sum()
                .alias(f"{name}_unexplained")
                for name, theirs, no_snap in _PAIRS[:2]
            ),
            pl.col("red_zone_trips").sum().alias("trips"),
            pl.col("source_trips").sum().alias("source_trips"),
            pl.col("no_snap_trips").sum().alias("no_snap_trips"),
            pl.col("red_zone_td_drives").sum().alias("td_drives"),
            pl.col("source_td_drives").sum().alias("source_td_drives"),
            pl.col("no_snap_td_drives").sum().alias("no_snap_td_drives"),
        )
        .sort("season")
    )
    totals = both.select(
        pl.col("red_zone_trips").sum().alias("trips"),
        pl.col("source_trips").sum().alias("source_trips"),
        pl.col("no_snap_trips").sum().alias("no_snap_trips"),
        pl.col("red_zone_td_drives").sum().alias("td_drives"),
        pl.col("source_td_drives").sum().alias("source_td_drives"),
        pl.col("no_snap_td_drives").sum().alias("no_snap_td_drives"),
        pl.col("no_result_trips").sum().alias("no_result_trips"),
    ).row(0, named=True)
    text += (
        f"trips {totals['trips']}, source {totals['source_trips']}, of which flagged groups "
        f"without a snap {totals['no_snap_trips']} and trips without a recorded drive result "
        f"{totals['no_result_trips']}; touchdown drives {totals['td_drives']}, source "
        f"{totals['source_td_drives']}, of which in groups without a snap "
        f"{totals['no_snap_td_drives']}; league touchdown rate per trip "
        f"{totals['td_drives'] / totals['trips']:.4f}\n{by_season}\n"
    )
    for name, frame in unexplained.items():
        text += f"{name}, team-games the groups without a snap leave unexplained:\n"
        text += f"{frame.sort(_KEYS)}\n"
    for touchdowns, trips, published, no_snap, without_result in _RATES:
        ours = pl.when(pl.col(trips) > 0).then(pl.col(touchdowns) / pl.col(trips))
        rates = both.filter(
            ((ours - pl.col(published)).abs() > _RATE_TOLERANCE).fill_null(value=False)
            | (ours.is_null() != pl.col(published).is_null())
        )
        held = (pl.col(no_snap) > 0) | (pl.col(without_result) > 0)
        left = rates.filter(~held).select(*_KEYS, trips, touchdowns, ours.alias("rate"), published)
        text += (
            f"{touchdowns} / {trips} differs from the published {published} in {rates.height} "
            f"team-games, {rates.filter(held).height} of them holding a group without a snap or "
            f"a trip without a recorded result; the rest:\n{left.sort(_KEYS)}\n"
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
            *(pl.col(name).cast(pl.Int64) for name, _, _ in _PAIRS),
        )
    )
    source, digest = _source_logs(args.sos_data, seasons)
    plays = _cached_plays(args.cache_dir, seasons)
    drives = _drives(plays)
    joined = _with_source(ours, source, _group_counts(drives))
    both = joined.filter(
        pl.col("red_zone_trips").is_not_null() & pl.col("source_trips").is_not_null()
    )
    no_result_by_season = (
        drives.filter(pl.col("flagged") & pl.col("no_result"))
        .group_by("season")
        .agg(pl.len().alias("trips_without_a_result"))
        .sort("season")
    )
    span = f"{args.first_season}-{args.last_season}"
    with _TABLES:
        sys.stdout.write(
            f"team-games: ours {ours.height}, source {source.height}\n"
            + _comparison(both, joined, digest, span)
            + f"\n## Flagged drive groups without an offensive snap\n{_no_snap_facts(drives)}"
            + f"\n## drive_inside20 on the plays of a drive\n{_drive_flag_facts(plays, drives)}"
            + "\n## Flagged groups whose drive has no recorded result, by season\n"
            + f"{no_result_by_season}\n"
            + f"\n## The try under the replaced rule\n{_try_facts(plays)}"
        )


if __name__ == "__main__":
    main()
