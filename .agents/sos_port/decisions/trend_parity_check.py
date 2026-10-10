"""Build one week as played and as unplayed through the ETL and compare each game's two rows.

Run from the repository root with the worktree's environment:

    .venv/bin/python .agents/sos_port/decisions/trend_parity_check.py \
        [--source ~/scratch/step6/data] [--nflreadpy-cache DIR] [--season 2025] [--week 18] \
        [--base COMMIT]

The source data directory and the nflreadpy cache directory are only read. The script copies
the inputs a build of ``--season`` reads into a new temporary directory twice: that season's
and the previous season's schedule, team stats and play-by-play (from ``--nflreadpy-cache`` when
the file is there, else from the source's ``cache/nflreadpy``), the source's nfelo cache, both
seasons' TeamRankings files, ``qb_elos.csv`` and the quarterback identity file. In the second
copy the week is made unplayed the way an upcoming week looks before kickoff: the schedule's
scores, result, total and overtime are blanked, and the week's team-stat rows and plays are
removed. The ELO file, the TeamRankings files and the nfelo lines stay as they are, because an
upcoming week has them before kickoff (the ETL reads only the pre-game ELO columns).

This branch's ETL requests play-by-play columns that caches written before them lack, and it
downloads such a file again for a historical season rather than read it. The script stops when a
copied play-by-play file is stale that way; pass ``--nflreadpy-cache`` with a current cache, or
``--download-stale-pbp`` to download the stale seasons into the temporary copy first (the only
network use).

Each copy is built in its own process through ``data_collection.collect_all_data`` with the
default options, ``constants.DATA_PATH`` and the two cache directories pointed at the copy, and
every network connection refused, so the loaders read their caches as on an offline machine.
With ``--base`` both builds use the package source as of that commit (extracted with ``git
archive`` and put ahead of the installed package with ``PYTHONPATH``) instead of the working
tree's. Nothing is written outside the temporary directory, which is removed at the end.

It prints:

1. the inputs' provenance (each copied nflreadpy file's SHA-256 prefix) and what the scenario
   changed in the second copy;
2. whether the played build reproduces the source's published trend values for the week's games
   (``all_data_ml.csv``), so the temporary build is the ETL the published data came from;
3. every column whose value differs between a game's completed row and its upcoming row, with
   the number of games that differ and the largest difference;
4. each game's four team-stat trend values in both builds;
5. for every season in the source's ``all_data_ml.csv``, whether each completed regular-season
   row's ``scoring_margin_4wk_trend`` equals the trend from the team's games strictly before the
   row's week or the trend that also includes the row's own game (scores from the schedule
   cache);
6. for the unplayed games of the latest season in the source's ``all_data_ml.csv``, per week,
   how many carry a team ELO trend of 0.0 on both sides and no quarterback ELO trend on either
   side: the weeks ``qb_elos.csv`` does not list yet fall back that way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

DEFAULT_SOURCE = Path.home() / "scratch" / "step6" / "data"
WINDOW = 4
TREND_STATS = ("scoring_margin", "turnover_margin")
POST_GAME_SCHEDULE_COLUMNS = ("away_score", "home_score", "result", "total", "overtime")
TOP_LEVEL_INPUTS = ("qb_elos.csv", "meta_data.csv", "nfl_elo.csv", "nfl_lines.csv")
REQUESTED_COLUMNS_KEY = "nfl_predictor.requested_columns"
# The first season with an nflverse schedule this repository caches.
FIRST_SCHEDULE_SEASON = 1999
GAME_KEY = "game_id"
LOG_TAIL = 6000


def _refuse_connection(*_args: object, **_kwargs: object) -> None:
    msg = "network refused: the trend parity scripts read cached inputs only"
    raise OSError(msg)


def _refuse_lookup(*_args: object, **_kwargs: object) -> None:
    msg = "network refused: the trend parity scripts read cached inputs only"
    raise socket.gaierror(msg)


def _refuse_network() -> None:
    """Refuse every connection and host lookup in this process."""
    socket.socket.connect = _refuse_connection
    socket.create_connection = _refuse_connection
    socket.getaddrinfo = _refuse_lookup


def build(data_dir: Path, seasons: list[int], out_csv: Path) -> None:
    """Build ``seasons`` from the inputs under ``data_dir``; write the ML rows to ``out_csv``.

    Runs in its own process: the constants are pointed at ``data_dir`` before any other module
    of the package is imported. Writes ``<out_csv>.json`` with the package path that built it.
    """
    _refuse_network()
    from nfl_predictor import constants

    constants.DATA_PATH = data_dir
    constants.NFLREADPY_CACHE_DIR = data_dir / "cache" / "nflreadpy"
    constants.NFELO_CACHE_DIR = data_dir / "cache" / "nfelo"

    import nfl_predictor
    from nfl_predictor import data_collection

    config = data_collection.DataCollectionConfig(
        enable_timing=False,
        enable_debug=False,
        force_refresh_nflreadpy=False,
        min_season=min(seasons),
        max_season=max(seasons),
    )
    games = data_collection.collect_all_data(seasons, config=config)
    games.write_csv(out_csv)
    out_csv.with_suffix(".json").write_text(
        json.dumps({"package": nfl_predictor.__file__, "seasons": seasons}), encoding="utf-8"
    )


def sha_prefix(path: Path) -> str:
    """Return the first 16 hex digits of a file's SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _first_existing(*candidates: Path) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def copy_inputs(source: Path, nflreadpy_cache: Path, dest: Path, years: list[int]) -> list[str]:
    """Copy the inputs a build of ``years`` reads into ``dest``; return each file's provenance.

    A season's nflreadpy files come from ``nflreadpy_cache`` when it has them, else from the
    source's ``cache/nflreadpy``. Every season's schedule and play-by-play from the first one
    through ``max(years)`` is copied, because the pick-time lines' moneyline maps read every
    earlier schedule and the quarterback features every earlier season's plays; team stats are
    copied for ``years`` only.
    """
    cache = dest / "cache" / "nflreadpy"
    cache.mkdir(parents=True)
    provenance: list[str] = []
    names = [
        name
        for year in range(FIRST_SCHEDULE_SEASON, max(years) + 1)
        for name in (f"schedule_{year}.parquet", f"pbp_{year}_reg.parquet")
    ] + [f"team_stats_{year}_reg.parquet" for year in years]
    for name in names:
        origin = _first_existing(nflreadpy_cache / name, source / "cache" / "nflreadpy" / name)
        if origin is None:
            provenance.append(f"{name}  missing")
            continue
        shutil.copy2(origin, cache / name)
        provenance.append(f"{name}  {sha_prefix(origin)}  from {origin.parent}")
    for year in years:
        if (source / str(year)).is_dir():
            shutil.copytree(source / str(year), dest / str(year))
    shutil.copytree(source / "cache" / "nfelo", dest / "cache" / "nfelo")
    for name in TOP_LEVEL_INPUTS:
        if (source / name).exists():
            shutil.copy2(source / name, dest / name)
            provenance.append(f"{name}  {sha_prefix(source / name)}  from {source}")
    return provenance


def stale_pbp_years(dest: Path, years: list[int], current_season: int) -> list[int]:
    """Return the historical seasons whose copied play-by-play the loader would download again."""
    from nfl_predictor import constants

    cache = dest / "cache" / "nflreadpy"
    stale: list[int] = []
    for year in years:
        path = cache / f"pbp_{year}_reg.parquet"
        if year >= current_season or not path.exists():
            continue
        schema = pl.read_parquet_schema(path)
        recorded = pl.read_parquet_metadata(path).get(REQUESTED_COLUMNS_KEY)
        requested_then = set(json.loads(recorded)) if recorded else set()
        missing = {c for c in constants.PBP_COLUMNS if c not in schema} - requested_then
        if missing:
            stale.append(year)
    return stale


def download_pbp(dest: Path, years: list[int], current_season: int) -> None:
    """Download the seasons' play-by-play into ``dest``'s cache the way the ETL would."""
    from nfl_predictor.utils import polars_utils

    polars_utils.load_pbp(
        years,
        cache_dir=dest / "cache" / "nflreadpy",
        force_refresh=True,
        current_season=current_season,
    )


def prepare(args: argparse.Namespace, dest: Path, years: list[int]) -> list[str]:
    """Copy the inputs into ``dest`` and make sure no play-by-play file is stale."""
    from nfl_predictor.utils import clock

    provenance = copy_inputs(args.source, args.nflreadpy_cache, dest, years)
    current_season = clock.nfl_season(clock.local_today())
    stale = stale_pbp_years(
        dest, list(range(FIRST_SCHEDULE_SEASON, max(years) + 1)), current_season
    )
    if stale and not args.download_stale_pbp:
        msg = (
            f"the play-by-play for {stale} lacks columns this branch requests, so the ETL would "
            "download it again; pass --nflreadpy-cache with a current cache or "
            "--download-stale-pbp"
        )
        raise SystemExit(msg)
    if stale:
        download_pbp(dest, stale, current_season)
        provenance.append(f"downloaded play-by-play for {stale} into the copy")
    return provenance


def base_source(commit: str, dest: Path) -> Path:
    """Extract the package source at ``commit`` into ``dest``; return its ``src`` directory."""
    dest.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit, "src"], check=True, capture_output=True
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)
    return dest / "src"


def run_build(
    data_dir: Path,
    seasons: list[int],
    *,
    out_csv: Path | None = None,
    pythonpath: Path | None = None,
) -> tuple[pl.DataFrame, str]:
    """Build ``seasons`` from ``data_dir`` in a fresh process; return its rows and package path.

    The rows are written to ``out_csv`` (default ``data_dir/built.csv``) and its log beside it.
    ``pythonpath`` puts another copy of the package's source ahead of the installed one.
    """
    out_csv = out_csv or data_dir / "built.csv"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--build",
        str(data_dir),
        "--min-season",
        str(min(seasons)),
        "--max-season",
        str(max(seasons)),
        "--out",
        str(out_csv),
    ]
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    out_csv.with_suffix(".log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        sys.stderr.write((result.stdout + result.stderr)[-LOG_TAIL:])
        msg = f"the build in {data_dir} failed with exit code {result.returncode}"
        raise RuntimeError(msg)
    record = json.loads(out_csv.with_suffix(".json").read_text(encoding="utf-8"))
    return pl.read_csv(out_csv, infer_schema_length=None), str(record["package"])


def trend_columns() -> list[str]:
    """Return the team-stat trend columns of a game row, the differences included."""
    return [
        *[f"{side}_{stat}_4wk_trend" for stat in TREND_STATS for side in ("away", "home")],
        *[f"{stat}_4wk_trend_diff" for stat in TREND_STATS],
    ]


def _numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or dtype == pl.Boolean


def column_differences(left: pl.DataFrame, right: pl.DataFrame, key: list[str]) -> pl.DataFrame:
    """Return, per shared column, how many rows differ between the two frames and by how much."""
    paired = left.join(right, on=key, how="inner", suffix="__right")
    rows: list[dict[str, object]] = []
    for column in left.columns:
        if column in key or f"{column}__right" not in paired.columns:
            continue
        a_raw = paired.get_column(column)
        b_raw = paired.get_column(f"{column}__right")
        if _numeric(a_raw.dtype) and _numeric(b_raw.dtype):
            a = a_raw.cast(pl.Float64)
            b = b_raw.cast(pl.Float64)
            gap = (a - b).abs()
            differs = (a.is_null() != b.is_null()) | (gap.fill_null(0.0) > 1e-9)
            largest = gap.filter(differs & gap.is_not_null()).max()
        else:
            a_text = a_raw.cast(pl.Utf8).fill_null("<null>")
            differs = a_text != b_raw.cast(pl.Utf8).fill_null("<null>")
            largest = None
        count = int(differs.sum())
        if count:
            rows.append({"column": column, "rows_differing": count, "largest_gap": largest})
    return pl.DataFrame(
        rows,
        schema={"column": pl.Utf8, "rows_differing": pl.Int64, "largest_gap": pl.Float64},
    ).sort("column")


def _faithfulness(source: Path, played: pl.DataFrame) -> str:
    """Compare the played build's week trends with the source's published rows."""
    published_path = source / "all_data_ml.csv"
    if not published_path.exists():
        return "   no all_data_ml.csv in the source; skipped"
    columns = [GAME_KEY, *trend_columns()]
    published = pl.read_csv(published_path, columns=columns, infer_schema_length=None)
    paired = played.select(columns).join(published, on=GAME_KEY, suffix="__published")
    lines = [f"   games matched: {paired.height} of {played.height}"]
    for column in trend_columns():
        gap = paired.select((pl.col(column) - pl.col(f"{column}__published")).abs().max()).item()
        lines.append(f"   {column}: largest gap {gap}")
    return "\n".join(lines)


def _strict_and_inclusive_trends(margins: list[float]) -> list[tuple[float, float]]:
    """Return, per game, the trend from earlier games and the trend including the game."""

    def trend(values: list[float]) -> float:
        if not values:
            return 0.0
        recent = values[-WINDOW:]
        return sum(recent) / len(recent) - sum(values) / len(values)

    return [(trend(margins[:i]), trend(margins[: i + 1])) for i in range(len(margins))]


def season_candidates(schedule: pl.DataFrame) -> pl.DataFrame:
    """Return both candidate trends for every team-game of a season's completed games."""
    sides = pl.concat(
        [
            schedule.select(
                GAME_KEY,
                "week",
                pl.col(f"{side}_abbr").alias("team"),
                (pl.col(f"{side}_score") - pl.col(f"{other}_score")).cast(pl.Float64).alias("m"),
                pl.lit(side).alias("side"),
            )
            for side, other in (("away", "home"), ("home", "away"))
        ]
    ).sort("team", "week")
    candidates: list[dict[str, object]] = []
    for _team, games in sides.group_by("team", maintain_order=True):
        pairs = _strict_and_inclusive_trends(games.get_column("m").to_list())
        for row, (strict, inclusive) in zip(games.iter_rows(named=True), pairs, strict=True):
            candidates.append(
                {
                    GAME_KEY: row[GAME_KEY],
                    "side": row["side"],
                    "strict": strict,
                    "inclusive": inclusive,
                }
            )
    return pl.DataFrame(candidates)


def _own_game_check(source: Path) -> str:
    """Report, per season, which candidate the published completed rows match."""
    published_path = source / "all_data_ml.csv"
    if not published_path.exists():
        return "   no all_data_ml.csv in the source; skipped"
    stored_columns = ["away_scoring_margin_4wk_trend", "home_scoring_margin_4wk_trend"]
    published = pl.read_csv(
        published_path,
        columns=[GAME_KEY, "season", "game_type", *stored_columns],
        infer_schema_length=None,
    ).filter(pl.col("game_type") == "REG")
    header = (
        "   season   rows  equal_strictly_before  equal_including_own_game  "
        "rows_where_the_two_differ  of_those_equal_including"
    )
    lines = [header]
    total = [0, 0, 0, 0, 0]
    for season in sorted(published.get_column("season").unique().to_list()):
        schedule_path = source / "cache" / "nflreadpy" / f"schedule_{season}.parquet"
        if not schedule_path.exists():
            continue
        schedule = pl.read_parquet(schedule_path).filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        stored = (
            published.filter(pl.col("season") == season)
            .unpivot(index=GAME_KEY, on=stored_columns, variable_name="side", value_name="stored")
            .with_columns(pl.col("side").str.split("_").list.first())
        )
        joined = stored.join(season_candidates(schedule), on=[GAME_KEY, "side"], how="inner")
        strict_equal = (pl.col("stored") - pl.col("strict")).abs() < 1e-4
        inclusive_equal = (pl.col("stored") - pl.col("inclusive")).abs() < 1e-4
        candidates_differ = (pl.col("strict") - pl.col("inclusive")).abs() > 1e-6
        counts = joined.select(
            pl.len().alias("rows"),
            strict_equal.sum().alias("strict"),
            inclusive_equal.sum().alias("inclusive"),
            candidates_differ.sum().alias("differ"),
            (candidates_differ & inclusive_equal).sum().alias("differ_inclusive"),
        ).row(0)
        total = [a + int(b) for a, b in zip(total, counts, strict=True)]
        lines.append(
            f"   {season}  {counts[0]:>5}  {counts[1]:>21}  {counts[2]:>24}  "
            f"{counts[3]:>25}  {counts[4]:>24}"
        )
    lines.append(
        f"   all   {total[0]:>6}  {total[1]:>21}  {total[2]:>24}  {total[3]:>25}  {total[4]:>24}"
    )
    return "\n".join(lines)


def _elo_fallback(source: Path) -> str:
    """Report, per week of the latest published season, the unplayed games' ELO trends."""
    published_path = source / "all_data_ml.csv"
    if not published_path.exists():
        return "   no all_data_ml.csv in the source; skipped"
    sides = ("away", "home")
    published = pl.read_csv(
        published_path,
        columns=[
            GAME_KEY,
            "season",
            "week",
            "home_score",
            *[f"{side}_elo_4wk_trend" for side in sides],
            *[f"{side}_qb_elo_4wk_trend" for side in sides],
        ],
        infer_schema_length=None,
    )
    latest = published.get_column("season").max()
    unplayed = published.filter((pl.col("season") == latest) & pl.col("home_score").is_null())
    table = (
        unplayed.group_by("week", maintain_order=True)
        .agg(
            pl.len().alias("unplayed_games"),
            pl.all_horizontal(pl.col(f"{side}_elo_4wk_trend") == 0 for side in sides)
            .sum()
            .alias("elo_trend_zero"),
            pl.all_horizontal(pl.col(f"{side}_qb_elo_4wk_trend").is_null() for side in sides)
            .sum()
            .alias("qb_elo_trend_null"),
        )
        .sort("week")
    )
    return f"   season {latest}\n{table}"


def configure_tables() -> None:
    """Print whole tables in Markdown form without shapes or types."""
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options both trend parity scripts share."""
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--nflreadpy-cache",
        type=Path,
        default=None,
        help="Directory to copy the nflreadpy files from first (default: the source's cache).",
    )
    parser.add_argument("--download-stale-pbp", action="store_true")


def resolve_input_arguments(args: argparse.Namespace) -> None:
    """Resolve the shared paths in place."""
    args.source = args.source.expanduser().resolve()
    args.nflreadpy_cache = (
        args.nflreadpy_cache.expanduser().resolve()
        if args.nflreadpy_cache is not None
        else args.source / "cache" / "nflreadpy"
    )


def main() -> None:
    """Run the scenario, or one build when called with ``--build``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_input_arguments(parser)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, default=18)
    parser.add_argument("--base", help="Build with the package source as of this commit.")
    parser.add_argument("--build", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--min-season", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-season", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.build is not None:
        build(args.build, list(range(args.min_season, args.max_season + 1)), args.out)
        return

    resolve_input_arguments(args)
    configure_tables()
    print(f"Source (read only): {args.source}")
    print(f"nflreadpy files first from (read only): {args.nflreadpy_cache}")
    print(f"Scenario: season {args.season}, week {args.week} made unplayed")
    print(f"Code: {args.base or 'the working tree'}")
    years = [args.season - 1, args.season]
    root = Path(tempfile.mkdtemp(prefix="trend-parity-"))
    try:
        played_dir = root / "played"
        upcoming_dir = root / "upcoming"
        provenance = prepare(args, played_dir, years)
        prepare(args, upcoming_dir, years)
        print("\n1. Inputs and what the unplayed copy changes")
        for line in provenance:
            print(f"   {line}")
        for change in _make_week_unplayed(upcoming_dir, args.season, args.week):
            print(f"   unplayed copy: {change}")

        in_week = (pl.col("season") == args.season) & (pl.col("week") == args.week)
        package_source = base_source(args.base, root / "base") if args.base else None
        played_rows, package = run_build(played_dir, [args.season], pythonpath=package_source)
        upcoming_rows, _ = run_build(upcoming_dir, [args.season], pythonpath=package_source)
        played = played_rows.filter(in_week)
        upcoming = upcoming_rows.filter(in_week)
        print(f"   built by: {package}")
        print(f"   built: {played.height} played rows, {upcoming.height} upcoming rows")
        print(
            "   upcoming rows with a score: "
            f"{upcoming.filter(pl.col('home_score').is_not_null()).height}"
        )

        print("\n2. The played build against the source's published rows (trend columns)")
        print(_faithfulness(args.source, played))

        print("\n3. Columns that differ between a game's completed row and its upcoming row")
        print(column_differences(played, upcoming, [GAME_KEY]))

        print("\n4. Each game's team-stat trends, completed row against upcoming row")
        print("   (sc = scoring margin, to = turnover margin; c = completed, u = upcoming)")
        labels = dict(
            zip(trend_columns()[:4], ("away_sc", "home_sc", "away_to", "home_to"), strict=True)
        )
        trend_table = (
            played.select(GAME_KEY, *[pl.col(c).alias(f"{n}_c") for c, n in labels.items()])
            .join(
                upcoming.select(GAME_KEY, *[pl.col(c).alias(f"{n}_u") for c, n in labels.items()]),
                on=GAME_KEY,
            )
            .sort(GAME_KEY)
        )
        print(trend_table)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\n5. Published completed regular-season rows: scoring_margin_4wk_trend candidates")
    print(_own_game_check(args.source))

    print("\n6. Unplayed games of the latest published season: the ELO trends' fallback")
    print(_elo_fallback(args.source))


def _rewrite_keeping_record(path: Path, frame: pl.DataFrame) -> None:
    """Rewrite a cached parquet file, keeping the loader's record of the requested columns."""
    metadata = {
        key: value
        for key, value in pl.read_parquet_metadata(path).items()
        if key == REQUESTED_COLUMNS_KEY
    }
    frame.write_parquet(path, metadata=metadata or None)


def _make_week_unplayed(dest: Path, season: int, week: int) -> list[str]:
    """Blank the week's results in ``dest``'s copies; return what changed."""
    cache = dest / "cache" / "nflreadpy"
    in_week = (pl.col("season") == season) & (pl.col("week") == week)
    changes: list[str] = []

    schedule_path = cache / f"schedule_{season}.parquet"
    schedule = pl.read_parquet(schedule_path)
    _rewrite_keeping_record(
        schedule_path,
        schedule.with_columns(
            pl.when(in_week).then(None).otherwise(pl.col(column)).alias(column)
            for column in POST_GAME_SCHEDULE_COLUMNS
            if column in schedule.columns
        ),
    )
    changes.append(
        f"schedule_{season}.parquet: {schedule.filter(in_week).height} games of week {week} "
        f"with {', '.join(POST_GAME_SCHEDULE_COLUMNS)} blanked"
    )

    for name in (f"team_stats_{season}_reg.parquet", f"pbp_{season}_reg.parquet"):
        path = cache / name
        frame = pl.read_parquet(path)
        kept = frame.filter(~in_week)
        _rewrite_keeping_record(path, kept)
        changes.append(f"{name}: {frame.height - kept.height} rows of week {week} removed")
    return changes


if __name__ == "__main__":
    main()
