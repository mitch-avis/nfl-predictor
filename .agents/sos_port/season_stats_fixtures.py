"""Write the season statistics parity fixtures the tests read, cut from the snapshot and the files.

Run from the repository root:

    .venv/bin/python .agents/sos_port/season_stats_fixtures.py [--source PATH] [--snapshot DIR]
        [--manifest FILE] [--out-dir DIR]

Two kinds of fixture go to ``tests/fixtures/sos_season_stats_parity/`` (``--out-dir``), each a
Parquet file with its source and SHA-256 in ``manifest.json`` and ``PROVENANCE.md`` there:

- Game inputs and outputs: five games chosen for nflverse's quirks (Washington's code and the
  Rams' ``LA`` in 2024, scrambles coded as runs in 2004, Jacksonville's 2001 home game credited
  to the visitor, and a 1999 game), as nflreadpy returns them from the snapshot of its disk cache
  that ``nfl-sos-ratings`` built from (``season_stats_parity.SnapshotDownloads``, every file
  checked against the manifest): the play-by-play columns the season statistics read, the weekly
  player and team stats, the schedule, snap counts, and the players file and rosters of the
  players in those games; and the published team and quarterback game logs and win-probability
  bins of those games, which a build from those inputs must reproduce.
- Season tables from published logs: for 1999 and 2024, a column subset of the published team and
  quarterback game logs (every row), the season's regular-season schedule from the snapshot, the
  published team ratings and quarterback rating columns, and the same columns of the published
  season tables, which the season tables built from those logs must reproduce.

Nothing is written outside ``--out-dir``; ``nfl-sos-ratings`` and the snapshot are only read.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "1")

import argparse
import json
import sys
import textwrap
from pathlib import Path

import nflreadpy as nfl
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from season_stats_parity import (  # a sibling module, importable once its dir is on the path
    PUBLISHED_LOG_KINDS,
    REPO_ROOT,
    SOURCE_COMMIT,
    SnapshotDownloads,
    default_source,
    read_manifest,
    sha256_file,
    source_head,
)

from nfl_predictor import constants
from nfl_predictor.season_stats import game_logs, published_logs, sources
from nfl_predictor.season_stats import season as season_tables

OUT_DIR = REPO_ROOT / "tests" / "fixtures" / "sos_season_stats_parity"
GAMES: dict[int, tuple[str, ...]] = {
    1999: ("1999_01_DET_SEA",),
    2001: ("2001_01_PIT_JAX",),
    2004: ("2004_01_ATL_SF",),
    2024: ("2024_01_WAS_TB", "2024_01_LA_DET"),
}
SEASON_TABLE_SEASONS = (1999, 2024)
# The season whose fixture game needs the roster repair (Jacksonville's 2001 home game).
ROSTER_REPAIR_SEASON = 2001
# The provenance lists a fixture's columns unless it keeps more than this many.
MAX_LISTED_COLUMNS = 40
# The published team-log columns the season-table fixtures keep: the game's identity and score,
# counts and maxima, rates whose parts the published logs determine, and the defensive context
# the quarterback opponent profiles average.
TEAM_LOG_COLUMNS = (
    "game_id",
    "week",
    "team",
    "opponent_team",
    "is_home",
    "points_for",
    "points_allowed",
    "point_margin",
    "win_value",
    "turnover_margin",
    "offensive_snaps",
    "defensive_snaps",
    "passing_yards",
    "rushing_yards",
    "passing_epa",
    "passing_yards_per_offensive_snap",
    "passing_epa_per_offensive_snap",
    "completions",
    "attempts",
    "completion_pct",
    "completions_allowed",
    "attempts_faced",
    "completion_pct_allowed",
    "third_down_conversions",
    "third_down_attempts",
    "third_down_pct",
    "third_down_pct_allowed",
    "longest_pass",
    "def_sacks",
    "def_interceptions",
    "def_pass_defended",
    "def_tackles_for_loss",
    "def_qb_hits",
)
QB_LOG_COLUMNS = (
    "game_id",
    "week",
    "team",
    "opponent_team",
    "qb_id",
    "qb_name",
    "qb_dropbacks",
    "qb_offense_snaps",
    "qb_attempts",
    "qb_completions",
    "qb_pass_yards",
    "qb_pass_touchdowns",
    "qb_interceptions",
    "qb_sacks",
    "qb_sack_yards_lost",
    "qb_passing_epa",
    "qb_spike_epa",
    "qb_epa_per_dropback",
    "qb_any_a",
    "qb_passer_rating",
    "qb_completion_pct",
    "qb_fourth_quarter_comeback",
    "qb_game_winning_drive",
    "qb_carries",
    "qb_rushing_yards",
    *game_logs.QB_LOG_CONTEXT_COLUMNS[1:],
)
PLAYER_ID_COLUMNS = (
    "passer_player_id",
    "rusher_player_id",
    "receiver_player_id",
    *constants.PBP_PLAYER_TEAM_ID_COLUMNS.values(),
)


class Writer:
    """Write fixtures and record each one's source, cut, and hashes."""

    def __init__(self, out_dir: Path, source: Path) -> None:
        """Start an empty record of the fixtures written to ``out_dir``."""
        self.out_dir = out_dir
        self.source = source
        self.records: list[dict[str, object]] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        frame: pl.DataFrame,
        name: str,
        origin: str | list[str],
        origin_sha256: str | list[str],
        rows: str,
    ) -> pl.DataFrame:
        """Write ``frame`` as ``name`` and record where it was cut from (one file or several)."""
        path = self.out_dir / name
        frame.write_parquet(path, statistics=False)
        self.records.append(
            {
                "fixture": name,
                "source": origin,
                "source_sha256": origin_sha256,
                "columns": "every column" if frame.width > MAX_LISTED_COLUMNS else frame.columns,
                "rows": rows,
                "row_count": frame.height,
                "sha256": sha256_file(path),
            }
        )
        return frame

    def published(self, season: int, kind: str) -> tuple[pl.DataFrame, str, str]:
        """Return a published file, its name under the source, and its SHA-256."""
        path = self.source / "data" / f"{season}_{kind}.parquet"
        return pl.read_parquet(path), f"data/{path.name}", sha256_file(path)


def _snapshot_file(snapshot: SnapshotDownloads, label: str) -> tuple[str, str]:
    """Return the snapshot file nflreadpy served last for ``label``, by name and SHA-256."""
    for name, record in snapshot.served.items():
        if record["url"].endswith(label):
            return f"snapshot/{name}", str(record["sha256"])
    msg = f"no snapshot file served for {label}"
    raise LookupError(msg)


def write_game_fixtures(writer: Writer, snapshot: SnapshotDownloads) -> None:
    """Write the inputs of the chosen games season by season, and their published outputs.

    The schedule rows and each kind of published output are written once for every game (one
    source file each season), since a wide file of few rows is mostly per-column overhead.
    """
    schedules: list[pl.DataFrame] = []
    outputs: dict[str, list[tuple[pl.DataFrame, str, str]]] = {
        kind: [] for kind in ("team_game_logs", "qb_game_logs", "team_wp_bins", "qb_wp_bins")
    }
    players = nfl.load_players()
    players_origin = _snapshot_file(snapshot, "players/players.parquet")
    for season, games in GAMES.items():
        game_filter = pl.col("game_id").is_in(list(games))
        pbp = nfl.load_pbp(seasons=[season]).filter(game_filter & (pl.col("season_type") == "REG"))
        pbp = pbp.select([c for c in sources.SEASON_STATS_PBP_COLUMNS if c in pbp.columns])
        origin = _snapshot_file(snapshot, f"play_by_play_{season}.parquet")
        writer.write(
            pbp, f"{season}_games_pbp.parquet", origin[0], origin[1], f"games {', '.join(games)}"
        )

        stats = nfl.load_player_stats(seasons=[season], summary_level="week").filter(game_filter)
        stats = stats.select(
            [c for c in constants.NFLREADPY_PLAYER_STATS_COLUMNS if c in stats.columns]
        )
        origin = _snapshot_file(snapshot, f"stats_player_week_{season}.parquet")
        writer.write(
            stats, f"{season}_games_player_stats.parquet", *origin, f"games {', '.join(games)}"
        )

        team_stats = nfl.load_team_stats(seasons=[season], summary_level="week").filter(game_filter)
        team_stats = team_stats.select(
            [c for c in sources.OFFICIAL_TEAM_STATS.columns if c in team_stats.columns]
        )
        origin = _snapshot_file(snapshot, f"stats_team_week_{season}.parquet")
        writer.write(
            team_stats, f"{season}_games_team_stats.parquet", *origin, f"games {', '.join(games)}"
        )

        schedules.append(nfl.load_schedules(seasons=[season]).filter(game_filter))

        ids = set()
        for column in PLAYER_ID_COLUMNS:
            if column in pbp.columns:
                ids |= set(pbp.get_column(column).drop_nulls().to_list())
        ids |= set(stats.get_column("player_id").drop_nulls().to_list())
        pfr_ids: set[str] = set()
        if season >= constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON:
            snaps = nfl.load_snap_counts(seasons=[season]).filter(game_filter)
            origin = _snapshot_file(snapshot, f"snap_counts_{season}.parquet")
            writer.write(
                snaps, f"{season}_games_snap_counts.parquet", *origin, f"games {', '.join(games)}"
            )
            pfr_ids = set(snaps.get_column("pfr_player_id").drop_nulls().to_list())
        player_rows = players.filter(
            pl.col("gsis_id").is_in(sorted(ids)) | pl.col("pfr_id").is_in(sorted(pfr_ids))
        ).select(constants.NFLREADPY_PLAYERS_COLUMNS)
        writer.write(
            player_rows,
            f"{season}_games_players.parquet",
            *players_origin,
            "players in those games",
        )
        ids |= set(player_rows.get_column("gsis_id").drop_nulls().to_list())
        if season >= constants.NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON:
            weekly = nfl.load_rosters_weekly(seasons=[season]).filter(
                pl.col("gsis_id").is_in(sorted(ids))
            )
            weekly = weekly.select(
                [c for c in constants.NFLREADPY_ROSTER_COLUMNS if c in weekly.columns]
            )
            origin = _snapshot_file(snapshot, f"roster_weekly_{season}.parquet")
            writer.write(
                weekly, f"{season}_games_weekly_rosters.parquet", *origin, "players in those games"
            )
        if season == ROSTER_REPAIR_SEASON:
            rosters = nfl.load_rosters(seasons=[season]).filter(
                pl.col("gsis_id").is_in(sorted(ids))
            )
            rosters = rosters.select(
                [c for c in constants.NFLREADPY_ROSTER_COLUMNS if c in rosters.columns]
            )
            origin = _snapshot_file(snapshot, f"roster_{season}.parquet")
            writer.write(
                rosters, f"{season}_games_rosters.parquet", *origin, "players in those games"
            )

        for kind, cuts in outputs.items():
            frame, origin_name, origin_hash = writer.published(season, kind)
            cuts.append((frame.filter(game_filter), origin_name, origin_hash))

    every_game = "games " + ", ".join(game for games in GAMES.values() for game in games)
    origin = _snapshot_file(snapshot, "schedules/games.parquet")
    writer.write(pl.concat(schedules), "games_schedule.parquet", *origin, every_game)
    for kind, cuts in outputs.items():
        writer.write(
            pl.concat([cut for cut, _, _ in cuts], how="vertical_relaxed"),
            f"games_{kind}.parquet",
            [name for _, name, _ in cuts],
            [digest for _, _, digest in cuts],
            every_game,
        )


def _tables_from_logs(
    team_logs: pl.DataFrame,
    qb_logs: pl.DataFrame,
    schedule: pl.DataFrame,
    ratings: pl.DataFrame,
    qb_ratings: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Return the season tables built from published logs, as the parity test builds them."""
    relabel = {"WAS": "WSH"}
    team_logs = team_logs.with_columns(pl.col("team", "opponent_team").replace(relabel))
    qb_logs = qb_logs.with_columns(pl.col("team", "opponent_team").replace(relabel))
    ratings = ratings.with_columns(pl.col("team").replace(relabel))
    games = season_tables.SeasonGames(
        team_games=published_logs.team_stat_rows_from_published(team_logs),
        qb_games=published_logs.qb_stat_rows_from_published(qb_logs),
        schedule=season_tables.played_schedule(
            sources.prepare_schedule(
                sources.normalize_team_codes(schedule, ("home_team", "away_team"))
            )
        ),
    )
    return season_tables.build_season_tables(
        games, pl.DataFrame(), team_ratings=ratings, qb_ratings=qb_ratings
    ).tables()


def write_season_table_fixtures(writer: Writer, snapshot: SnapshotDownloads) -> None:
    """Write the published logs, schedule, ratings, and season-table columns of two seasons.

    Each published season table is cut to the columns a build from the kept log columns gives.
    """
    for season in SEASON_TABLE_SEASONS:
        logs, origin_name, origin_hash = writer.published(season, "team_game_logs")
        team_logs = writer.write(
            logs.select(TEAM_LOG_COLUMNS),
            f"{season}_season_team_game_logs.parquet",
            origin_name,
            origin_hash,
            "every row",
        )
        qb_logs, origin_name, origin_hash = writer.published(season, "qb_game_logs")
        qb_logs = writer.write(
            qb_logs.select(QB_LOG_COLUMNS),
            f"{season}_season_qb_game_logs.parquet",
            origin_name,
            origin_hash,
            "every row",
        )
        schedule = nfl.load_schedules(seasons=[season]).filter(pl.col("game_type") == "REG")
        origin = _snapshot_file(snapshot, "schedules/games.parquet")
        schedule = writer.write(
            schedule.select("game_id", "game_type", *sources.SCHEDULE_COLUMNS[1:]),
            f"{season}_season_schedule.parquet",
            *origin,
            "regular-season games",
        )
        ratings, origin_name, origin_hash = writer.published(season, "ratings")
        ratings = writer.write(
            ratings, f"{season}_season_ratings.parquet", origin_name, origin_hash, "every row"
        )
        qb_combined, origin_name, origin_hash = writer.published(season, "qb_combined")
        qb_ratings = writer.write(
            qb_combined.select(season_tables.QB_RATING_INPUT_COLUMNS),
            f"{season}_season_qb_ratings.parquet",
            origin_name,
            origin_hash,
            "every row",
        )
        built = _tables_from_logs(team_logs, qb_logs, schedule, ratings, qb_ratings)
        for kind in PUBLISHED_LOG_KINDS:
            frame, origin_name, origin_hash = writer.published(season, kind)
            writer.write(
                frame.select(built[kind].columns),
                f"{season}_season_{kind}.parquet",
                origin_name,
                origin_hash,
                "every row",
            )


def _provenance(records: list[dict[str, object]], head: str) -> str:
    """Return ``PROVENANCE.md`` for the written fixtures."""
    intro = (
        "Generated by `.agents/sos_port/season_stats_fixtures.py`; do not edit by hand. Each file "
        "is cut from a published `nfl-sos-ratings` Parquet file (`data/...`, read at the "
        f"checkout's commit `{head}`) or from the snapshot of nflreadpy's disk cache that "
        "project's files were built from (`snapshot/...`, checked against its SHA-256 manifest). "
        "`manifest.json` holds the same records with full hashes, and "
        "`tests/test_season_stats_published_parity.py` checks every fixture against its hash. "
        "The SHA-256 columns show the first 12 characters."
    )
    lines = [
        "# Season statistics parity fixtures",
        "",
        textwrap.fill(intro, width=100, break_on_hyphens=False),
        "",
        "| fixture | source (SHA-256, first 12) | rows kept | rows | SHA-256 |",
        "| --- | --- | --- | ---: | --- |",
    ]

    def sources_cell(record: dict[str, object]) -> str:
        names = record["source"] if isinstance(record["source"], list) else [record["source"]]
        hashes = record["source_sha256"]
        hashes = hashes if isinstance(hashes, list) else [hashes]
        return "<br>".join(
            f"`{name}` (`{str(digest)[:12]}`)" for name, digest in zip(names, hashes, strict=True)
        )

    lines.extend(
        f"| `{r['fixture']}` | {sources_cell(r)} | {r['rows']} | {r['row_count']} |"
        f" `{str(r['sha256'])[:12]}` |"
        for r in records
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write every fixture, the manifest, and the provenance note."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path.home() / "scratch" / "nflreadpy_cache_snapshot_20261008",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path.home() / "scratch" / "nflreadpy_cache_snapshot_20261008.sha256",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    head = source_head(args.source)
    if head != SOURCE_COMMIT:
        msg = f"{args.source} is at {head}, not the frozen commit {SOURCE_COMMIT}"
        raise SystemExit(msg)
    snapshot = SnapshotDownloads(args.snapshot, read_manifest(args.manifest))
    snapshot.install()
    writer = Writer(args.out_dir, args.source)
    write_game_fixtures(writer, snapshot)
    write_season_table_fixtures(writer, snapshot)
    manifest = {
        "source_repository": "nfl-sos-ratings",
        "source_commit": head,
        "snapshot": "nflreadpy_cache_snapshot_20261008 (SHA-256 manifest beside it)",
        "generator": ".agents/sos_port/season_stats_fixtures.py",
        "games": {str(season): list(games) for season, games in GAMES.items()},
        "files": writer.records,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "PROVENANCE.md").write_text(_provenance(writer.records, head), encoding="utf-8")
    total = sum((args.out_dir / str(r["fixture"])).stat().st_size for r in writer.records)
    sys.stderr.write(f"wrote {len(writer.records)} fixtures, {total:,} bytes\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
