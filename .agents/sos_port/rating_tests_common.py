"""Shared pieces of the rating-definition tests: intervals, the season bootstrap, and inputs.

Used by ``qb_plays_test.py``, ``botched_snaps.py`` and ``team_plays_test.py`` beside this file,
the scripts of the pre-registered protocols in ``protocols/``. The bootstrap resamples whole
seasons, as the source's preseason-prior and quarterback protocols specify: the games of one
season share its penalties and prior, and the games of one week share a rating snapshot, so they
are not independent draws.

The inputs come through this repository's loaders from an nflreadpy cache directory, read only:
every season is treated as historical (``CACHE_CURRENT_SEASON``), so a cached file is read and
never refreshed, and :func:`block_downloads` makes a cache miss fail instead of downloading.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from importlib import metadata
from typing import TYPE_CHECKING, Any

import nflreadpy as nfl
import numpy as np
import polars as pl
from scipy import stats

from nfl_predictor import constants
from nfl_predictor.ratings.pbp_inputs import SeasonSources
from nfl_predictor.utils.polars import loaders, nflverse_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import numpy.typing as npt

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
FAMILY_CONFIDENCE = 0.95
# The single-game bootstrap draws in chunks, so a 5,000-game by 10,000-resample matrix never sits
# in memory at once.
_RESAMPLE_CHUNK = 500
# Every season of the inputs is historical to the loaders, so they read the cache and never
# refresh it.
CACHE_CURRENT_SEASON = 2027
# The reports are Markdown the repository's linter checks at 100 columns.
REPORT_WIDTH = 100
# A bound smaller than this would print as zero at three decimals.
_SMALLEST_SHOWN = 0.0005
# The season t interval has seasons - 1 degrees of freedom, so it needs two seasons.
_MIN_T_SEASONS = 2
_LIST_ITEM = re.compile(r"^(\s*)(- |\d+\. )")
_NFLREADPY_LOADERS = (
    "load_pbp",
    "load_player_stats",
    "load_players",
    "load_rosters",
    "load_rosters_weekly",
    "load_schedules",
    "load_snap_counts",
    "load_team_stats",
)


class DownloadBlockedError(RuntimeError):
    """Raised when a rating test tries to download from nflverse."""


def block_downloads() -> None:
    """Replace nflreadpy's loaders with ones that raise, so a cache miss cannot download."""

    def refuse(*_args: object, **_kwargs: object) -> pl.DataFrame:
        msg = "a rating test tried to download from nflverse; the cache lacks a file"
        raise DownloadBlockedError(msg)

    for name in _NFLREADPY_LOADERS:
        setattr(nfl, name, refuse)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path = constants.ROOT_DIR) -> str | None:
    """Return the commit ``repo`` is checked out at, marked when its tracked files differ.

    As ``git describe --dirty`` marks it, a staged or unstaged change to a tracked file appends
    ``-dirty`` to the full commit hash; untracked files do not count. Git runs without any
    ``GIT_*`` variable, so a git hook's ``GIT_DIR`` cannot point it at another repository.

    Args:
        repo: A directory inside the checkout; this repository's root by default.

    Returns:
        The commit hash, with ``-dirty`` when the tracked files differ from it, or ``None``
        outside a checkout or without ``git``.

    """
    git = shutil.which("git")
    if git is None:
        return None
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    head = subprocess.run(
        [git, "rev-parse", "HEAD"], cwd=repo, env=env, check=False, capture_output=True, text=True
    )
    if head.returncode != 0 or not head.stdout.strip():
        return None
    changes = subprocess.run(
        [git, "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    suffix = "-dirty" if changes.stdout.strip() else ""
    return f"{head.stdout.strip()}{suffix}"


def run_provenance(
    code: Sequence[Path], ledger: InputLedger, command: str, seconds: float
) -> dict[str, Any]:
    """Return what reproduces a run: the version, commit and code hashes, inputs and command.

    Keys: ``version``, ``code`` (each file's SHA-256 by name), ``inputs`` (the file count),
    ``fingerprint`` (:meth:`InputLedger.fingerprint`), ``command``, ``seconds`` and
    ``git_commit``.
    """
    return {
        "version": metadata.version("nfl-predictor"),
        "code": {path.name: sha256_file(path) for path in code},
        "inputs": len(ledger.files),
        "fingerprint": ledger.fingerprint(),
        "command": command,
        "seconds": seconds,
        "git_commit": git_commit(),
    }


def write_json_record(
    path: Path, payload: object, default: Callable[[object], object] | None = None
) -> None:
    """Write a report's JSON record, indented by one space and ending with a newline.

    The newline matches what the repository's end-of-file hook commits, so a rerun reproduces
    the committed bytes. ``default`` converts a value ``json`` cannot write.
    """
    path.write_text(json.dumps(payload, indent=1, default=default) + "\n")


@dataclass
class InputLedger:
    """The SHA-256 of every input file a run reads, by path."""

    files: dict[str, str] = field(default_factory=dict)

    def record(self, path: Path) -> None:
        """Hash ``path`` once.

        Raises:
            FileNotFoundError: If the file does not exist, so the run would not read what it says.

        """
        if not path.is_file():
            msg = f"input file missing: {path}"
            raise FileNotFoundError(msg)
        key = str(path)
        if key not in self.files:
            self.files[key] = sha256_file(path)

    def fingerprint(self) -> str:
        """Return one SHA-256 over every recorded file's ``sha256sum`` line, in path order."""
        listing = "".join(f"{digest}  {path}\n" for path, digest in sorted(self.files.items()))
        return hashlib.sha256(listing.encode()).hexdigest()


def load_season_plays(
    cache_dir: Path, season: int, ledger: InputLedger
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one season's regular-season play-by-play and schedule from the cache."""
    ledger.record(cache_dir / f"pbp_{season}_reg.parquet")
    ledger.record(cache_dir / f"schedule_{season}.parquet")
    plays = loaders.load_pbp([season], cache_dir=cache_dir, current_season=CACHE_CURRENT_SEASON)
    schedule = loaders.load_schedule(
        [season], cache_dir=cache_dir, current_season=CACHE_CURRENT_SEASON
    )
    return plays, schedule


def load_players(cache_dir: Path, ledger: InputLedger) -> pl.DataFrame:
    """Return the cached players file."""
    ledger.record(cache_dir / "players.parquet")
    return nflverse_sources.load_players(cache_dir=cache_dir)


def load_season_sources(
    cache_dir: Path, season: int, players: pl.DataFrame, ledger: InputLedger
) -> tuple[SeasonSources, pl.DataFrame]:
    """Return one season's quarterback-row sources and its schedule, as the ETL loads them.

    Each player source is asked only for a season it publishes, as the ETL asks
    (``ratings_build.load_ratings_sources``); an earlier season gets the source's empty frame.
    """
    plays, schedule = load_season_plays(cache_dir, season, ledger)
    sources = {
        "player_stats": (
            constants.NFLREADPY_PLAYER_STATS_MIN_SEASON,
            "player_stats_week",
            nflverse_sources.load_weekly_player_stats,
        ),
        "snap_counts": (
            constants.NFLREADPY_SNAP_COUNTS_MIN_SEASON,
            "snap_counts",
            nflverse_sources.load_snap_counts,
        ),
        "weekly_rosters": (
            constants.NFLREADPY_WEEKLY_ROSTERS_MIN_SEASON,
            "rosters_weekly",
            nflverse_sources.load_weekly_rosters,
        ),
    }
    frames: dict[str, pl.DataFrame] = {}
    for name, (first, stem, load) in sources.items():
        if season >= first:
            ledger.record(cache_dir / f"{stem}_{season}_reg.parquet")
        frames[name] = load(
            [season] if season >= first else [],
            cache_dir=cache_dir,
            current_season=CACHE_CURRENT_SEASON,
        )
    return (
        SeasonSources(
            season=season,
            plays=plays,
            player_stats=frames["player_stats"],
            snap_counts=frames["snap_counts"],
            players=players,
            weekly_rosters=frames["weekly_rosters"],
        ),
        schedule,
    )


def as_float(value: object) -> float:
    """Return a Polars scalar (a sum, mean or largest value) as a float, 0.0 for a null.

    Raises:
        TypeError: If the value is not a number.

    """
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    msg = f"expected a number, got {value!r}"
    raise TypeError(msg)


def bound_text(value: float) -> str:
    """Format an interval bound to three decimals, or four when three would show it as zero."""
    if value != 0.0 and abs(value) < _SMALLEST_SHOWN:
        return f"{value:+.4f}"
    return f"{value:+.3f}"


def wrap_markdown(lines: Sequence[str], width: int = REPORT_WIDTH) -> list[str]:
    """Return report lines with prose and list items filled to ``width``.

    Headings, table rows, blank lines, code blocks and lines that already fit stay as they are;
    a list item's continuation lines are indented under its text. Hyphenated spans such as
    ``1999-2025`` are never split.
    """
    wrapped: list[str] = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            wrapped.append(line)
            continue
        if in_code or len(line) <= width or line.startswith(("#", "|")):
            wrapped.append(line)
            continue
        item = _LIST_ITEM.match(line)
        wrapped.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=" " * len(item.group(0)) if item else "",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return wrapped


# The columns that place a row in time stay as they are when the information-set checks change
# the values of later games.
_PLACING_COLUMNS = frozenset({"season", "week"})


def perturbed_numbers(frame: pl.DataFrame, rows: pl.Expr) -> pl.DataFrame:
    """Return ``frame`` with every numeric value in ``rows`` changed, ``season`` and ``week`` aside.

    Floats become ``7 - 3x`` and integers ``x + 5``, so any value of those rows a prediction
    reads moves it. The information-set checks pass the rows of the games from the predicted
    week on.
    """
    return frame.with_columns(
        pl.when(rows)
        .then(pl.col(name) * -3.0 + 7.0 if dtype.is_float() else pl.col(name) + 5)
        .otherwise(pl.col(name))
        .alias(name)
        for name, dtype in frame.schema.items()
        if dtype.is_numeric() and name not in _PLACING_COLUMNS
    )


def comparison_confidence(comparisons: int, family: float = FAMILY_CONFIDENCE) -> float:
    """Return the per-comparison level that keeps ``family`` over ``comparisons`` (Bonferroni)."""
    return 1.0 - (1.0 - family) / comparisons


@dataclass(frozen=True, slots=True)
class Interval:
    """A paired difference's estimate and interval; negative favors the first of the pair."""

    estimate: float
    lower: float
    upper: float

    @property
    def below_zero(self) -> bool:
        """Return whether the whole interval lies below zero."""
        return self.upper < 0.0

    @property
    def above_zero(self) -> bool:
        """Return whether the whole interval lies above zero."""
        return self.lower > 0.0

    @property
    def excludes_zero(self) -> bool:
        """Return whether the interval lies entirely on one side of zero."""
        return self.below_zero or self.above_zero

    def flipped(self) -> Interval:
        """Return the interval of the reversed difference (Y - X instead of X - Y)."""
        return Interval(estimate=-self.estimate, lower=-self.upper, upper=-self.lower)


def season_draws(
    season_count: int, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> npt.NDArray[np.int64]:
    """Return ``resamples`` rows of ``season_count`` season indexes drawn with replacement."""
    return np.random.default_rng(seed).integers(0, season_count, size=(resamples, season_count))


def _tail(confidence: float) -> float:
    """Return the probability in each tail of a central interval at ``confidence``."""
    return (1.0 - confidence) / 2.0


def _require_every_season(counts: dict[int, int], seasons: Sequence[int]) -> None:
    """Raise a ``ValueError`` naming each season of ``seasons`` with no row in ``counts``."""
    empty = [season for season in seasons if counts.get(season, 0) == 0]
    if empty:
        msg = f"seasons without a paired difference: {', '.join(map(str, empty))}"
        raise ValueError(msg)


def season_bootstrap(
    differences: pl.DataFrame,
    seasons: Sequence[int],
    draws: npt.NDArray[np.int64],
    confidence: float,
) -> Interval:
    """Return the mean paired difference and its percentile interval from whole-season resamples.

    Args:
        differences: One row per game (or team-game) with ``season`` and ``difference``.
        seasons: The seasons the draws index, in order.
        draws: From :func:`season_draws`; each row picks ``len(seasons)`` seasons, keeping each
            drawn season's rows together.
        confidence: The central share the interval covers.

    Returns:
        The mean over every row, and the quantiles of the mean over each resample's drawn rows.

    Raises:
        ValueError: If a season of ``seasons`` has no row, so a resample could hold none.

    """
    totals = dict(differences.group_by("season").agg(pl.col("difference").sum()).iter_rows())
    counts = dict(differences.group_by("season").len().iter_rows())
    _require_every_season(counts, seasons)
    sums = np.array([totals[season] for season in seasons], dtype=np.float64)
    games = np.array([counts[season] for season in seasons], dtype=np.float64)
    resampled = sums[draws].sum(axis=1) / games[draws].sum(axis=1)
    tail = _tail(confidence)
    lower, upper = np.quantile(resampled, [tail, 1.0 - tail])
    return Interval(float(sums.sum() / games.sum()), float(lower), float(upper))


def season_t_interval(
    differences: pl.DataFrame, seasons: Sequence[int], confidence: float
) -> Interval:
    """Return the mean of the season means and its Student-t interval (seasons - 1 df).

    Raises:
        ValueError: If ``seasons`` holds fewer than two seasons, which leave no degree of freedom,
            or if a season of ``seasons`` has no row.

    """
    if len(seasons) < _MIN_T_SEASONS:
        msg = f"a season t interval needs at least two seasons; got {len(seasons)}"
        raise ValueError(msg)
    _require_every_season(dict(differences.group_by("season").len().iter_rows()), seasons)
    means = dict(differences.group_by("season").agg(pl.col("difference").mean()).iter_rows())
    values = np.array([means[season] for season in seasons], dtype=np.float64)
    center = float(values.mean())
    half = float(
        stats.t.ppf(1.0 - _tail(confidence), df=values.size - 1)
        * values.std(ddof=1)
        / np.sqrt(values.size)
    )
    return Interval(center, center - half, center + half)


def game_bootstrap(
    values: npt.NDArray[np.float64],
    *,
    confidence: float,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Return the mean of ``values`` and its percentile interval from resampling single rows."""
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, _RESAMPLE_CHUNK):
        size = min(_RESAMPLE_CHUNK, resamples - start)
        means[start : start + size] = values[
            rng.integers(0, values.size, (size, values.size))
        ].mean(axis=1)
    tail = _tail(confidence)
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(values.mean()), float(lower), float(upper))
