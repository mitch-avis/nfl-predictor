"""Both families of this repository's own Elo, as the ETL publishes them in the game dataset.

``elo_ratings.build_own_elo_features`` publishes one preset at a time; the dataset carries both:
the market-free preset (``own_elo``, ``constants.OWN_ELO_COLUMNS``) and the preset with the
win-total pull (``own_elo_wt``, ``constants.OWN_WT_ELO_COLUMNS``). Both read the same
quarterback adjustments, 538's VALUE model over the run's quarterback box scores
(``elo_quarterbacks``): each completed regular-season game rated for its play-by-play starter,
and every other game for the starter the ETL names, else the team's last one.

The win totals are market information and a published input of the win-total preset, so a build
whose win-total file covers none of its seasons fails instead of publishing that preset without
its pull; a built season the file lacks (the current one, before nfelosrs publishes it) is
logged. The build's record names both presets (their switches and fitted weights) and the
win-total file it read (its address, origin and SHA-256).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import elo_quarterbacks, elo_ratings

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from nfl_predictor.utils.polars.elo_win_totals import WinTotalsSnapshot

# Every schedule column the two presets read (travel needs the stadium).
SCHEDULE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_type",
    "home_abbr",
    "away_abbr",
    "home_score",
    "away_score",
    "neutral",
    "stadium_id",
)
COLUMNS: list[str] = [*constants.OWN_ELO_COLUMNS, *constants.OWN_WT_ELO_COLUMNS]


@dataclass(frozen=True, eq=False)
class OwnEloColumns:
    """Both own Elo families and the record of how they were built.

    Attributes:
        frame: ``game_id`` and ``COLUMNS``, one row per schedule row, in its order.
        metadata: ``market_free`` and ``win_totals`` (each preset's
            ``OwnEloFeatures.metadata``), ``win_total_file`` (``WinTotalsSnapshot.metadata``)
            and the build's first and last ``seasons``.

    """

    frame: pl.DataFrame
    metadata: dict[str, object]


def require_win_totals(win_totals: pl.DataFrame, seasons: Sequence[int]) -> None:
    """Raise when the win totals cover none of ``seasons``; log the seasons they lack.

    Raises:
        ValueError: When no row of ``win_totals`` falls in ``seasons``.

    """
    covered = set(win_totals.get_column("season").to_list()) if win_totals.height else set()
    built = sorted(set(seasons))
    if not covered & set(built):
        msg = (
            f"the win totals cover none of the build's seasons ({built[0]}-{built[-1]}): the "
            "win-total preset would carry no pull; check the win-total download and its cache"
        )
        raise ValueError(msg)
    first = min(covered)
    lacking = [season for season in built if season >= first and season not in covered]
    if lacking:
        log.warning(
            "Own Elo: no win totals for season(s) %s; their preseason ratings get no pull",
            lacking,
        )


def own_elo_columns(
    schedule: pl.DataFrame,
    *,
    box_scores: pl.DataFrame,
    named_starters: pl.DataFrame,
    win_totals: WinTotalsSnapshot,
    seasons: Sequence[int],
) -> OwnEloColumns:
    """Return both own Elo families for every game of ``schedule``.

    Args:
        schedule: Every game from the frame's first season (1999) through the last built one,
            upcoming games included, with ``SCHEDULE_COLUMNS`` (``date`` when present).
        box_scores: ``elo_quarterbacks.quarterback_box_scores`` output.
        named_starters: ``game_id``, ``team_abbr`` and ``qb_id`` of the starter the ETL names
            for each side (``qb_stats.named_starters``), read where a game has no
            play-by-play starter.
        win_totals: The win-total file the run read (``elo_win_totals.load_win_totals``).
        seasons: The seasons the build publishes.

    Returns:
        The two families and their record (``OwnEloColumns``).

    Raises:
        ValueError: When the win totals cover none of ``seasons``, or as
            ``elo_ratings.build_own_elo_features`` does.

    """
    require_win_totals(win_totals.frame, seasons)
    adjustments = elo_quarterbacks.value_qb_adjustments(
        schedule, box_scores, projected_starters=named_starters
    )
    market_free = elo_ratings.build_own_elo_features(
        schedule,
        variant="market_free",
        inputs=elo_ratings.EloInputs(qb_adjustments=adjustments),
    )
    with_win_totals = elo_ratings.build_own_elo_features(
        schedule,
        variant="win_totals",
        inputs=elo_ratings.EloInputs(qb_adjustments=adjustments, win_totals=win_totals.frame),
    )
    frame = market_free.frame.join(
        with_win_totals.frame, on="game_id", how="left", maintain_order="left"
    )
    built = sorted(set(seasons))
    metadata: dict[str, object] = {
        "market_free": market_free.metadata,
        "win_totals": with_win_totals.metadata,
        "win_total_file": win_totals.metadata(),
        "seasons": [built[0], built[-1]],
    }
    return OwnEloColumns(frame.select("game_id", *COLUMNS), metadata)
