"""Score the ported quarterback rating as the own Elo's quarterback adjustment, against VALUE.

Run from the repository root (a worktree is fine), with the inputs ``elo_eval.py`` reads and a
candidate build whose ``ratings/`` directory holds the published quarterback ratings:

    VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/elo_qb_alt.py \
        --data-dir /home/mitch/workspace/nfl-predictor/data \
        --nflreadpy-cache /tmp/step6_int/cache \
        --candidate-data /home/mitch/scratch/step6/data

It writes ``ELO_QB_ALT.md`` beside this file and nothing else. The rule that reads it is
``decisions/elo.md``, entry 17, written before the first run. Everything is the market-free
recommendation (``elo_ratings.MARKET_FREE_SETTINGS``) with VALUE's games, game order and starters
(``elo_quarterbacks.value_qb_adjustments``, ``pre_game_starter="game_starter"``); only the per-game
``home_qb_adj`` and ``away_qb_adj`` handed to ``EloInputs.qb_adjustments`` change.

The rating ``r(q)`` a game in season ``s``, week ``w`` uses for quarterback ``q``:

1. his season-to-date ``adj_qb_epa_per_dropback`` fit on season ``s``'s games before week ``w``,
   when he has a dropback in them: the candidate dataset's attached starter rating
   (``ratings_qb``), fit with the previous season's penalty. The published
   ``{s}_qb_ratings_by_week.parquet`` row through week ``w - 1`` is the same fit with season
   ``s``'s own cross-validated penalty, chosen on the whole season, which is later information;
   the report checks the two against each other season by season;
2. else his latest-row rating (``{s}_qb_ratings_by_week.parquet``) of the most recent earlier
   season where it is not null;
3. else the replacement level of season ``s``: the mean latest-row rating of the quarterbacks
   who debuted (first non-null season) in 2001 through ``s - 1``, or 0;
4. in a playoff game, his latest regular-season row of season ``s``, without updates.

A team's baseline rolls ``0.95 * team + 0.05 * r`` after each regular-season game with a
play-by-play starter, starts at its first starter's ``r`` and blends a third of its end-of-season
value with two thirds of its first starter's ``r`` each preseason. The adjustment is
``25 * D(s) * (r(starter) - team)`` Elo points, ``D(s)`` the mean dropbacks per team-game of the
regular seasons 1999 through ``s - 1`` (0 in 1999, and when no starter is known).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import elo_eval
import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.polars import elo_quarterbacks, elo_ratings, qb_starters

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "ELO_QB_ALT.md"

RATING = "adj_qb_epa_per_dropback"
DROPBACKS = "qb_dropbacks"
# 538's Elo points per point of spread.
ELO_PER_POINT = 25.0
TEAM_WEIGHT = 0.05
PRESEASON_CARRYOVER = 1.0 / 3.0
# 1999 has no rating and 2000 rates veterans for the first time: neither is a debut season.
FIRST_DEBUT_SEASON = 2001
# The candidate dataset rounds its attached ratings to six decimals.
MATCH_TOLERANCE = 1e-6
SCALES = (1.0, 0.5, 1.5)
# VALUE's market-free numbers in ELO_EVAL.md, which this run must reproduce.
VALUE_BRIER = 0.21701
VALUE_LOG_LOSS = 0.62243


@dataclass(frozen=True)
class _Row:
    """One quarterback's row of a by-week table."""

    week: int
    dropbacks: int
    rating: float | None


class QuarterbackRatings:
    """Each quarterback's published ratings by season and week, read as the rule reads them."""

    def __init__(
        self,
        tables: Mapping[int, pl.DataFrame],
        *,
        offset: int = 1,
        attached: Mapping[tuple[int, int, str], float | None] | None = None,
    ) -> None:
        """Index the by-week tables.

        Args:
            tables: Each season's ``{season}_qb_ratings_by_week.parquet`` rows (``week``,
                ``qb_id``, ``qb_dropbacks``, ``adj_qb_epa_per_dropback``).
            offset: 1 when the row labelled ``week`` holds the fit through that week, so a game
                of week ``w`` reads the latest row labelled ``w - 1`` or earlier; 0 when it holds
                the fit before that week.
            attached: When given, the season-to-date ratings by season, week and quarterback
                (the candidate dataset's attached starter rating, null before his first
                dropback), read instead of the tables' rows; a key it lacks has none.

        """
        self._offset = offset
        self._attached = attached
        self._rows: dict[int, dict[str, list[_Row]]] = {}
        for season in sorted(tables):
            by_qb: dict[str, list[_Row]] = {}
            ordered = (
                tables[season].sort("week", "qb_id").select("week", "qb_id", DROPBACKS, RATING)
            )
            for week, qb_id, dropbacks, rating in ordered.iter_rows():
                by_qb.setdefault(qb_id, []).append(_Row(week, dropbacks, rating))
            self._rows[season] = by_qb
        self._latest = {
            season: {qb_id: rows[-1] for qb_id, rows in by_qb.items()}
            for season, by_qb in self._rows.items()
        }
        self.debuts = self._debut_ratings()

    def _debut_ratings(self) -> list[tuple[int, str, float]]:
        """Return each quarterback's first season with a non-null latest-row rating, and it."""
        seen: set[str] = set()
        debuts = []
        for season, latest in self._latest.items():
            for qb_id in sorted(latest):
                rating = latest[qb_id].rating
                if rating is not None and qb_id not in seen:
                    seen.add(qb_id)
                    debuts.append((season, qb_id, rating))
        return debuts

    def season_to_date(self, qb_id: str, season: int, week: int) -> float | None:
        """Return his rating fit on the season's games before ``week``, or None without one."""
        if self._attached is not None:
            return self._attached.get((season, week, qb_id))
        rows = self._rows.get(season, {}).get(qb_id)
        if not rows:
            return None
        index = bisect.bisect_right([row.week for row in rows], week - self._offset)
        if index == 0 or rows[index - 1].dropbacks <= 0:
            return None
        return rows[index - 1].rating

    def earlier_season(self, qb_id: str, season: int) -> float | None:
        """Return his latest-row rating of the most recent earlier season where it is not null."""
        for year in sorted((year for year in self._latest if year < season), reverse=True):
            row = self._latest[year].get(qb_id)
            if row is not None and row.rating is not None:
                return row.rating
        return None

    def replacement_level(self, season: int) -> float:
        """Return the mean debut rating of the debut seasons 2001 through ``season - 1``, or 0."""
        values = [rating for year, _, rating in self.debuts if FIRST_DEBUT_SEASON <= year < season]
        return sum(values) / len(values) if values else 0.0

    def rating(self, qb_id: str, season: int, week: int, *, playoff: bool) -> float:
        """Return the rating a game of ``season`` and ``week`` uses for quarterback ``qb_id``."""
        if playoff:
            row = self._latest.get(season, {}).get(qb_id)
            value = None if row is None else row.rating
        else:
            value = self.season_to_date(qb_id, season, week)
        if value is None:
            value = self.earlier_season(qb_id, season)
        return self.replacement_level(season) if value is None else value

    def dropbacks_per_team_game(
        self, season: int, regular_games: Mapping[int, int]
    ) -> float | None:
        """Return the mean dropbacks per team-game of the earlier seasons, or None without one."""
        rates = [
            sum(row.dropbacks for row in self._latest[year].values()) / (2 * regular_games[year])
            for year in self._latest
            if year < season
        ]
        return sum(rates) / len(rates) if rates else None


@dataclass
class _Team:
    """A team's rolling baseline and the season it was last rated in."""

    rating: float
    season: int


def alternative_adjustments(
    games: pl.DataFrame,
    starts: Mapping[tuple[int, int, str], str],
    ratings: QuarterbackRatings,
    regular_games: Mapping[int, int],
    *,
    scale: float = 1.0,
) -> pl.DataFrame:
    """Return each game's quarterback adjustments from the ported rating, in the games' order.

    Args:
        games: ``game_id``, ``season``, ``week``, ``game_type``, ``home_abbr``, ``away_abbr``,
            ``home_qb_id`` and ``away_qb_id`` (VALUE's starters), in rating order.
        starts: Each regular-season team-game's play-by-play starter, by season, week and team.
        ratings: The published quarterback ratings.
        regular_games: Each season's number of regular-season games, for ``D(s)``.
        scale: A multiplier on every adjustment (the sensitivity runs).

    Returns:
        ``game_id`` and, per side, ``qb_adj`` in Elo points (0 without a starter or ``D(s)``),
        the starter's ``qb_rating`` and the pre-game ``team_rating``.

    """
    teams: dict[str, _Team] = {}
    per_game: dict[int, float | None] = {}
    rows: list[dict[str, object]] = []
    columns = ("game_id", "season", "week", "game_type", "home_abbr", "away_abbr")
    for game_id, season, week, game_type, home, away, home_qb, away_qb in games.select(
        *columns, "home_qb_id", "away_qb_id"
    ).iter_rows():
        playoff = game_type != "REG"
        if season not in per_game:
            per_game[season] = ratings.dropbacks_per_team_game(season, regular_games)
        dropbacks = per_game[season]
        row: dict[str, object] = {"game_id": game_id}
        for side, team, qb_id in (("home", home, home_qb), ("away", away, away_qb)):
            value = None if qb_id is None else ratings.rating(qb_id, season, week, playoff=playoff)
            state = teams.get(team)
            if state is None:
                initial = value if value is not None else ratings.replacement_level(season)
                state = teams.setdefault(team, _Team(initial, season))
            elif state.season < season:
                if value is not None:
                    state.rating = (
                        PRESEASON_CARRYOVER * state.rating + (1.0 - PRESEASON_CARRYOVER) * value
                    )
                state.season = season
            adjustment = (
                0.0
                if value is None or dropbacks is None
                else scale * ELO_PER_POINT * dropbacks * (value - state.rating)
            )
            row |= {
                f"{side}_qb_adj": adjustment,
                f"{side}_qb_rating": value,
                f"{side}_team_rating": state.rating,
            }
        for team in () if playoff else (home, away):
            starter = starts.get((season, week, team))
            if starter is not None:
                value = ratings.rating(starter, season, week, playoff=False)
                teams[team].rating = (1.0 - TEAM_WEIGHT) * teams[team].rating + TEAM_WEIGHT * value
        rows.append(row)
    schema = {"game_id": pl.String} | {
        f"{side}_{name}": pl.Float64
        for side in ("home", "away")
        for name in ("qb_adj", "qb_rating", "team_rating")
    }
    return pl.DataFrame(rows, schema=schema)


def starter_rating_check(
    sides: pl.DataFrame, tables: Mapping[int, pl.DataFrame]
) -> dict[int, dict[str, int]]:
    """Count, per week-label offset, the sides whose attached starter rating the tables give.

    Args:
        sides: ``season``, ``week``, ``qb_id`` and ``attached``, the candidate dataset's starter
            rating, of completed regular-season game sides whose starter is VALUE's.
        tables: Each season's by-week quarterback table.

    Returns:
        For offsets 0 (the row labelled ``week`` is the fit before it) and 1 (through it), how
        many sides match: both null, or within ``MATCH_TOLERANCE``.

    """
    counts: dict[int, dict[str, int]] = {}
    for offset in (0, 1):
        ratings = QuarterbackRatings(tables, offset=offset)
        matches = 0
        for season, week, qb_id, attached in sides.select(
            "season", "week", "qb_id", "attached"
        ).iter_rows():
            value = ratings.season_to_date(qb_id, season, week)
            if value is None or attached is None:
                matched = value is None and attached is None
            else:
                matched = abs(value - attached) <= MATCH_TOLERANCE
            matches += int(matched)
        counts[offset] = {"matches": matches, "mismatches": sides.height - matches}
    return counts


def season_check(
    sides: pl.DataFrame, tables: Mapping[int, pl.DataFrame]
) -> dict[int, dict[str, int]]:
    """Compare, season by season, the attached starter rating with the through-week reading.

    The published history fits each season with that season's own cross-validated penalty, the
    attached rating with the previous season's (``ratings.pre_week``), so a season whose two
    penalties differ differs at every side with a rating, and one whose penalties agree at none.

    Args:
        sides: As for ``starter_rating_check``.
        tables: Each season's by-week quarterback table.

    Returns:
        Per season, the sides with both ratings equal (within ``MATCH_TOLERANCE``), different,
        both null, and one null.

    """
    ratings = QuarterbackRatings(tables, offset=1)
    counts: dict[int, dict[str, int]] = {}
    for season, week, qb_id, attached in (
        sides.sort("season").select("season", "week", "qb_id", "attached").iter_rows()
    ):
        value = ratings.season_to_date(qb_id, season, week)
        if value is None or attached is None:
            kind = "both_null" if value is None and attached is None else "one_null"
        else:
            kind = "equal" if abs(value - attached) <= MATCH_TOLERANCE else "different"
        season_counts = counts.setdefault(
            season, {"equal": 0, "different": 0, "both_null": 0, "one_null": 0}
        )
        season_counts[kind] += 1
    return counts


def load_tables(ratings_dir: Path) -> tuple[dict[int, pl.DataFrame], list[Path]]:
    """Return each season's by-week quarterback table, 1999's ratings set to null.

    The ETL rates no quarterback in 1999, the first play-by-play season, since no earlier season
    supplies the fit's penalties (``ratings_build``); its published table still carries values,
    which the rule treats as null. Its dropbacks stay, for ``D(2000)``.
    """
    paths = [
        ratings_dir / f"{season}_qb_ratings_by_week.parquet"
        for season in range(elo_eval.FIRST_SEASON, elo_eval.LAST_SEASON + 1)
    ]
    tables = {}
    for season, path in zip(
        range(elo_eval.FIRST_SEASON, elo_eval.LAST_SEASON + 1), paths, strict=True
    ):
        table = pl.read_parquet(path).select("week", "qb_id", DROPBACKS, RATING)
        if season == elo_eval.FIRST_SEASON:
            table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias(RATING))
        tables[season] = table
    return tables, paths


def play_by_play_starters(box: pl.DataFrame) -> dict[tuple[int, int, str], str]:
    """Return each team-game's play-by-play starter, the first as VALUE's lookup keeps it."""
    starts: dict[tuple[int, int, str], str] = {}
    for season, week, team, qb_id in (
        box.filter(pl.col("starter")).select("season", "week", "team_abbr", "qb_id").iter_rows()
    ):
        starts.setdefault((season, week, team), qb_id)
    return starts


def check_sides(
    games: pl.DataFrame, starts: Mapping[tuple[int, int, str], str], dataset: pl.DataFrame
) -> pl.DataFrame:
    """Return the completed regular-season sides whose dataset starter is VALUE's.

    The dataset attaches its rating to a completed game's play-by-play starter
    (``qb_stats.expected_starter_ids``), so the sides kept are those VALUE rates for that same
    play-by-play starter.
    """
    joined = games.filter(pl.col("game_type") == "REG").join(
        dataset, on="game_id", how="inner", maintain_order="left"
    )
    parts = []
    for side in ("home", "away"):
        part = joined.select(
            "season",
            "week",
            pl.col(f"{side}_abbr").alias("team"),
            pl.col(f"{side}_qb_id").alias("qb_id"),
            pl.col(f"{side}_ratings_qb").cast(pl.Float64).alias("attached"),
        )
        keep = [
            starts.get((season, week, team)) == qb_id and qb_id is not None
            for season, week, team, qb_id in part.select(
                "season", "week", "team", "qb_id"
            ).iter_rows()
        ]
        parts.append(part.filter(pl.Series(keep, dtype=pl.Boolean)))
    return pl.concat(parts).select("season", "week", "qb_id", "attached")


def verdict(comparison: dict[str, Any]) -> str:
    """Return the rule's outcome for a candidate-minus-VALUE comparison."""
    if elo_eval.passes_rule(comparison):
        return "would replace VALUE"
    if not comparison["brier_ci"][0] > 0.0 and not comparison["log_loss_ci"][0] > 0.0:
        return "measures at least as well; VALUE stays"
    return "VALUE stays"


def _sha256_of_files(paths: list[Path]) -> str:
    """Return one SHA-256 over the files' bytes, in order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _week_briers(window: elo_eval.Window, probability: np.ndarray) -> dict[str, float]:
    """Return the Brier score of each week window."""
    weeks = window.games["week"].to_numpy()
    return {
        label: elo_eval.summary(window.outcome[mask], probability[mask])["brier"]
        for label, first, last in elo_eval.WEEK_WINDOWS
        if (mask := (weeks >= first) & (weeks <= last)).any()
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Build both adjustments, check the inputs, score every scale and apply the rule."""
    inputs = elo_eval.load_inputs(
        args.data_dir, args.nflreadpy_cache, args.fivethirtyeight, args.win_totals
    )
    window = elo_eval.evaluation_window(inputs.schedule)
    history = qb_starters.quarterback_history(inputs.plays, inputs.positions)
    box = elo_quarterbacks.quarterback_box_scores(history)
    value = elo_quarterbacks.value_qb_adjustments(
        inputs.schedule, box, projected_starters=elo_eval.named_starters(inputs, history.qb_games)
    )
    settings = elo_ratings.MARKET_FREE_SETTINGS
    value_probability = elo_eval.Runner(inputs, window, value).probability(settings)
    value_summary = elo_eval.summary(window.outcome, value_probability)
    reproduced = (round(value_summary["brier"], 5), round(value_summary["log_loss"], 5))
    if reproduced != (VALUE_BRIER, VALUE_LOG_LOSS):
        msg = f"VALUE does not reproduce ELO_EVAL.md: {reproduced}"
        raise SystemExit(msg)
    games = value.select("game_id", "home_qb_id", "away_qb_id").join(
        inputs.schedule.select("game_id", "season", "week", "game_type", "home_abbr", "away_abbr"),
        on="game_id",
        how="left",
        maintain_order="left",
    )
    starts = play_by_play_starters(box)
    ratings_dir = args.candidate_data / constants.RATINGS_DIRNAME
    tables, table_paths = load_tables(ratings_dir)
    dataset_path = args.candidate_data / "completed_games.csv"
    dataset = pl.read_csv(
        dataset_path,
        columns=["game_id", "home_ratings_qb", "away_ratings_qb"],
        infer_schema_length=None,
    )
    sides = check_sides(games, starts, dataset)
    check = starter_rating_check(sides, tables)
    by_season = season_check(sides, tables)
    unexplained = {
        season: counts
        for season, counts in by_season.items()
        if counts["one_null"] or (counts["equal"] and counts["different"])
    }
    if unexplained:
        msg = f"the through-week reading differs other than by season: {unexplained}"
        raise SystemExit(msg)
    attached = {
        (season, week, qb_id): rating
        for season, week, qb_id, rating in sides.select(
            "season", "week", "qb_id", "attached"
        ).iter_rows()
    }
    ratings = QuarterbackRatings(tables, attached=attached)
    window_games = set(window.games["game_id"].to_list())
    window_sides = [
        (season, week, qb_id)
        for game_id, season, week, home_qb, away_qb in games.select(
            "game_id", "season", "week", "home_qb_id", "away_qb_id"
        ).iter_rows()
        if game_id in window_games
        for qb_id in (home_qb, away_qb)
    ]
    unattached = sum(key not in attached for key in window_sides)
    regular_games = dict(
        inputs.schedule.filter(pl.col("game_type") == "REG").group_by("season").len().iter_rows()
    )
    scored = []
    for scale in SCALES:
        adjustments = alternative_adjustments(games, starts, ratings, regular_games, scale=scale)
        probability = elo_eval.Runner(inputs, window, adjustments).probability(settings)
        comparison = elo_eval.paired(window.outcome, probability, value_probability)
        scored.append(
            {
                "scale": scale,
                "comparison": comparison,
                "verdict": verdict(comparison),
                "week_brier": _week_briers(window, probability),
            }
        )
    seasons = range(elo_eval.FIRST_SEASON, elo_eval.LAST_SEASON + 1)
    return {
        "provenance": inputs.provenance
        | {
            "candidate_data": str(args.candidate_data),
            "qb_ratings_by_week_sha256": _sha256_of_files(table_paths),
            "completed_games": {
                "path": str(dataset_path),
                "sha256": _sha256_of_files([dataset_path]),
            },
        },
        "settings": elo_eval.settings_label(settings),
        "value": value_summary | {"week_brier": _week_briers(window, value_probability)},
        "check": check,
        "by_season": by_season,
        "window_sides": len(window_sides),
        "unattached": unattached,
        "scored": scored,
        "per_season": [
            {
                "season": season,
                "dropbacks": ratings.dropbacks_per_team_game(season, regular_games),
                "replacement": ratings.replacement_level(season),
                "debuts": sum(year == season for year, _, _ in ratings.debuts),
            }
            for season in seasons
        ],
    }


REPRODUCE = """```bash
VIRTUAL_ENV="$PWD/.venv" nice -n 10 .venv/bin/python .agents/sos_port/elo_qb_alt.py \\
    --data-dir /home/mitch/workspace/nfl-predictor/data \\
    --nflreadpy-cache /tmp/step6_int/cache \\
    --candidate-data /home/mitch/scratch/step6/data
```"""


def _scale_label(scale: float) -> str:
    """Return the plain name of a scale."""
    return "the rule's scale" if scale == 1.0 else f"sensitivity, scale times {scale:g}"


def _input_lines(result: dict[str, Any]) -> list[str]:
    """Return the inputs section."""
    provenance = result["provenance"]
    rows = [
        ["code commit", f"`{provenance['git_commit']}`"],
        ["data directory", f"`{provenance['data_dir']}`"],
        ["nflreadpy cache", f"`{provenance['nflreadpy_cache']}`"],
        ["schedule cache files SHA-256", f"`{provenance['schedule_files_sha256']}`"],
        ["play-by-play cache files SHA-256", f"`{provenance['play_by_play_files_sha256']}`"],
        ["players file and weekly rosters SHA-256", f"`{provenance['identity_files_sha256']}`"],
        ["`qb_elos.csv` SHA-256", f"`{provenance['qb_elos']['sha256']}`"],
        ["`meta_data.csv` SHA-256", f"`{provenance['meta_data']['sha256']}`"],
        ["candidate build", f"`{provenance['candidate_data']}`"],
        [
            "quarterback ratings by week (1999-2025) SHA-256",
            f"`{provenance['qb_ratings_by_week_sha256']}`",
        ],
        ["candidate `completed_games.csv` SHA-256", f"`{provenance['completed_games']['sha256']}`"],
    ]
    return ["## Inputs", "", *elo_eval._table(["input", "detail"], rows)]


def _check_lines(result: dict[str, Any]) -> list[str]:
    """Return the starter-rating check section."""
    names = {0: "the fit before that week", 1: "the fit through that week"}
    rows = [
        [
            f"row labelled `week` holds {names[offset]}",
            str(counts["matches"]),
            str(counts["mismatches"]),
        ]
        for offset, counts in sorted(result["check"].items())
    ]
    season_rows = [
        [
            str(season),
            *(str(counts[key]) for key in ("equal", "different", "both_null", "one_null")),
        ]
        for season, counts in sorted(result["by_season"].items())
    ]
    return [
        "## Which rating a game reads",
        "",
        *elo_eval._paragraph(
            "Each completed regular-season game side whose starter is VALUE's (its play-by-play",
            "starter, the one the candidate dataset attaches its `ratings_qb` to), compared with",
            "the season-to-date rating each reading of the by-week table gives: a match is both",
            f"null or within {MATCH_TOLERANCE:g} (the dataset rounds to six decimals).",
        ),
        *elo_eval._table(["reading", "matches", "mismatches"], rows),
        *elo_eval._paragraph(
            "The row labelled `week` holds the fit through that week, but it equals the attached",
            "rating only in some seasons. By season, with the through-week reading:",
        ),
        *elo_eval._table(["season", "equal", "different", "both null", "one null"], season_rows),
        *elo_eval._paragraph(
            "Every season either matches at every rated side or differs at every one: the",
            "pattern of the documented difference between the two fits",
            "(`src/nfl_predictor/ratings/pre_week.py`), where the published history uses the",
            "season's own cross-validated quarterback penalty, chosen on the whole season, and",
            "the attached rating the previous season's, so a season differs when its two",
            "penalties do. The penalties are not on disk, so that last step is not checked. A",
            "penalty chosen on the whole season is later information, so the run reads the",
            "season-to-date rating from the attached `ratings_qb` (the fit on the season's games",
            "before the week, with the previous season's penalty), keyed by season, week and",
            "starter, and the by-week table only for its latest rows (earlier seasons, the",
            "replacement level, `D(s)` and playoff games). Of the",
            f"{result['window_sides']} window game sides, {result['unattached']} have no attached",
            "rating (no play-by-play starter for VALUE to share) and read cases 2 and 3.",
        ),
    ]


def _result_lines(result: dict[str, Any]) -> list[str]:
    """Return the paired table and the week windows."""
    value = result["value"]
    rows = []
    for item in result["scored"]:
        comparison = item["comparison"]
        rows.append(
            [
                _scale_label(item["scale"]),
                elo_eval._number(comparison["candidate_brier"]),
                elo_eval._signed(comparison["brier_diff"]),
                elo_eval._interval(comparison["brier_ci"]),
                elo_eval._number(comparison["candidate_log_loss"]),
                elo_eval._signed(comparison["log_loss_diff"]),
                elo_eval._interval(comparison["log_loss_ci"]),
                item["verdict"] if item["scale"] == 1.0 else "decides nothing",
            ]
        )
    header = ["adjustment", "Brier", "Brier diff", "95% interval", "log loss", "log-loss diff"]
    windows = [label for label, _, _ in elo_eval.WEEK_WINDOWS]
    week_rows = [["VALUE", *(elo_eval._number(value["week_brier"][label]) for label in windows)]]
    week_rows += [
        [
            _scale_label(item["scale"]),
            *(elo_eval._number(item["week_brier"][label]) for label in windows),
        ]
        for item in result["scored"]
    ]
    return [
        "## Ported rating minus VALUE",
        "",
        *elo_eval._paragraph(
            f"VALUE ({result['settings']}) reproduces `ELO_EVAL.md`: Brier",
            f"{elo_eval._number(value['brier'])} and log loss",
            f"{elo_eval._number(value['log_loss'])}",
            f"on {value['games']} games. Each row is the same settings with the ported rating's",
            "adjustment; differences are the row minus VALUE, per game.",
        ),
        *elo_eval._table([*header, "95% interval", "rule"], rows),
        "## Brier by week window",
        "",
        *elo_eval._paragraph("No intervals; these decide nothing."),
        *elo_eval._table(["adjustment", *windows], week_rows),
    ]


def _season_lines(result: dict[str, Any]) -> list[str]:
    """Return the per-season scale and replacement level."""
    rows = [
        [
            str(item["season"]),
            "n/a" if item["dropbacks"] is None else elo_eval._number(item["dropbacks"], 2),
            elo_eval._signed(item["replacement"]),
            str(item["debuts"]),
        ]
        for item in result["per_season"]
    ]
    return [
        "## Scale and replacement level by season",
        "",
        *elo_eval._paragraph(
            "`D(s)` is the mean dropbacks per team-game of the regular seasons 1999 through",
            "`s - 1` (each quarterback's latest-row dropbacks, summed, over twice the season's",
            "regular-season games); the adjustment is `25 * D(s)` Elo points per point of",
            "rating. The replacement level of season `s` is the mean latest-row rating of the",
            "quarterbacks who debuted in 2001 through `s - 1`. Debuts counts the quarterbacks",
            "whose first non-null season is that one (1999 and 2000 are not debut seasons).",
        ),
        *elo_eval._table(["season", "D(s)", "replacement level", "debuts"], rows),
    ]


def markdown(result: dict[str, Any]) -> str:
    """Return the report."""
    lines = [
        "# The ported quarterback rating as the own Elo's adjustment",
        "",
        *elo_eval._paragraph(
            "Generated by `.agents/sos_port/elo_qb_alt.py`; do not edit by hand. Reproduce",
            "from the repository root with:",
        ),
        REPRODUCE,
        "",
        *elo_eval._paragraph(
            "The rule that reads these numbers is `decisions/elo.md`, entry 17, written before",
            "the first run. The method is in the script's docstring. Window: completed",
            f"regular-season games, 2007-2025, {result['value']['games']} games. Intervals are",
            f"95% percentile intervals over {elo_eval.BOOTSTRAP_SAMPLES} resamples of the games",
            f"(seed {elo_eval.BOOTSTRAP_SEED}).",
        ),
        *_input_lines(result),
        *_check_lines(result),
        *_result_lines(result),
        *_season_lines(result),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    """Run the evaluation and write the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nflreadpy-cache", type=Path, required=True)
    parser.add_argument("--candidate-data", type=Path, required=True)
    parser.add_argument(
        "--fivethirtyeight",
        type=Path,
        default=Path("~/scratch/elo_research/nfl-elo-game/data/nfl_games.csv").expanduser(),
    )
    parser.add_argument(
        "--win-totals",
        type=Path,
        default=Path("~/scratch/elo_research/wt_ratings_nfelosrs.csv").expanduser(),
    )
    OUT_MD.write_text(markdown(evaluate(parser.parse_args())), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
