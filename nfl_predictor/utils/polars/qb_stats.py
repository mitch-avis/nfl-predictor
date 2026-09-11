"""Quarterback per-dropback features for the expected starter, built from play-by-play.

The family gives each game row the production of the quarterback the ETL expects to start
(``away_qb`` / ``home_qb``), measured only from that quarterback's regular-season games in
earlier weeks, across teams and seasons.

Pipeline:

1. Identity. Elo quarterback names map to GSIS ids through a read-only copy of the nfeloqb
   metadata (``name_id`` -> ``gsis_id``), which is also the play-by-play ``passer_player_id``.
   Names that map to more than one id are dropped as ambiguous, and
   ``constants.QB_NAME_ALIASES`` covers Elo spellings the metadata lacks. A name still
   unmapped falls back to the play-by-play passer name, abbreviated as ``F.Last``, when that
   abbreviation belongs to exactly one passer.
2. Quarterback games. ``aggregate_qb_game_stats`` sums one row per
   ``(season, week, team, quarterback)`` over dropbacks, using the team families' dropback
   definition (``pbp.dropback_condition``). Sums only; rates come later.
3. Pre-game rates. For a row in week ``w`` of season ``s``, a quarterback's history is every
   game with ``season * 100 + week`` strictly below ``s * 100 + w``, so the game itself and
   anything later never count. Playoff rows (weeks above the regular season) therefore see
   the whole regular season, and only regular-season plays are ever summed.

Formulas, with ``K = constants.QB_PRIOR_DROPBACKS``:

- ``epa`` per dropback: ``qb_epa_sum / dropbacks``.
- ``sack_rate``: ``sacks / dropbacks``.
- ``any_a``: ``(pass_yards + 20 * pass_tds - 45 * interceptions - sack_yards) /
  (attempts + sacks)``, the standard adjusted net yards per attempt (as in the
  ``nfl-sos-ratings`` reference).
- ``cpoe``: ``cpoe_sum / cpoe_count`` over attempts with a published CPOE (2006 onward).
- League prior for each rate: the same ratio over every quarterback game strictly before the
  row's week. With no earlier game at all the prior, and every rate, is null.
- Career rate (``qb_dropback_epa``, ``qb_sack_rate``, ``qb_any_a``, ``qb_cpoe``):
  ``(career_numerator + K * league_rate) / (career_denominator + K)``. A first start has no
  history and gets exactly the league rate.
- Recent rate (``qb_dropback_epa_recent``, ``qb_any_a_recent``): the last
  ``constants.QB_RECENT_GAMES`` games, ``(recent_numerator + K * career_rate) /
  (recent_denominator + K)``.
- ``qb_history_dropbacks``: career dropbacks, so the model can see how much evidence stands
  behind the rates. Null when the quarterback cannot be identified.

Deviation from the ``nfl-sos-ratings`` reference (read-only): it credits scrambles through
``rusher_player_id``, which this repo's play-by-play cache does not select. Here a scramble
(a dropback with no passer) is credited to the team-game's primary passer, the one with the
most dropbacks that carry a passer id. Scrambles are about 5% of dropbacks, and nearly all of
them belong to the starter.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predictor import constants
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars import pbp

# Per quarterback-game sums carried from play-by-play into the pre-game rates.
QB_GAME_SUM_COLUMNS = [
    "dropbacks",
    "attempts",
    "completions",
    "pass_yards",
    "pass_tds",
    "interceptions",
    "sacks",
    "sack_yards",
    "qb_epa_sum",
    "cpoe_sum",
    "cpoe_count",
]

QB_GAME_SCHEMA: dict[str, type[pl.DataType]] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team_abbr": pl.String,
    "qb_id": pl.String,
    "passer_name": pl.String,
    **dict.fromkeys(QB_GAME_SUM_COLUMNS, pl.Float64),
}

_IDENTITY_SCHEMA: dict[str, type[pl.DataType]] = {"qb_name": pl.String, "qb_id": pl.String}
_KEY = "_qb_key"
_ROW = "_qb_row"
_SIDES = ("away", "home")
# ANY/A bonus per passing touchdown and penalty per interception.
_ANY_A_TD_BONUS = 20.0
_ANY_A_INT_PENALTY = 45.0


def empty_qb_game_frame() -> pl.DataFrame:
    """Return a typed empty quarterback-game frame."""
    return pl.DataFrame(schema=QB_GAME_SCHEMA)


def build_qb_identity(meta_df: pl.DataFrame) -> pl.DataFrame:
    """Map quarterback names to GSIS ids from the nfeloqb metadata columns.

    Args:
        meta_df: Frame with ``name_id`` and ``gsis_id`` columns.

    Returns:
        One row per unambiguous name (``qb_name``, ``qb_id``), plus one row per alias in
        ``constants.QB_NAME_ALIASES`` whose target name is mapped. Rows with a missing name
        or id are dropped, and so is every name that maps to more than one id.

    """
    frame = (
        meta_df.select(
            pl.col("name_id").cast(pl.String).str.strip_chars().alias("qb_name"),
            pl.col("gsis_id").cast(pl.String).str.strip_chars().alias("qb_id"),
        )
        .filter(pl.col("qb_name").is_not_null() & pl.col("qb_id").is_not_null())
        .filter((pl.col("qb_name") != "") & (pl.col("qb_id") != ""))
        .unique()
    )
    ambiguous = frame.group_by("qb_name").len().filter(pl.col("len") > 1).select("qb_name")
    if ambiguous.height:
        log.info(
            "QB identity: dropped %d ambiguous names: %s",
            ambiguous.height,
            sorted(ambiguous["qb_name"].to_list())[:10],
        )
    frame = frame.join(ambiguous, on="qb_name", how="anti")
    aliases = pl.DataFrame(
        {
            "qb_name": list(constants.QB_NAME_ALIASES),
            "target": list(constants.QB_NAME_ALIASES.values()),
        },
        schema={"qb_name": pl.String, "target": pl.String},
    )
    alias_rows = (
        aliases.join(frame.rename({"qb_name": "target"}), on="target", how="inner")
        .select("qb_name", "qb_id")
        .join(frame.select("qb_name"), on="qb_name", how="anti")
    )
    return pl.concat([frame, alias_rows]).sort("qb_name")


def load_qb_identity(path: Path) -> pl.DataFrame:
    """Read the quarterback identity file and build the name-to-id map.

    A missing file is not fatal: it logs a warning and returns an empty map. Quarterbacks are
    then identified only through the abbreviated passer-name fallback of ``attach_qb_features``,
    which leaves a name shared by several passers, and every unmatched name, null.
    """
    if not path.exists():
        log.warning(
            "QB identity file %s not found; quarterbacks are matched by passer name only", path
        )
        return pl.DataFrame(schema=_IDENTITY_SCHEMA)
    meta = pl.read_csv(path, columns=["name_id", "gsis_id"], infer_schema_length=0)
    return build_qb_identity(meta)


def _flag(columns: list[str], column: str) -> pl.Expr:
    """Return a null-safe ``column > 0`` expression, or literal False when absent."""
    if column in columns:
        return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0) > 0
    return pl.lit(False)


def _value(columns: list[str], column: str) -> pl.Expr:
    """Return a null-safe float expression, or literal 0.0 when absent."""
    if column in columns:
        return pl.col(column).cast(pl.Float64, strict=False).fill_null(0.0)
    return pl.lit(0.0)


def aggregate_qb_game_stats(pbp_df: pl.DataFrame) -> pl.DataFrame:
    """Sum play-by-play dropbacks into one row per quarterback game.

    Rows are ``(season, week, team_abbr, qb_id)`` with ``passer_name`` and the sums in
    ``QB_GAME_SUM_COLUMNS``. Definitions (regular season only, dropbacks as in
    ``pbp.dropback_condition``):

    - ``dropbacks``: every dropback, a scramble included (credited as the module docstring
      describes).
    - ``attempts``: dropbacks with a passer id that are not sacks.
    - ``completions``, ``pass_tds``, ``interceptions``: attempts with ``complete_pass``,
      ``pass_touchdown`` and ``interception`` set.
    - ``pass_yards``: ``yards_gained`` summed over attempts.
    - ``sacks``: dropbacks with ``sack`` set; ``sack_yards``: ``-yards_gained`` on them,
      floored at 0.
    - ``qb_epa_sum``: ``qb_epa`` over every dropback.
    - ``cpoe_sum`` / ``cpoe_count``: ``cpoe`` summed and counted over attempts where it is
      published.

    Returns an empty typed frame when there are no plays or no passer ids to key on.
    """
    if pbp_df.height == 0 or "passer_player_id" not in pbp_df.columns:
        if pbp_df.height:
            log.warning("Play-by-play has no passer_player_id; quarterback games are empty")
        return empty_qb_game_frame()
    plays = pbp.regular_season_plays(pbp_df)
    columns = plays.columns
    drops = plays.filter(pbp.dropback_condition(columns)).with_columns(
        pl.col("passer_player_id").cast(pl.String).alias("_passer"),
        (
            pl.col("passer_player_name").cast(pl.String)
            if "passer_player_name" in columns
            else pl.lit(None, dtype=pl.String)
        ).alias("_passer_name"),
    )
    if drops.height == 0:
        return empty_qb_game_frame()

    team_game = ["season", "week", "posteam"]
    primary = (
        drops.filter(pl.col("_passer").is_not_null())
        .group_by([*team_game, "_passer"])
        .len()
        .sort([*team_game, "len", "_passer"], descending=[False, False, False, True, False])
        .unique(subset=team_game, keep="first", maintain_order=True)
        .select(*team_game, pl.col("_passer").alias("_primary"))
    )
    drops = (
        drops.join(primary, on=team_game, how="left")
        .with_columns(pl.coalesce("_passer", "_primary").alias("qb_id"))
        .filter(pl.col("qb_id").is_not_null())
    )

    is_attempt = pl.col("_passer").is_not_null() & ~_flag(columns, "sack")
    is_sack = _flag(columns, "sack")
    yards = _value(columns, "yards_gained")
    has_cpoe = is_attempt & (pl.col("cpoe").is_not_null() if "cpoe" in columns else pl.lit(False))

    def _count(condition: pl.Expr, name: str) -> pl.Expr:
        return condition.cast(pl.Float64).sum().alias(name)

    def _sum(condition: pl.Expr, value: pl.Expr, name: str) -> pl.Expr:
        return pl.when(condition).then(value).otherwise(0.0).sum().alias(name)

    games = drops.group_by([*team_game, "qb_id"]).agg(
        pl.col("_passer_name").drop_nulls().first().alias("passer_name"),
        pl.len().cast(pl.Float64).alias("dropbacks"),
        _count(is_attempt, "attempts"),
        _count(is_attempt & _flag(columns, "complete_pass"), "completions"),
        _sum(is_attempt, yards, "pass_yards"),
        _count(is_attempt & _flag(columns, "pass_touchdown"), "pass_tds"),
        _count(is_attempt & _flag(columns, "interception"), "interceptions"),
        _count(is_sack, "sacks"),
        _sum(is_sack, (-yards).clip(0.0, None), "sack_yards"),
        _value(columns, "qb_epa").sum().alias("qb_epa_sum"),
        _sum(has_cpoe, _value(columns, "cpoe"), "cpoe_sum"),
        _count(has_cpoe, "cpoe_count"),
    )
    return (
        games.rename({"posteam": "team_abbr"})
        .select(pl.col(name).cast(dtype) for name, dtype in QB_GAME_SCHEMA.items())
        .sort(["season", "week", "team_abbr", "qb_id"])
    )


def _keyed(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach the ordering key ``season * 100 + week``."""
    return frame.with_columns((pl.col("season") * 100 + pl.col("week")).cast(pl.Int64).alias(_KEY))


def _history_tables(qb_games: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return career, recent-window and league running sums, each through its key.

    Each table holds, at a key, the sums over games up to and including that key; the
    strictly-before rule is applied by the as-of join in ``attach_qb_features``.
    """
    per_game = (
        _keyed(qb_games)
        .group_by(["qb_id", _KEY])
        .agg(pl.col(c).sum() for c in QB_GAME_SUM_COLUMNS)
        .sort(["qb_id", _KEY])
    )
    career = per_game.select(
        "qb_id",
        _KEY,
        *[pl.col(c).cum_sum().over("qb_id").alias(f"c_{c}") for c in QB_GAME_SUM_COLUMNS],
    ).sort(_KEY)
    recent = per_game.select(
        "qb_id",
        _KEY,
        *[
            pl.col(c)
            .rolling_sum(window_size=constants.QB_RECENT_GAMES, min_samples=1)
            .over("qb_id")
            .alias(f"r_{c}")
            for c in QB_GAME_SUM_COLUMNS
        ],
    ).sort(_KEY)
    league = (
        per_game.group_by(_KEY)
        .agg(pl.col(c).sum() for c in QB_GAME_SUM_COLUMNS)
        .sort(_KEY)
        .select(_KEY, *[pl.col(c).cum_sum().alias(f"l_{c}") for c in QB_GAME_SUM_COLUMNS])
    )
    return career, recent, league


def _abbreviated(name: pl.Expr) -> pl.Expr:
    """Return ``F.Last`` from ``First Last ...``, the play-by-play passer-name style."""
    parts = name.str.strip_chars().str.split(" ")
    return pl.concat_str(
        [parts.list.first().str.slice(0, 1), pl.lit("."), parts.list.slice(1).list.join(" ")]
    )


def _passer_name_map(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Return play-by-play passer names that belong to exactly one quarterback id."""
    pairs = qb_games.select("passer_name", "qb_id").drop_nulls().unique()
    unique_names = pairs.group_by("passer_name").len().filter(pl.col("len") == 1)
    return pairs.join(unique_names.select("passer_name"), on="passer_name", how="inner")


def _resolve_ids(
    targets: pl.DataFrame, identity: pl.DataFrame, passer_names: pl.DataFrame
) -> pl.DataFrame:
    """Attach ``qb_id`` to ``targets`` (column ``qb_name``) via identity, then passer names."""
    resolved = targets.join(identity, on="qb_name", how="left")
    fallback = (
        resolved.filter(pl.col("qb_id").is_null() & pl.col("qb_name").is_not_null())
        .with_columns(_abbreviated(pl.col("qb_name")).alias("passer_name"))
        .drop("qb_id")
        .join(passer_names, on="passer_name", how="left")
        .drop("passer_name")
    )
    matched = resolved.filter(pl.col("qb_id").is_not_null() | pl.col("qb_name").is_null())
    return pl.concat([matched, fallback.select(matched.columns)]).sort(_ROW)


def _shrink(numerator: pl.Expr, denominator: pl.Expr, prior: pl.Expr) -> pl.Expr:
    """Return ``(numerator + K * prior) / (denominator + K)`` with ``K = QB_PRIOR_DROPBACKS``."""
    k = float(constants.QB_PRIOR_DROPBACKS)
    return (numerator + k * prior) / (denominator + k)


def _ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """Return ``numerator / denominator``, null when the denominator is not positive."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(None)


def _any_a_parts(prefix: str) -> tuple[pl.Expr, pl.Expr]:
    """Return the ANY/A numerator and denominator over columns with ``prefix``."""
    numerator = (
        pl.col(f"{prefix}pass_yards")
        + _ANY_A_TD_BONUS * pl.col(f"{prefix}pass_tds")
        - _ANY_A_INT_PENALTY * pl.col(f"{prefix}interceptions")
        - pl.col(f"{prefix}sack_yards")
    )
    return numerator, pl.col(f"{prefix}attempts") + pl.col(f"{prefix}sacks")


def _side_features(
    targets: pl.DataFrame,
    career: pl.DataFrame,
    recent: pl.DataFrame,
    league: pl.DataFrame,
    side: str,
) -> pl.DataFrame:
    """Compute one side's quarterback features for rows keyed by ``_ROW``, ``_KEY``, ``qb_id``."""
    # Strictly before: the row's own week never matches (``allow_exact_matches=False``).
    # Every frame is sorted by the key just before joining, which also orders the key within
    # each quarterback, so the sortedness check Polars cannot run with `by` groups is moot.
    frame = (
        targets.sort(_KEY)
        .join_asof(
            career,
            on=_KEY,
            by="qb_id",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join_asof(
            recent,
            on=_KEY,
            by="qb_id",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join_asof(league, on=_KEY, strategy="backward", allow_exact_matches=False)
    )
    known = pl.col("qb_id").is_not_null()
    frame = frame.with_columns(
        pl.when(known).then(pl.col(f"{p}{c}").fill_null(0.0)).otherwise(None).alias(f"{p}{c}")
        for p in ("c_", "r_")
        for c in QB_GAME_SUM_COLUMNS
    )
    league_any_a = _ratio(*_any_a_parts("l_"))
    career_epa = _shrink(
        pl.col("c_qb_epa_sum"),
        pl.col("c_dropbacks"),
        _ratio(pl.col("l_qb_epa_sum"), pl.col("l_dropbacks")),
    )
    career_any_a = _shrink(*_any_a_parts("c_"), league_any_a)
    frame = frame.with_columns(career_epa.alias("_epa"), career_any_a.alias("_any_a"))
    stats = {
        "qb_dropback_epa": pl.col("_epa"),
        "qb_dropback_epa_recent": _shrink(
            pl.col("r_qb_epa_sum"), pl.col("r_dropbacks"), pl.col("_epa")
        ),
        "qb_cpoe": _shrink(
            pl.col("c_cpoe_sum"),
            pl.col("c_cpoe_count"),
            _ratio(pl.col("l_cpoe_sum"), pl.col("l_cpoe_count")),
        ),
        "qb_sack_rate": _shrink(
            pl.col("c_sacks"),
            pl.col("c_dropbacks"),
            _ratio(pl.col("l_sacks"), pl.col("l_dropbacks")),
        ),
        "qb_any_a": pl.col("_any_a"),
        "qb_any_a_recent": _shrink(*_any_a_parts("r_"), pl.col("_any_a")),
        "qb_history_dropbacks": pl.col("c_dropbacks"),
    }
    return frame.select(_ROW, *[expr.alias(f"{side}_{name}") for name, expr in stats.items()])


def attach_qb_features(
    games: pl.DataFrame, qb_games: pl.DataFrame, identity: pl.DataFrame
) -> pl.DataFrame:
    """Add the quarterback family for ``away_qb`` and ``home_qb`` to every game row.

    Args:
        games: Game rows with ``season``, ``week``, ``away_qb`` and ``home_qb``.
        qb_games: Quarterback-game sums from ``aggregate_qb_game_stats``.
        identity: Name-to-id map from ``build_qb_identity``.

    Returns:
        ``games`` in its original row order with ``away_<stat>``, ``home_<stat>`` and
        ``<stat>_diff`` (away minus home) for every stat in ``constants.QB_PBP_STATS``.
        Formulas are in the module docstring.

    """
    new_columns = [
        f"{prefix}{stat}" for stat in constants.QB_PBP_STATS for prefix in ("away_", "home_")
    ] + [f"{stat}_diff" for stat in constants.QB_PBP_STATS]
    base = games.drop([c for c in new_columns if c in games.columns])
    keyed = _keyed(base.with_row_index(_ROW))
    career, recent, league = _history_tables(qb_games)
    passer_names = _passer_name_map(qb_games)

    out = keyed
    for side in _SIDES:
        targets = keyed.select(_ROW, _KEY, pl.col(f"{side}_qb").cast(pl.String).alias("qb_name"))
        resolved = _resolve_ids(targets, identity, passer_names)
        named = resolved.filter(pl.col("qb_name").is_not_null())
        unmatched = named.filter(pl.col("qb_id").is_null())
        if named.height:
            log.info(
                "QB features (%s): %d of %d rows unmatched (%.2f%%); names: %s",
                side,
                unmatched.height,
                named.height,
                100.0 * unmatched.height / named.height,
                sorted(set(unmatched["qb_name"].to_list()))[:10],
            )
        features = _side_features(resolved, career, recent, league, side)
        out = out.join(features, on=_ROW, how="left")

    out = out.with_columns(
        (pl.col(f"away_{stat}") - pl.col(f"home_{stat}")).alias(f"{stat}_diff")
        for stat in constants.QB_PBP_STATS
    )
    return out.sort(_ROW).drop(_ROW, _KEY)
