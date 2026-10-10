"""Quarterback-specific opponent profile helpers."""

import polars as pl

from nfl_predictor.constants import ALIAS_TO_CANONICAL
from nfl_predictor.season_stats.opponent_stats import is_division_opponent
from nfl_predictor.season_stats.pooled_rates import is_rate_part, pooled_rate, rate_parts
from nfl_predictor.season_stats.qb_stats import QB_RATES, qb_rate_exprs, select_primary_qb_rows
from nfl_predictor.season_stats.team_stats import compute_team_stats_excluding_opponents

DEFENSIVE_CONTEXT_COLS: list[str] = [
    "points_allowed",
    "def_sacks",
    "def_interceptions",
    "def_pass_defended",
    "def_tackles_for_loss",
    "def_qb_hits",
]

# Rates a QB game row lacks or derives, appended after the per-game columns, in this order.
_ALLOWED_RATES_APPENDED = (
    "qb_yards_per_attempt",
    "qb_touchdown_rate",
    "qb_interception_rate",
    "qb_epa_per_dropback",
    "qb_pass_yards_per_dropback",
    "qb_sack_rate",
    "qb_td_int_margin_rate",
    "qb_any_a",
)


def _normalize_team_abbreviations(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Normalize known source-specific team abbreviations in selected columns."""
    exprs = [
        pl.col(column).replace(ALIAS_TO_CANONICAL).alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def _summed(column: str) -> pl.Expr:
    """Return the total of one QB game-row column over the games a defense faced."""
    return pl.col(column).sum()


def _qb_allowed_rate_exprs(columns: set[str]) -> list[pl.Expr]:
    """Return the defense-allowed rates a QB game row lacks or derives, pooled over its games.

    Each is rebuilt from the totals of its inputs (``QB_RATES``); one whose inputs are absent is
    left out.
    """
    return [
        expr.alias(f"qopp_{expr.meta.output_name()}")
        for expr in qb_rate_exprs(columns, _summed, _ALLOWED_RATES_APPENDED)
    ]


def _allowed_stat_expr(column: str, columns: set[str], with_parts: set[str]) -> pl.Expr:
    """Return one QB stat over the games a defense faced.

    A rate is pooled over the games: one in ``with_parts`` (CPOE) through its hidden parts, any
    other through the totals of its inputs among ``columns`` (``QB_RATES``). A count is averaged
    per game.
    """
    if column in with_parts:
        return pooled_rate(column)
    rate = QB_RATES.get(column)
    if rate is not None and columns.issuperset(rate.inputs):
        return rate.build(_summed)
    return pl.col(column).mean()


_PAIR_KEYS = ["opponent", "excluded_opponent"]


def _qb_allowed_stats_by_pair(
    weekly_df: pl.DataFrame, qb_df: pl.DataFrame, pairs: pl.DataFrame
) -> pl.DataFrame:
    """Return the QB stats each defense allowed in its games not against the team it is paired with.

    ``pairs`` holds the defense as ``team`` and the evaluated team as ``excluded_opponent``. The
    result has one row per pair whose defense faced a passer in ``qb_df`` in such a game, keyed
    ``opponent`` (the defense) and ``excluded_opponent``: counts per game, rates pooled over the
    games (``_allowed_stat_expr``).
    """
    qb_stat_cols = [
        col
        for col, dtype in zip(qb_df.columns, qb_df.dtypes, strict=True)
        if dtype.is_numeric()
        and col != "week"
        and col not in _ALLOWED_RATES_APPENDED
        and not is_rate_part(col)
    ]
    empty = pl.DataFrame(schema=dict.fromkeys(_PAIR_KEYS, pl.String))
    if not qb_stat_cols:
        return empty

    faced = weekly_df.select(
        pl.col("opponent_team").alias("opponent"), pl.col("team").alias("offense"), "week"
    )
    # Both joins keep the left rows' order, then the right's: the aggregation below sums floats
    # in row order, so an unspecified join order could move a sum's last bits.
    qb_allowed = (
        pairs.select(pl.col("team").alias("opponent"), "excluded_opponent")
        .unique(maintain_order=True)
        .join(faced, on="opponent", how="inner", maintain_order="left_right")
        .filter(pl.col("offense") != pl.col("excluded_opponent"))
        .join(
            qb_df,
            left_on=["offense", "week"],
            right_on=["team_abbr", "week"],
            how="inner",
            maintain_order="left_right",
        )
    )
    if qb_allowed.is_empty():
        return empty

    columns = set(qb_allowed.columns)
    with_parts = set(rate_parts(qb_allowed.columns))
    agg_exprs = [
        _allowed_stat_expr(column, columns, with_parts).alias(f"qopp_{column}")
        for column in qb_stat_cols
    ]
    agg_exprs.extend(_qb_allowed_rate_exprs(columns))
    return qb_allowed.group_by(_PAIR_KEYS, maintain_order=True).agg(agg_exprs)


def _select_primary_qb_games(qb_df: pl.DataFrame) -> pl.DataFrame:
    """Return one primary quarterback row per team-week, as ``qb_stats`` picks it."""
    return select_primary_qb_rows(qb_df)


def _get_faced_opponents(qb_games: pl.DataFrame) -> list[str]:
    """Return unique faced opponents in week order from a quarterback's actual game rows."""
    if qb_games.is_empty() or "opponent_team" not in qb_games.columns:
        return []
    sort_cols = [col for col in ("week", "opponent_team") if col in qb_games.columns]
    source = qb_games.sort(sort_cols) if sort_cols else qb_games
    opponents = source.select("opponent_team").to_series().to_list()
    return list(dict.fromkeys(str(opponent) for opponent in opponents))


def _details_key(qb_row: dict[str, object], qb_keys: list[str], team_label: str) -> str:
    """Return the details-map key for one quarterback season row."""
    for key in qb_keys:
        value = qb_row.get(key)
        if value is not None:
            return str(value)
    return team_label


def _qb_identity_filter(qb_row: dict[str, object], qb_keys: list[str]) -> pl.Expr:
    """Return the most specific QB identity filter available for one season row."""
    for key in qb_keys:
        value = qb_row.get(key)
        if value is not None:
            return pl.col(key) == pl.lit(value)
    return pl.lit(False)


def _qb_opponent_rows(
    team_rows: pl.DataFrame,
    allowed_rows: pl.DataFrame,
    opponents: list[str],
    evaluated_team: str,
) -> tuple[pl.DataFrame, list[dict[str, str | bool | int]]]:
    """Return one head-to-head-excluded profile row per faced opponent, plus detail records.

    ``team_rows`` and ``allowed_rows`` are keyed by ``opponent`` and ``excluded_opponent``: each
    opponent's own stats, and the QB stats it allowed, in its games not against the evaluated
    team. An opponent without such games gets a detail record and no row.
    """
    faced = pl.DataFrame(
        {"opponent": opponents, "excluded_opponent": [evaluated_team] * len(opponents)},
        schema=dict.fromkeys(_PAIR_KEYS, pl.String),
    )
    rows = faced.join(team_rows, on=_PAIR_KEYS, how="inner", maintain_order="left").join(
        allowed_rows, on=_PAIR_KEYS, how="left", maintain_order="left"
    )
    games = dict(rows.select("opponent", "games_included").iter_rows())
    team_details: list[dict[str, str | bool | int]] = [
        {
            "opponent": opponent,
            "division": is_division_opponent(evaluated_team, opponent),
            "games_included": int(games.get(opponent, 0)),
        }
        for opponent in opponents
    ]
    return rows, team_details


def _qb_profile_agg_exprs(combined: pl.DataFrame) -> list[pl.Expr]:
    """Return the opponent-profile averages, one equal-weight entry per faced opponent."""
    available_cols = [col for col in DEFENSIVE_CONTEXT_COLS if col in combined.columns]
    available_cols.extend(
        col for col in combined.columns if col.startswith("qopp_qb_") and col not in available_cols
    )
    return [
        pl.col(col).mean().alias(col if col.startswith("qopp_") else f"qopp_{col}")
        for col in available_cols
    ]


def compute_qb_opponent_profiles(
    weekly_df: pl.DataFrame,
    qb_df: pl.DataFrame,
    qb_season_df: pl.DataFrame,
) -> tuple[pl.DataFrame | None, dict[str, list[dict[str, str | bool | int]]]]:
    """Compute QB opponent profiles for each individual quarterback season row."""
    weekly_df = _normalize_team_abbreviations(weekly_df, ["team", "opponent_team"])
    qb_df = _normalize_team_abbreviations(qb_df, ["team_abbr"])
    qb_season_df = _normalize_team_abbreviations(qb_season_df, ["team"])

    qb_keys = [key for key in ("qb_id", "qb_name") if key in qb_season_df.columns]
    if not qb_keys:
        msg = "qb_season_df needs a qb_id or qb_name column to profile each quarterback"
        raise ValueError(msg)

    details: dict[str, list[dict[str, str | bool | int]]] = {}
    profile_rows: list[pl.DataFrame] = []

    qb_rows = qb_season_df.select([*qb_keys, "team"]).to_dicts()
    primary_qb_df = _select_primary_qb_games(qb_df)

    qb_games_with_opponents = primary_qb_df.join(
        weekly_df.select(["team", "week", "opponent_team"]),
        left_on=["team_abbr", "week"],
        right_on=["team", "week"],
        how="inner",
    )

    faced: list[tuple[dict[str, object], str, str, list[str]]] = []
    for qb_row in qb_rows:
        team_label = str(qb_row.get("team", ""))
        qb_label = _details_key(qb_row, qb_keys, team_label)
        qb_games = qb_games_with_opponents.filter(_qb_identity_filter(qb_row, qb_keys))
        faced.append((qb_row, team_label, qb_label, _get_faced_opponents(qb_games)))

    # Every (opponent, evaluated team) pair at once: each opponent profiled without its games
    # against the passer's team, as its own stats and as the passing it allowed.
    pairs = pl.DataFrame(
        [(opponent, team) for _, team, _, opponents in faced for opponent in opponents],
        schema={"team": pl.String, "excluded_opponent": pl.String},
        orient="row",
    )
    team_rows = compute_team_stats_excluding_opponents(weekly_df, pairs).rename(
        {"team": "opponent"}
    )
    allowed_rows = _qb_allowed_stats_by_pair(weekly_df, primary_qb_df, pairs)

    for qb_row, team_label, qb_label, opponents in faced:
        if not opponents:
            details[qb_label] = []
            continue

        combined, team_details = _qb_opponent_rows(team_rows, allowed_rows, opponents, team_label)
        details[qb_label] = team_details

        if combined.is_empty():
            continue
        agg_exprs = _qb_profile_agg_exprs(combined)
        if not agg_exprs:
            continue

        key_exprs = [pl.lit(qb_row[key]).alias(key) for key in qb_keys]
        profile_rows.append(
            combined.select([*key_exprs, pl.lit(team_label).alias("team"), *agg_exprs])
        )

    if not profile_rows:
        return None, details

    sort_keys = [key for key in ["team", *qb_keys] if key in profile_rows[0].columns]
    return pl.concat(profile_rows).sort(sort_keys), details
