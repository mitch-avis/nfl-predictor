r"""Power rankings and projected standings.

This module provides utilities to:
- compute fan-friendly 1-10 power ratings per team
- compute projected standings based on current record + expected future results

Design goals:
- Deterministic outputs under fixed inputs.
- Works off model-predicted win probabilities for future games.
- Uses the repo's canonical division/conference mapping from `nfl_predictor.constants`.

Notes on methodology
-------------------
Power ratings are estimated by fitting a simple Bradley-Terry style latent-strength model on
per-game home win probabilities:

  logit(p_home) \approx r_home - r_away + h

Where `r_team` is the latent team strength and `h` is a home-field advantage term.

To incorporate *both* past games and future expectations:
- completed games contribute targets derived from the observed result
- future games contribute targets from the trained model's predicted home win probability

The resulting latent ratings are then scaled to a 1-10 power-rating scale.

Teams can instead be ranked on the ETL's pre-week schedule-adjusted strength composite,
read from one week of the per-team strength snapshot file. The composite is a weighted
mean of within-week z-scores, not a win probability, so it is first expressed in points
through that week's own SRS and then mapped onto the same scales; see
`rank_teams_on_composite`.

The second half of the module runs a ranking end to end for one season through one week:
it loads current records from the schedule, predicts the remaining games with a trained
model, reads the strength snapshot or builds the Bradley-Terry fit inputs, and writes the
output tables (`compute_power_rankings`, `write_ranking_outputs`). The rankings command and
the weekly run both call it.

This is a display/reporting artifact only; it does not affect training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl

from nfl_predictor import constants
from nfl_predictor.ml import ml_model_core
from nfl_predictor.ml.ml_model_core import margin_to_home_win_prob
from nfl_predictor.utils.logger import log
from nfl_predictor.utils.polars.strength_snapshot import COMPOSITE_WEIGHTS

# Internal column used to carry per-game fit weights; never published.
_WEIGHT_COLUMN = "__fit_weight"

# Optional column on the ratings input carrying each game's weight in the fit.
FIT_WEIGHT_COLUMN = "fit_weight"

# The snapshot column the composite method ranks on.
COMPOSITE_COLUMN = "adj_strength_composite"

# Published next to a composite rank: the weighted components, the SRS the points scale
# is read from, and the in-season games behind the solve.
COMPOSITE_PUBLISHED_COLUMNS: tuple[str, ...] = (
    *COMPOSITE_WEIGHTS,
    "adj_srs",
    "strength_games_played",
)

# Fewest teams with both a composite and an SRS for the points scale to be estimated.
_MIN_TEAMS_FOR_POINTS_SCALE = 3


@dataclass(frozen=True)
class PowerRatingsResult:
    """Outputs for power rankings and projected standings."""

    power_rankings: pd.DataFrame
    projected_standings: pd.DataFrame
    projected_division_standings: pd.DataFrame


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x_arr))


def clamp_prob(p: pd.Series | np.ndarray, *, eps: float = 0.03) -> np.ndarray:
    """Clamp probabilities into (eps, 1-eps) to keep logits finite."""
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, eps, 1.0 - eps)


def outcome_to_home_prob(
    home_score: pd.Series,
    away_score: pd.Series,
    *,
    eps: float = 0.03,
    target: str = "binary",
) -> np.ndarray:
    """Convert an observed game outcome into a probability target.

    Two targets are supported, and ties are 0.5 under both:

    - ``"binary"``: a win is ``1 - eps`` and a loss is ``eps``, so every win counts
      the same regardless of margin. This is the historical behavior.
    - ``"margin"``: the observed point margin is mapped through the same
      margin-to-win-probability curve the model itself uses, so a 28-point win is
      stronger evidence than a 1-point win. Values are clamped into
      ``(eps, 1 - eps)`` so the logit stays finite.

    Args:
        home_score: Home team score.
        away_score: Away team score.
        eps: Clamp applied to keep logits finite.
        target: Either ``"binary"`` or ``"margin"``.

    Returns:
        Array of home win probability targets.

    Raises:
        ValueError: If ``target`` is not a recognized name.

    """
    if target not in {"binary", "margin"}:
        raise ValueError(f"target must be 'binary' or 'margin', got: {target!r}")

    home = pd.to_numeric(home_score, errors="coerce")
    away = pd.to_numeric(away_score, errors="coerce")
    margin = home - away
    margin_values = margin.to_numpy(dtype=float)

    if target == "margin":
        probs = np.asarray(margin_to_home_win_prob(margin_values), dtype=float)
        probs = np.where(margin_values == 0.0, 0.5, probs)
        probs = np.clip(probs, eps, 1.0 - eps)
        return np.where(np.isnan(margin_values), np.nan, probs)

    p = np.full(len(margin), np.nan, dtype=float)
    p[margin > 0] = 1.0 - eps
    p[margin < 0] = eps
    p[margin == 0] = 0.5
    return p


def fit_bradley_terry_ratings(
    games: pd.DataFrame,
    *,
    home_team_col: str = "home_abbr",
    away_team_col: str = "away_abbr",
    p_home_col: str = "p_home",
    include_home_advantage: bool = True,
    ridge_alpha: float = 1.0,
    sample_weights: pd.Series | np.ndarray | None = None,
) -> tuple[pd.Series, float]:
    """Fit latent team ratings from per-game home win probabilities.

    Args:
        games: Must include home/away team columns and a probability column.
        home_team_col: Home team abbreviation column.
        away_team_col: Away team abbreviation column.
        p_home_col: Home win probability column.
        include_home_advantage: If True, fit a global home-field advantage term.
        ridge_alpha: L2 regularization strength.
        sample_weights: Optional non-negative per-game weights. Rows are scaled by the
            square root of their weight, which turns the solve into ordinary weighted
            least squares. Used to down-weight older seasons so the fit describes
            current strength rather than franchise history. Uniform weights reproduce
            the unweighted fit exactly.

    Returns:
        (team_rating_series, home_advantage)

    """
    required = {home_team_col, away_team_col, p_home_col}
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"games missing required columns: {missing}")

    df = games[[home_team_col, away_team_col, p_home_col]].copy()
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape[0] != df.shape[0]:
            raise ValueError("sample_weights must have one entry per game")
        df[_WEIGHT_COLUMN] = weights
    df = df.dropna().copy()
    if df.empty:
        raise ValueError("No games available after dropping nulls")

    teams: list[str] = sorted(
        set(df[home_team_col].astype(str).tolist()) | set(df[away_team_col].astype(str).tolist())
    )
    team_to_idx = {t: i for i, t in enumerate(teams)}

    n_games = len(df)
    n_teams = len(teams)
    extra = 1 if include_home_advantage else 0

    x = np.zeros((n_games, n_teams + extra), dtype=float)
    home_idx = df[home_team_col].astype(str).map(team_to_idx).to_numpy()
    away_idx = df[away_team_col].astype(str).map(team_to_idx).to_numpy()
    x[np.arange(n_games), home_idx] = 1.0
    x[np.arange(n_games), away_idx] = -1.0
    if include_home_advantage:
        x[:, -1] = 1.0

    p = clamp_prob(df[p_home_col].to_numpy())
    y = _logit(p)

    if _WEIGHT_COLUMN in df.columns:
        # Weighted least squares: scaling each row by sqrt(w) makes the ordinary normal
        # equations below minimize the weighted squared error.
        root_weights = np.sqrt(np.clip(df[_WEIGHT_COLUMN].to_numpy(dtype=float), 0.0, None))
        x = x * root_weights[:, np.newaxis]
        y = y * root_weights

    xtx = x.T @ x
    xty = x.T @ y

    reg = ridge_alpha * np.eye(xtx.shape[0], dtype=float)
    coef = np.linalg.solve(xtx + reg, xty)

    if include_home_advantage:
        team_coef = coef[:-1]
        home_adv = float(coef[-1])
    else:
        team_coef = coef
        home_adv = 0.0

    # Center so the average team is ~0 (ratings are relative).
    team_coef = team_coef - float(np.mean(team_coef))
    ratings = pd.Series(team_coef, index=teams, name="rating_raw")
    return ratings, home_adv


def win_prob_to_power_1_10(p: np.ndarray | float) -> np.ndarray:
    """Map a win probability against an average team onto the 1-10 display scale.

    Formula: ``power_rating_1_10 = 1 + 9 * p``, so 0 maps to 1, 0.5 to 5.5 and 1 to 10.
    """
    return 1.0 + 9.0 * np.asarray(p, dtype=float)


def win_prob_to_power_0_10(p: np.ndarray | float) -> np.ndarray:
    """Map a win probability against an average team onto the 0-10 display scale.

    Formula: ``power_rating_0_10 = 10 * p``.
    """
    return 10.0 * np.asarray(p, dtype=float)


def scale_ratings_1_to_10(ratings_raw: pd.Series) -> pd.Series:
    """Scale raw ratings to a stable 1-10 range.

    This uses an *absolute* mapping rather than per-run min/max scaling.

    Interpretation:
    - `rating_raw` is centered so the average team is ~0.
    - `sigmoid(rating_raw)` is the implied win probability vs an average team
      on a neutral field (under the Bradley–Terry logit model).
    - We map that probability into a display-friendly 1..10 scale.
    """
    x = ratings_raw.astype(float)
    p_vs_avg = _sigmoid(x.to_numpy())
    scaled = win_prob_to_power_1_10(p_vs_avg)
    return pd.Series(scaled, index=x.index, name="power_rating_1_10").round(2)


def ratings_to_power_0_to_10(ratings_raw: pd.Series) -> pd.Series:
    """Convert raw ratings to a stable 0-10 scale.

    `10 * sigmoid(rating_raw)` is interpretable as a 0–10 strength score derived
    from win probability vs an average team on a neutral field.
    """
    x = ratings_raw.astype(float)
    scaled = win_prob_to_power_0_10(_sigmoid(x.to_numpy()))
    return pd.Series(scaled, index=x.index, name="power_rating_0_10").round(2)


def composite_points_scale(snapshot: pd.DataFrame) -> float:
    """Return how many points of adjusted margin one composite unit is worth.

    The composite has no units: it is a weighted mean of within-week z-scores. Its points
    equivalent is read from the same snapshot, as the ordinary least-squares slope of each
    team's adjusted SRS (points per game against an average schedule) on its composite::

        beta = cov(adj_srs, composite) / var(composite)

    over the teams that have both values.

    Returns:
        ``beta``, or NaN when fewer than three teams have both values, when the composite
        has no spread, or when the slope is not positive: a scale that would flatten or
        invert the ranking is worse than none.

    """
    if not {COMPOSITE_COLUMN, "adj_srs"} <= set(snapshot.columns):
        return float("nan")
    composite = pd.to_numeric(snapshot[COMPOSITE_COLUMN], errors="coerce").to_numpy(dtype=float)
    srs = pd.to_numeric(snapshot["adj_srs"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(composite) & np.isfinite(srs)
    if int(usable.sum()) < _MIN_TEAMS_FOR_POINTS_SCALE:
        return float("nan")
    x = composite[usable]
    y = srs[usable]
    variance = float(np.var(x))
    if variance <= 0.0:
        return float("nan")
    slope = float(np.mean((x - x.mean()) * (y - y.mean()))) / variance
    return slope if slope > 0.0 else float("nan")


def rank_teams_on_composite(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Rate every team in one week's strength snapshot on the display scales.

    ``adj_strength_composite`` is unitless and is not a win probability, so the
    Bradley-Terry ``sigmoid(rating)`` mapping does not apply to it. It is first expressed
    in points with the slope ``beta`` from `composite_points_scale`, then turned into a
    win probability against an average team on a neutral field through the model's own
    normal margin curve, and only then placed on the scales::

        points_vs_average = beta * (composite - mean(composite))
        p_vs_average = Phi(points_vs_average / SCORE_DIFF_STD_DEV)
        power_rating_1_10 = 1 + 9 * p_vs_average
        power_rating_0_10 = 10 * p_vs_average

    With ``beta > 0`` both scales rise with the composite and stay bounded, and an exactly
    average team sits at 5.5 and 5.0. When ``beta`` is unavailable the points and scale
    columns are null and the composite alone orders the teams.

    Args:
        snapshot: One row per team with ``team_abbr`` and ``adj_strength_composite``. The
            components in `COMPOSITE_PUBLISHED_COLUMNS` are carried through when present.

    Returns:
        One row per team with ``team_abbr``, ``adj_strength_composite``,
        ``points_vs_average``, ``power_rating_1_10``, ``power_rating_0_10`` and the
        published components.

    """
    frame = pd.DataFrame({"team_abbr": snapshot["team_abbr"].astype(str).to_numpy()})
    for column in (COMPOSITE_COLUMN, *COMPOSITE_PUBLISHED_COLUMNS):
        if column in snapshot.columns:
            frame[column] = pd.to_numeric(snapshot[column], errors="coerce").to_numpy(dtype=float)
        else:
            frame[column] = np.nan

    beta = composite_points_scale(frame)
    composite = frame[COMPOSITE_COLUMN].to_numpy(dtype=float)
    if np.isnan(beta):
        points = np.full(len(frame), np.nan)
    else:
        points = beta * (composite - float(np.nanmean(composite)))
    p_vs_average = np.asarray(margin_to_home_win_prob(points), dtype=float)

    frame.insert(2, "points_vs_average", np.round(points, 2))
    frame.insert(3, "power_rating_1_10", np.round(win_prob_to_power_1_10(p_vs_average), 2))
    frame.insert(4, "power_rating_0_10", np.round(win_prob_to_power_0_10(p_vs_average), 2))
    return frame


def compute_projected_standings(
    current_records: pd.DataFrame,
    future_games: pd.DataFrame,
    *,
    season: int,
    p_home_col: str = "home_win_prob",
) -> pd.DataFrame:
    """Compute expected final record = current record + expected future results."""
    required_records = {"team_abbr", "wins", "losses", "ties", "games_played"}
    missing = sorted(required_records - set(current_records.columns))
    if missing:
        raise ValueError(f"current_records missing required columns: {missing}")

    df_rec = current_records.copy()
    df_rec["team_abbr"] = df_rec["team_abbr"].astype(str)

    # Expected wins from future games.
    future = future_games.copy()
    required_future = {"home_abbr", "away_abbr", p_home_col}
    missing_f = sorted(required_future - set(future.columns))
    if missing_f:
        # If there are no future games (common after the regular season), treat as
        # "no remaining schedule" instead of erroring.
        if len(future) == 0:
            future = pd.DataFrame(columns=sorted(required_future))
        else:
            raise ValueError(f"future_games missing required columns: {missing_f}")

    future["home_abbr"] = future["home_abbr"].astype(str)
    future["away_abbr"] = future["away_abbr"].astype(str)
    p_home = pd.to_numeric(future[p_home_col], errors="coerce")

    # A team has no record row until it has played, yet it still has a schedule to project
    # (every team before week 1), so it starts from a zero record.
    unrecorded = sorted(
        (set(future["home_abbr"]) | set(future["away_abbr"])) - set(df_rec["team_abbr"])
    )
    if unrecorded:
        zero_records = pd.DataFrame(
            {"team_abbr": unrecorded, "wins": 0, "losses": 0, "ties": 0, "games_played": 0}
        )
        df_rec = zero_records if df_rec.empty else pd.concat([df_rec, zero_records])

    home_exp = (
        future.assign(exp_wins=p_home)
        .groupby("home_abbr", as_index=False)
        .agg(exp_wins=("exp_wins", "sum"))
        .rename(columns={"home_abbr": "team_abbr"})
    )
    away_exp = (
        future.assign(exp_wins=1.0 - p_home)
        .groupby("away_abbr", as_index=False)
        .agg(exp_wins=("exp_wins", "sum"))
        .rename(columns={"away_abbr": "team_abbr"})
    )
    exp = (
        pd.concat([home_exp, away_exp], ignore_index=True)
        .groupby("team_abbr", as_index=False)
        .agg(exp_wins=("exp_wins", "sum"))
    )

    out = df_rec.merge(exp, on="team_abbr", how="left").fillna({"exp_wins": 0.0})

    # Remaining games by schedule slice.
    remaining_home = (
        future.groupby("home_abbr", as_index=False)
        .size()
        .rename(columns={"home_abbr": "team_abbr"})
    )
    remaining_away = (
        future.groupby("away_abbr", as_index=False)
        .size()
        .rename(columns={"away_abbr": "team_abbr"})
    )
    remaining = (
        pd.concat([remaining_home, remaining_away], ignore_index=True)
        .groupby("team_abbr", as_index=False)
        .agg(size=("size", "sum"))
        .rename(columns={"size": "games_remaining"})
    )
    out = out.merge(remaining, on="team_abbr", how="left").fillna({"games_remaining": 0})
    out["games_remaining"] = out["games_remaining"].astype(int)

    out["projected_wins"] = out["wins"].astype(float) + out["exp_wins"].astype(float)
    out["projected_losses"] = out["losses"].astype(float) + (
        out["games_remaining"].astype(float) - out["exp_wins"].astype(float)
    )

    out["exp_wins"] = out["exp_wins"].astype(float).round(3)
    out["projected_wins"] = out["projected_wins"].astype(float).round(3)
    out["projected_losses"] = out["projected_losses"].astype(float).round(3)

    out["division"] = out["team_abbr"].map(constants.TEAM_TO_DIVISION)
    out["conference"] = out["team_abbr"].map(constants.TEAM_TO_CONFERENCE)

    total_games = (
        out["wins"].astype(float)
        + out["losses"].astype(float)
        + out["ties"].astype(float)
        + out["games_remaining"].astype(float)
    )
    out["projected_win_pct"] = (out["projected_wins"] / total_games.replace(0.0, np.nan)).fillna(
        0.0
    )
    out["projected_win_pct"] = out["projected_win_pct"].fillna(0.0).round(4)

    out["season"] = int(season)
    out = out.sort_values(
        ["projected_win_pct", "projected_wins", "team_abbr"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return out


def _team_frame(team_universe: list[str]) -> pd.DataFrame:
    """Return one row per team with its division and conference."""
    base = pd.DataFrame({"team_abbr": team_universe})
    base["division"] = base["team_abbr"].map(constants.TEAM_TO_DIVISION).fillna("Unknown")
    base["conference"] = base["team_abbr"].map(constants.TEAM_TO_CONFERENCE).fillna("Unknown")
    return base


def _with_records(rankings: pd.DataFrame, current_records: pd.DataFrame) -> pd.DataFrame:
    """Attach each team's current record, zero for a team with no games yet."""
    return rankings.merge(
        current_records[["team_abbr", "wins", "losses", "ties", "games_played"]],
        on="team_abbr",
        how="left",
    ).fillna({"wins": 0, "losses": 0, "ties": 0, "games_played": 0})


def _record_teams(current_records: pd.DataFrame) -> set[str]:
    """Return the teams named in the current records."""
    return set(pd.Series(current_records["team_abbr"], dtype="string").dropna().astype(str))


def _composite_power_rankings(
    snapshot: pd.DataFrame,
    current_records: pd.DataFrame,
    *,
    season: int,
    through_week: int,
) -> pd.DataFrame:
    """Rank teams on one week's adjusted composite and attach their records.

    A team with a record but no composite is kept, with null ratings, and ranked last.
    """
    ratings = rank_teams_on_composite(snapshot)
    team_universe = sorted(_record_teams(current_records) | set(ratings["team_abbr"]))
    rated = set(ratings.loc[ratings[COMPOSITE_COLUMN].notna(), "team_abbr"])
    unrated = [team for team in team_universe if team not in rated]
    if unrated:
        log.warning("No adjusted composite for %s; ranked last.", ", ".join(unrated))

    pr = _with_records(
        _team_frame(team_universe).merge(ratings, on="team_abbr", how="left"), current_records
    )
    pr["season"] = int(season)
    pr["through_week"] = int(through_week)
    snapshot_weeks = (
        pd.to_numeric(snapshot["week"], errors="coerce").dropna().unique()
        if "week" in snapshot.columns
        else []
    )
    pr["snapshot_week"] = (
        int(snapshot_weeks[0]) if len(snapshot_weeks) == 1 else int(through_week) + 1
    )

    pr = pr.sort_values(
        [COMPOSITE_COLUMN, "team_abbr"], ascending=[False, True], na_position="last"
    )
    pr["rank"] = np.arange(1, len(pr) + 1)
    return pr


def _bradley_terry_power_rankings(
    games_for_ratings: pd.DataFrame,
    current_records: pd.DataFrame,
    *,
    season: int,
    through_week: int,
) -> pd.DataFrame:
    """Fit Bradley-Terry ratings to the games and rank teams on them.

    When `games_for_ratings` carries a `fit_weight` column it is used as per-game
    sample weights, which is how the caller down-weights older seasons.
    """
    # Ratings fit uses a combined table with a `p_home` target.
    fit_weights = (
        games_for_ratings[FIT_WEIGHT_COLUMN]
        if FIT_WEIGHT_COLUMN in games_for_ratings.columns
        else None
    )
    ratings_raw, home_adv = fit_bradley_terry_ratings(games_for_ratings, sample_weights=fit_weights)
    power_rating_1_10 = scale_ratings_1_to_10(ratings_raw)
    power_rating_0_10 = ratings_to_power_0_to_10(ratings_raw)

    team_universe = sorted(
        _record_teams(current_records)
        | set(pd.Series(games_for_ratings["home_abbr"], dtype="string").dropna().astype(str))
        | set(pd.Series(games_for_ratings["away_abbr"], dtype="string").dropna().astype(str))
    )
    base = _team_frame(team_universe)

    pr = base.merge(
        ratings_raw.rename("rating_raw").reset_index().rename(columns={"index": "team_abbr"}),
        on="team_abbr",
        how="left",
    ).merge(
        power_rating_1_10.reset_index().rename(columns={"index": "team_abbr"}),
        on="team_abbr",
        how="left",
    )

    pr = pr.merge(
        power_rating_0_10.reset_index().rename(columns={"index": "team_abbr"}),
        on="team_abbr",
        how="left",
    )

    pr = _with_records(pr, current_records)

    pr["home_advantage_logit"] = home_adv
    pr["home_advantage_prob"] = float(_sigmoid(home_adv))
    pr["season"] = int(season)
    pr["through_week"] = int(through_week)

    pr = pr.sort_values(["power_rating_1_10", "team_abbr"], ascending=[False, True])
    pr["rank"] = np.arange(1, len(pr) + 1)
    return pr


def build_power_rankings_and_standings(
    *,
    season: int,
    through_week: int,
    current_records: pd.DataFrame,
    future_games_with_probs: pd.DataFrame,
    games_for_ratings: pd.DataFrame | None = None,
    strength_snapshot: pd.DataFrame | None = None,
) -> PowerRatingsResult:
    """Build power rankings and projected standings tables.

    Pass exactly one ratings source:

    - ``strength_snapshot``: one week's per-team strength snapshot. Teams are ranked on
      its adjusted composite (see `rank_teams_on_composite`), with the components
      published next to the rank. A ``week`` column is published as ``snapshot_week``.
    - ``games_for_ratings``: games with a ``p_home`` target for the Bradley-Terry fit. A
      `fit_weight` column is used as per-game sample weights, which is how the caller
      down-weights older seasons.

    Projected standings are the same under both: current record plus the model's win
    probabilities for the remaining games.

    Raises:
        ValueError: If neither or both ratings sources are given.

    """
    if strength_snapshot is not None and games_for_ratings is None:
        pr = _composite_power_rankings(
            strength_snapshot, current_records, season=season, through_week=through_week
        )
    elif games_for_ratings is not None and strength_snapshot is None:
        pr = _bradley_terry_power_rankings(
            games_for_ratings, current_records, season=season, through_week=through_week
        )
    else:
        raise ValueError(
            "Pass exactly one ratings source: games_for_ratings (Bradley-Terry) or "
            "strength_snapshot (adjusted composite)."
        )

    projected = compute_projected_standings(
        current_records=current_records,
        future_games=future_games_with_probs,
        season=season,
    )
    projected["through_week"] = int(through_week)

    proj_div = projected.copy()
    proj_div["division"] = proj_div["division"].fillna("Unknown")
    proj_div["conference"] = proj_div["conference"].fillna("Unknown")
    proj_div = proj_div.sort_values(
        ["division", "projected_win_pct", "projected_wins", "team_abbr"],
        ascending=[True, False, False, True],
    ).copy()
    proj_div["projected_division_rank"] = (
        proj_div.groupby("division")["projected_win_pct"].rank(method="dense", ascending=False)
    ).astype(int)

    return PowerRatingsResult(
        power_rankings=pr.reset_index(drop=True),
        projected_standings=projected.reset_index(drop=True),
        projected_division_standings=proj_div.reset_index(drop=True),
    )


# Seasons the ratings fit sees by default, counting the current one. Two keeps a
# full prior season of evidence for early-season weeks without letting a franchise's
# history dominate the current one.
DEFAULT_RATINGS_WINDOW_SEASONS = 2

# Weight on games from before the current season. Low enough that the current season
# dominates once a few games exist, high enough to stabilize week 1.
DEFAULT_PRIOR_SEASON_WEIGHT = 0.25

# Columns the ratings fit consumes.
_RATINGS_COLUMNS = (
    "season",
    "week",
    "away_abbr",
    "home_abbr",
    "p_home",
    FIT_WEIGHT_COLUMN,
)

# Ranking methods, default first.
RANKING_METHODS = ("composite", "bradley_terry")

# Where the ETL writes the per-team weekly strength snapshots the composite method reads.
DEFAULT_STRENGTH_SNAPSHOTS = Path(constants.DATA_PATH) / f"{constants.STRENGTH_SNAPSHOTS_NAME}.csv"

# Scores are read as numbers explicitly. The ETL writes the newest games first, so the rows
# Polars samples to infer types can all be unplayed, and scores inferred as text compare
# alphabetically ("9" > "31").
_SCORE_DTYPES: dict[str, pl.DataType | type[pl.DataType]] = {
    "away_score": pl.Float64,
    "home_score": pl.Float64,
}

# Key columns of the strength snapshot file; every other column is a float.
_SNAPSHOT_KEY_DTYPES: dict[str, pl.DataType | type[pl.DataType]] = {
    "season": pl.Int64,
    "week": pl.Int64,
    "team_abbr": pl.String,
}


class StrengthSnapshotUnavailableError(ValueError):
    """The strength snapshot a composite ranking needs cannot be read."""


@dataclass(frozen=True)
class RankingOptions:
    """How a ranking is computed. Build it with `resolve_ranking_options`.

    The Bradley-Terry fields are ignored by the composite method, and the snapshot path
    is ignored by the Bradley-Terry method.
    """

    method: str = RANKING_METHODS[0]
    window_seasons: int = DEFAULT_RATINGS_WINDOW_SEASONS
    prior_season_weight: float = DEFAULT_PRIOR_SEASON_WEIGHT
    target: str = "margin"
    include_future: bool = False
    ratings_min_season: int | None = None
    strength_snapshots: Path = DEFAULT_STRENGTH_SNAPSHOTS


def _pl_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame to pandas without requiring pyarrow.

    Polars' `to_pandas()` requires `pyarrow` in many environments; for this reporting code,
    using `to_dicts()` keeps the dependency surface smaller.
    """
    if df.is_empty():
        # Preserve the schema so downstream code can rely on column presence even when
        # there are zero rows (e.g., postseason weeks with no future REG games).
        return pd.DataFrame(columns=list(df.columns))
    return pd.DataFrame(df.to_dicts())


POSTSEASON_GAME_TYPES = frozenset({"WC", "DIV", "CON", "SB", "POST"})


def _filter_by_game_type(df: pl.DataFrame, *, include_postseason: bool) -> pl.DataFrame:
    """Filter to regular season (or regular + postseason) games when game_type exists."""
    if "game_type" not in df.columns:
        return df
    game_type = pl.col("game_type").cast(pl.Utf8).str.to_uppercase()
    if include_postseason:
        allowed = ["REG", *sorted(POSTSEASON_GAME_TYPES)]
        return df.filter(game_type.is_in(allowed))
    return df.filter(game_type == "REG")


def _load_current_records(
    schedule_path: Path,
    *,
    season: int,
    through_week: int,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """Load current records through the specified week."""
    df = (
        pl.read_csv(schedule_path, schema_overrides=_SCORE_DTYPES)
        .select(
            [
                "season",
                "week",
                "game_type",
                "away_abbr",
                "home_abbr",
                "away_score",
                "home_score",
            ]
        )
        .filter(pl.col("season") == season)
    )

    # Only count games up through the specified week that have scores.
    df = df.filter(pl.col("week") <= through_week)
    df = _filter_by_game_type(df, include_postseason=include_postseason)
    df = df.filter(pl.col("away_score").is_not_null() & pl.col("home_score").is_not_null())

    # Compute per-team record.
    away = df.select(
        [
            pl.col("away_abbr").alias("team_abbr"),
            (pl.col("away_score") > pl.col("home_score")).cast(pl.Int32).alias("wins"),
            (pl.col("away_score") < pl.col("home_score")).cast(pl.Int32).alias("losses"),
            (pl.col("away_score") == pl.col("home_score")).cast(pl.Int32).alias("ties"),
        ]
    )
    home = df.select(
        [
            pl.col("home_abbr").alias("team_abbr"),
            (pl.col("home_score") > pl.col("away_score")).cast(pl.Int32).alias("wins"),
            (pl.col("home_score") < pl.col("away_score")).cast(pl.Int32).alias("losses"),
            (pl.col("home_score") == pl.col("away_score")).cast(pl.Int32).alias("ties"),
        ]
    )

    rec = (
        pl.concat([away, home], how="vertical")
        .group_by("team_abbr")
        .agg(
            [
                pl.col("wins").sum().alias("wins"),
                pl.col("losses").sum().alias("losses"),
                pl.col("ties").sum().alias("ties"),
            ]
        )
        .with_columns((pl.col("wins") + pl.col("losses") + pl.col("ties")).alias("games_played"))
    )

    return _pl_to_pandas(rec)


def _missing_market_inputs(
    required_features: list[str],
    available_cols: set[str],
) -> list[str]:
    """Return market-derived feature names that lack required raw inputs."""
    required = set(required_features)
    missing: list[str] = []

    if (
        "market_home_margin" in required
        and "market_home_margin" not in available_cols
        and not {"home_spread", "away_spread"} & available_cols
    ):
        missing.append("market_home_margin")
    if (
        "market_total_line" in required
        and "market_total_line" not in available_cols
        and "total_line" not in available_cols
    ):
        missing.append("market_total_line")
    if (
        "home_market_prob" in required
        and "home_market_prob" not in available_cols
        and "home_moneyline" not in available_cols
    ):
        missing.append("home_market_prob")
    if (
        "away_market_prob" in required
        and "away_market_prob" not in available_cols
        and "away_moneyline" not in available_cols
    ):
        missing.append("away_market_prob")

    return missing


def _format_missing_columns(missing: list[str], *, limit: int = 10) -> str:
    """Format a missing-column list for error messages."""
    unique = sorted(set(missing))
    if len(unique) <= limit:
        return ", ".join(unique)
    shown = ", ".join(unique[:limit])
    return f"{shown} (+{len(unique) - limit} more)"


def _predict_future_games(
    model: Any,
    *,
    model_kind: str,
    data_ml: Path,
    season: int,
    through_week: int,
    include_postseason: bool = False,
) -> pd.DataFrame:
    """Predict future games for the specified season."""
    # Load a season slice from the ML dataset (REG only by default), then predict for future games.
    # We read only the columns required by the model's FeatureSpec.
    spec = getattr(model, "feature_spec", None)
    if spec is None:
        raise ValueError("Model is missing feature_spec; cannot predict")

    available_cols = set(pl.read_csv(data_ml, n_rows=0).columns)
    derived_cols = set(ml_model_core.MARKET_DERIVED_COLUMNS)
    required_features = list(getattr(spec, "feature_columns", []))
    missing_required = set(required_features) - available_cols - derived_cols
    missing_required.update(_missing_market_inputs(required_features, available_cols))
    if missing_required:
        missing_text = _format_missing_columns(sorted(missing_required))
        raise ValueError(
            f"Missing required feature columns ({len(missing_required)}): {missing_text}"
        )
    base_cols = ["season", "week", "game_type", "away_abbr", "home_abbr"]
    market_raw_cols = [
        "home_spread",
        "away_spread",
        "total_line",
        "home_moneyline",
        "away_moneyline",
    ]
    base_cols = base_cols + [c for c in market_raw_cols if c in available_cols]
    feature_cols = [c for c in list(getattr(spec, "feature_columns", [])) if c in available_cols]
    usecols = sorted(set(base_cols + feature_cols))

    games = pl.read_csv(data_ml, columns=usecols).filter(pl.col("season") == season)
    games = _filter_by_game_type(games, include_postseason=include_postseason)
    games = games.filter(pl.col("week") > through_week)

    games = _pl_to_pandas(games)

    if games.empty:
        return games.assign(home_win_prob=pd.Series(dtype=float))

    if model_kind == "margin_total":
        mt = cast(ml_model_core.MarginTotalModel, model)
        pred_margin, _pred_total = ml_model_core.predict_margin_total_from_model(mt, games)
        use_uncertainty = bool(getattr(mt, "win_prob_use_uncertainty", False))
        sigma_margin = None
        if use_uncertainty:
            margin_quantiles, _ = ml_model_core._predict_margin_total_quantiles_from_model(
                mt, games
            )
            sigma_margin = ml_model_core._resolve_margin_sigma(
                pred_margin,
                margin_quantiles,
                fallback=constants.SCORE_DIFF_STD_DEV,
            )
        home_win_prob = ml_model_core.predict_home_win_prob(
            pred_margin,
            mt.calibrator,
            sigma=sigma_margin,
            use_uncertainty=use_uncertainty,
        )
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(mt, "market_prob_config", None)
        )
    elif model_kind == "blended_margin_total":
        bm = cast(ml_model_core.BlendedMarginTotalModel, model)
        # Use the model's blended margin as the win-prob driver.
        # The public helper takes pred_margin; for blended we reuse internal predict path.
        # We call build_prediction_output via the predict module would require a Path.
        team_margin, _team_total = ml_model_core.predict_margin_total_from_model(
            bm.team_model, games
        )
        if bm.market_model is None:
            market_margin, _market_total = ml_model_core.get_market_baseline(games)
        else:
            market_margin, _market_total = ml_model_core.predict_margin_total_from_model(
                bm.market_model, games
            )
        blended_margin = bm.blend_layer.margin_model.predict(
            np.column_stack([team_margin, market_margin])
        )
        home_win_prob = ml_model_core.predict_home_win_prob(blended_margin, bm.calibrator)
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(bm, "market_prob_config", None)
        )
    else:
        sm = cast(ml_model_core.ScoreModel, model)
        # ScoreModel: predict home/away scores then derive margin->prob.
        feature_df = ml_model_core.apply_feature_spec(games, sm.feature_spec)
        x = ml_model_core._transform_matrix(sm.preprocessor, feature_df)
        pred_away = ml_model_core.predict_xgb(sm.away_model, x)
        pred_home = ml_model_core.predict_xgb(sm.home_model, x)
        pred_margin = pred_home - pred_away
        home_win_prob = ml_model_core.predict_home_win_prob(
            pred_margin,
            getattr(sm, "calibrator", None),
        )
        home_win_prob = ml_model_core.adjust_home_win_prob(
            games, home_win_prob, getattr(sm, "market_prob_config", None)
        )

    out = games[["season", "week", "away_abbr", "home_abbr"]].copy()
    out["home_win_prob"] = home_win_prob
    return out


def _build_games_for_ratings(
    *,
    schedule_path: Path,
    season: int,
    through_week: int,
    ratings_min_season: int | None,
    future_games_with_probs: pd.DataFrame,
    include_postseason: bool,
    window_seasons: int = DEFAULT_RATINGS_WINDOW_SEASONS,
    prior_season_weight: float = DEFAULT_PRIOR_SEASON_WEIGHT,
    target: str = "margin",
    include_future: bool = False,
) -> pd.DataFrame:
    """Assemble the games the strength fit sees, with a per-game weight.

    The fit answers "how strong is each team going into next week", so by default it
    sees a short window of recent seasons rather than the whole archive, weights games
    from before the current season down by `prior_season_weight`, and scores completed
    games by margin rather than by a flat win/loss. Future games are excluded by
    default: feeding the model's own forecasts back into the strength fit makes the
    ranking partly a picture of the model rather than of results. They still drive
    projected standings.

    Args:
        schedule_path: Schedule/results dataset.
        season: Season being ranked.
        through_week: Records and results are counted through this week inclusive.
        ratings_min_season: Optional hard floor on seasons included.
        future_games_with_probs: Future games carrying model win probabilities.
        include_postseason: Whether postseason games count.
        window_seasons: Seasons the fit sees, counting the current one; 0 means all.
        prior_season_weight: Weight for games before the current season.
        target: "margin" or "binary" target for completed games.
        include_future: Whether future model probabilities enter the fit.

    Returns:
        Games with `season`, `week`, `away_abbr`, `home_abbr`, `p_home` and `fit_weight`.

    """
    sched = pl.read_csv(schedule_path, schema_overrides=_SCORE_DTYPES).select(
        [
            "season",
            "week",
            "game_type",
            "away_abbr",
            "home_abbr",
            "away_score",
            "home_score",
        ]
    )
    sched = _filter_by_game_type(sched, include_postseason=include_postseason)

    floor_season = ratings_min_season
    if window_seasons and window_seasons > 0:
        window_floor = season - window_seasons + 1
        floor_season = (
            window_floor if floor_season is None else max(int(floor_season), window_floor)
        )
    if floor_season is not None:
        sched = sched.filter(pl.col("season") >= int(floor_season))

    sched = _pl_to_pandas(sched)

    past = sched[
        (
            (sched["season"] < season)
            | ((sched["season"] == season) & (sched["week"] <= through_week))
        )
        & sched["away_score"].notna()
        & sched["home_score"].notna()
    ].copy()
    past["p_home"] = outcome_to_home_prob(past["home_score"], past["away_score"], target=target)
    # Current-season results carry full weight; earlier seasons are evidence, not equals.
    past[FIT_WEIGHT_COLUMN] = np.where(
        past["season"].to_numpy() == season, 1.0, float(prior_season_weight)
    )

    fut = future_games_with_probs.copy()
    if include_future and not fut.empty:
        fut = fut.rename(columns={"home_win_prob": "p_home"})
        fut = fut[["season", "week", "away_abbr", "home_abbr", "p_home"]].copy()
        fut[FIT_WEIGHT_COLUMN] = 1.0
    else:
        fut = pd.DataFrame(columns=[*_RATINGS_COLUMNS])

    games = pd.concat(
        [past[list(_RATINGS_COLUMNS)], fut[list(_RATINGS_COLUMNS)]], ignore_index=True
    )
    log.info(
        "Ratings fit diagnostics: past_games=%d future_games=%d min_season=%s "
        "prior_season_weight=%s target=%s",
        len(past),
        len(fut),
        floor_season if floor_season is not None else "all",
        prior_season_weight,
        target,
    )
    return games.dropna(subset=["p_home", "away_abbr", "home_abbr"]).copy()


def load_strength_snapshot(path: Path, *, season: int, through_week: int) -> pd.DataFrame:
    """Read the ETL's per-team strength snapshot for the week after ``through_week``.

    A ranking through week ``N`` describes teams going into week ``N + 1``. The ETL solves
    the week ``N + 1`` snapshot from games strictly before that week, so it holds every
    result through week ``N`` and nothing later, and it has a row for every team on the
    season's schedule, teams on a bye included. Rows for any other week are never used.

    Raises:
        StrengthSnapshotUnavailableError: If the file is missing, lacks the key columns or
            the composite, has no rows for that week, or lists a team twice in it.

    """
    target_week = int(through_week) + 1
    hint = (
        "Rebuild the data with `python -m nfl_predictor.data_collection`, or rank with "
        "--method bradley_terry."
    )
    if not path.exists():
        raise StrengthSnapshotUnavailableError(f"Missing strength snapshot file {path}. {hint}")

    header = pl.read_csv(path, n_rows=0).columns
    missing = sorted({*_SNAPSHOT_KEY_DTYPES, COMPOSITE_COLUMN} - set(header))
    if missing:
        raise StrengthSnapshotUnavailableError(f"{path} lacks the columns {missing}. {hint}")

    overrides = {column: _SNAPSHOT_KEY_DTYPES.get(column, pl.Float64) for column in header}
    frame = pl.read_csv(path, schema_overrides=overrides).filter(
        (pl.col("season") == int(season)) & (pl.col("week") == target_week)
    )
    if frame.is_empty():
        raise StrengthSnapshotUnavailableError(
            f"No strength snapshot for season {season} week {target_week} (a ranking "
            f"through week {through_week}) in {path}. {hint}"
        )
    duplicated = sorted(set(frame.filter(pl.col("team_abbr").is_duplicated())["team_abbr"]))
    if duplicated:
        raise StrengthSnapshotUnavailableError(
            f"{path} has duplicate rows for {duplicated} in season {season} week "
            f"{target_week}. {hint}"
        )
    return _pl_to_pandas(frame.sort("team_abbr"))


def resolve_ranking_options(
    *,
    method: str | None,
    legacy_franchise_fit: bool,
    window_seasons: int = DEFAULT_RATINGS_WINDOW_SEASONS,
    prior_season_weight: float = DEFAULT_PRIOR_SEASON_WEIGHT,
    target: str = "margin",
    include_future: bool = False,
    ratings_min_season: int | None = None,
    strength_snapshots: Path = DEFAULT_STRENGTH_SNAPSHOTS,
) -> RankingOptions:
    """Turn the ranking settings from a command line into one consistent set of options.

    ``method=None`` means the default, the composite, unless ``legacy_franchise_fit`` is
    set: the legacy fit is a Bradley-Terry fit, so it implies that method and replaces
    the other Bradley-Terry settings with the historical ones (every season weighted
    equally, binary targets, future model probabilities in the fit).

    Raises:
        ValueError: For an unknown method, or for the legacy fit combined with the
            composite.

    """
    if method is not None and method not in RANKING_METHODS:
        raise ValueError(
            f"Unknown ranking method {method!r}; expected one of {', '.join(RANKING_METHODS)}."
        )
    if legacy_franchise_fit:
        if method == "composite":
            raise ValueError(
                "--legacy-franchise-fit is a Bradley-Terry fit and cannot be combined with "
                "--method composite."
            )
        log.info("Legacy franchise fit requested; the other --ratings-* options are ignored.")
        return RankingOptions(
            method="bradley_terry",
            window_seasons=0,
            prior_season_weight=1.0,
            target="binary",
            include_future=True,
            ratings_min_season=ratings_min_season,
            strength_snapshots=Path(strength_snapshots),
        )
    return RankingOptions(
        method=method or RANKING_METHODS[0],
        window_seasons=int(window_seasons),
        prior_season_weight=float(prior_season_weight),
        target=str(target),
        include_future=bool(include_future),
        ratings_min_season=ratings_min_season,
        strength_snapshots=Path(strength_snapshots),
    )


def compute_power_rankings(
    model: Any,
    *,
    model_kind: str,
    data_ml: Path,
    data_schedule: Path,
    season: int,
    through_week: int,
    include_postseason: bool,
    options: RankingOptions,
) -> PowerRatingsResult:
    """Build the ranking and projected standings for one season through one week.

    Shared by the rankings command and the weekly run so both write the same artifact.
    Records count results through ``through_week``; projected standings add the model's
    win probabilities for the later games; the ranking follows ``options.method``.

    Raises:
        StrengthSnapshotUnavailableError: For the composite method, when the snapshot for
            the week after ``through_week`` cannot be read. This is checked before the
            model predicts anything.

    """
    snapshot = None
    if options.method == "composite":
        snapshot = load_strength_snapshot(
            options.strength_snapshots, season=season, through_week=through_week
        )

    current_records = _load_current_records(
        data_schedule,
        season=season,
        through_week=through_week,
        include_postseason=include_postseason,
    )
    future_games = _predict_future_games(
        model,
        model_kind=model_kind,
        data_ml=data_ml,
        season=season,
        through_week=through_week,
        include_postseason=include_postseason,
    )

    if snapshot is not None:
        log.info(
            "Ranking on the adjusted composite: season %d, snapshot week %d, from %s",
            season,
            through_week + 1,
            options.strength_snapshots,
        )
        return build_power_rankings_and_standings(
            season=season,
            through_week=through_week,
            current_records=current_records,
            future_games_with_probs=future_games,
            strength_snapshot=snapshot,
        )

    games_for_ratings = _build_games_for_ratings(
        schedule_path=data_schedule,
        season=season,
        through_week=through_week,
        ratings_min_season=options.ratings_min_season,
        future_games_with_probs=future_games,
        include_postseason=include_postseason,
        window_seasons=options.window_seasons,
        prior_season_weight=options.prior_season_weight,
        target=options.target,
        include_future=options.include_future,
    )
    return build_power_rankings_and_standings(
        season=season,
        through_week=through_week,
        current_records=current_records,
        games_for_ratings=games_for_ratings,
        future_games_with_probs=future_games,
    )


def write_ranking_outputs(
    result: PowerRatingsResult,
    *,
    out_dir: Path,
    season: int,
    through_week: int,
) -> None:
    """Write the ranking and both standings tables as CSVs under ``out_dir``.

    The files are ``power_rankings_``, ``projected_standings_`` and
    ``projected_division_standings_`` followed by ``season_<season>_week_<WW>.csv``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"season_{season}_week_{through_week:02d}"

    pr_path = out_dir / f"power_rankings_{suffix}.csv"
    st_path = out_dir / f"projected_standings_{suffix}.csv"
    div_path = out_dir / f"projected_division_standings_{suffix}.csv"

    result.power_rankings.to_csv(pr_path, index=False)
    result.projected_standings.to_csv(st_path, index=False)
    result.projected_division_standings.to_csv(div_path, index=False)

    log.info("Wrote %s", pr_path)
    log.info("Wrote %s", st_path)
    log.info("Wrote %s", div_path)
