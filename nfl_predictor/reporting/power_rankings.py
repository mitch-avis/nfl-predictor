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

This is a display/reporting artifact only; it does not affect training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor import constants
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
