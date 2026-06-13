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

This is a display/reporting artifact only; it does not affect training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor import constants


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
) -> np.ndarray:
    """Convert an observed game outcome into a probability target.

    Ties are represented as 0.5.
    """
    home = pd.to_numeric(home_score, errors="coerce")
    away = pd.to_numeric(away_score, errors="coerce")
    margin = home - away

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
) -> tuple[pd.Series, float]:
    """Fit latent team ratings from per-game home win probabilities.

    Args:
        games: Must include home/away team columns and a probability column.
        home_team_col: Home team abbreviation column.
        away_team_col: Away team abbreviation column.
        p_home_col: Home win probability column.
        include_home_advantage: If True, fit a global home-field advantage term.
        ridge_alpha: L2 regularization strength.

    Returns:
        (team_rating_series, home_advantage)

    """
    required = {home_team_col, away_team_col, p_home_col}
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"games missing required columns: {missing}")

    df = games[[home_team_col, away_team_col, p_home_col]].dropna().copy()
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
    scaled = 1.0 + 9.0 * p_vs_avg
    return pd.Series(scaled, index=x.index, name="power_rating_1_10").round(2)


def ratings_to_power_0_to_10(ratings_raw: pd.Series) -> pd.Series:
    """Convert raw ratings to a stable 0-10 scale.

    `10 * sigmoid(rating_raw)` is interpretable as a 0–10 strength score derived
    from win probability vs an average team on a neutral field.
    """
    x = ratings_raw.astype(float)
    scaled = 10.0 * _sigmoid(x.to_numpy())
    return pd.Series(scaled, index=x.index, name="power_rating_0_10").round(2)


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


def build_power_rankings_and_standings(
    *,
    season: int,
    through_week: int,
    current_records: pd.DataFrame,
    games_for_ratings: pd.DataFrame,
    future_games_with_probs: pd.DataFrame,
) -> PowerRatingsResult:
    """Build power rankings and projected standings tables."""
    # Ratings fit uses a combined table with a `p_home` target.
    ratings_raw, home_adv = fit_bradley_terry_ratings(games_for_ratings)
    power_rating_1_10 = scale_ratings_1_to_10(ratings_raw)
    power_rating_0_10 = ratings_to_power_0_to_10(ratings_raw)

    team_universe = sorted(
        set(pd.Series(current_records["team_abbr"], dtype="string").dropna().astype(str))
        | set(pd.Series(games_for_ratings["home_abbr"], dtype="string").dropna().astype(str))
        | set(pd.Series(games_for_ratings["away_abbr"], dtype="string").dropna().astype(str))
    )
    base = pd.DataFrame({"team_abbr": team_universe})
    base["division"] = base["team_abbr"].map(constants.TEAM_TO_DIVISION).fillna("Unknown")
    base["conference"] = base["team_abbr"].map(constants.TEAM_TO_CONFERENCE).fillna("Unknown")

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

    pr = pr.merge(
        current_records[["team_abbr", "wins", "losses", "ties", "games_played"]],
        on="team_abbr",
        how="left",
    ).fillna({"wins": 0, "losses": 0, "ties": 0, "games_played": 0})

    pr["home_advantage_logit"] = home_adv
    pr["home_advantage_prob"] = float(_sigmoid(home_adv))
    pr["season"] = int(season)
    pr["through_week"] = int(through_week)

    pr = pr.sort_values(["power_rating_1_10", "team_abbr"], ascending=[False, True])
    pr["rank"] = np.arange(1, len(pr) + 1)

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
