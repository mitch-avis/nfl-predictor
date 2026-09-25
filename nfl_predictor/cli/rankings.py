"""Generate power rankings and projected standings from a trained model.

This script is meant for fan-friendly reporting:
- A 1-10 (and 0-10) power rating per team (absolute scale vs an average team)
- Projected standings based on current record + expected wins in remaining REG games

Rating methods
--------------
``--method composite`` (the default) is the current-season view: how strong each team is
going into the next week. It ranks teams on the ETL's pre-week schedule-adjusted
strength composite, read from ``data/strength_snapshots.csv`` for the week after
``--through-week``. Every team on the schedule has a row there, teams on a bye included,
and each row is solved only from games before that week, so a ranking through week N
sees results through week N and nothing later.

``--method bradley_terry`` fits latent strengths to game results instead: completed
games as probability targets (margin-based by default) over a window of recent seasons
with earlier seasons down-weighted, optionally with the model's win probabilities for
future games (the ``--ratings-*`` options). ``--legacy-franchise-fit`` restores the
original all-seasons, equal-weight "franchise" fit and implies this method.

Both methods map their rating onto the 1-10 and 0-10 scales through a win probability
against an average team on a neutral field (see
``nfl_predictor.reporting.power_rankings``). Projected standings are the same under both:
current record plus the model's win probabilities for the remaining games.

Usage example
-------------
nfl-predictor rankings \
  --model-in models/<run_id>/model.joblib \
  --model-kind margin_total \
  --season 2025 \
  --through-week 10 \
  --data-ml data/all_data_ml.csv \
  --data-schedule data/all_data.csv \
  --out-dir data/predict

Outputs
-------
- power_rankings_season_XXXX_week_YY.csv
- projected_standings_season_XXXX_week_YY.csv
- projected_division_standings_season_XXXX_week_YY.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nfl_predictor.cli import options
from nfl_predictor.ml import ml_model_core
from nfl_predictor.reporting.power_rankings import (
    DEFAULT_PRIOR_SEASON_WEIGHT,
    DEFAULT_RATINGS_WINDOW_SEASONS,
    DEFAULT_STRENGTH_SNAPSHOTS,
    RANKING_METHODS,
    compute_power_rankings,
    resolve_ranking_options,
    write_ranking_outputs,
)
from nfl_predictor.utils.logger import log


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate power rankings + projected standings")
    p.add_argument("--model-in", type=Path, required=True, help="Path to model checkpoint .joblib")
    options.add_model_kind_option(p, "Model kind, matching the saved checkpoint")
    p.add_argument("--season", type=int, required=True, help="Season year")
    p.add_argument(
        "--through-week",
        type=int,
        required=True,
        help="Compute records through this week (inclusive); future games are week > this.",
    )
    p.add_argument(
        "--data-ml",
        type=Path,
        default=Path("data/all_data_ml.csv"),
        help="ML dataset (must include future game feature rows).",
    )
    p.add_argument(
        "--data-schedule",
        type=Path,
        default=Path("data/all_data.csv"),
        help="Schedule/results dataset (used for current records).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/predict"),
        help="Directory to write outputs.",
    )
    p.add_argument(
        "--method",
        choices=RANKING_METHODS,
        default=None,
        help=(
            "How teams are rated. 'composite' (default) ranks on the ETL's "
            "schedule-adjusted strength composite for the week after --through-week: the "
            "current-season view. 'bradley_terry' fits ratings to game results with the "
            "--ratings-* options. --legacy-franchise-fit implies bradley_terry."
        ),
    )
    p.add_argument(
        "--strength-snapshots",
        type=Path,
        default=DEFAULT_STRENGTH_SNAPSHOTS,
        help=(
            "Per-team weekly strength file written by the ETL, read by the composite "
            "method (default: data/strength_snapshots.csv)."
        ),
    )
    p.add_argument(
        "--ratings-min-season",
        type=int,
        default=None,
        help=(
            "Bradley-Terry only. Optional minimum season to include in the ratings fit. "
            "Default: include all historical seasons available in --data-schedule."
        ),
    )
    p.add_argument(
        "--include-postseason",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include postseason games in records and in the Bradley-Terry fit (default: "
            "regular season only). The composite never includes postseason games."
        ),
    )
    p.add_argument(
        "--ratings-window-seasons",
        type=int,
        default=DEFAULT_RATINGS_WINDOW_SEASONS,
        help=(
            "Bradley-Terry only. How many seasons the ratings fit sees, counting the "
            f"current one. Default {DEFAULT_RATINGS_WINDOW_SEASONS} (current plus "
            "previous). Use 0 for every available season."
        ),
    )
    p.add_argument(
        "--ratings-prior-season-weight",
        type=float,
        default=DEFAULT_PRIOR_SEASON_WEIGHT,
        help=(
            "Bradley-Terry only. Weight applied to games from seasons before the current "
            f"one. Default {DEFAULT_PRIOR_SEASON_WEIGHT}. Current-season games always weigh "
            "1.0."
        ),
    )
    p.add_argument(
        "--ratings-target",
        choices=("margin", "binary"),
        default="margin",
        help=(
            "Bradley-Terry only. Target for completed games. 'margin' maps the observed "
            "point margin through the model's win-probability curve; 'binary' scores "
            "every win the same."
        ),
    )
    p.add_argument(
        "--ratings-include-future",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Bradley-Terry only. Feed future games' model win probabilities into the "
            "strength fit. Off by default: the fit should describe results, not the "
            "model's own forecasts. Future games always remain in projected standings."
        ),
    )
    p.add_argument(
        "--legacy-franchise-fit",
        action="store_true",
        help=(
            "Reproduce the historical 'franchise' ranking exactly: a Bradley-Terry fit "
            "with every season since 1999 weighted equally, binary win/loss targets, and "
            "future model probabilities in the fit. Implies --method bradley_terry and "
            "overrides the other --ratings-* flags."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the power rankings CLI."""
    args = _parse_args(argv)
    try:
        options = resolve_ranking_options(
            method=args.method,
            legacy_franchise_fit=bool(args.legacy_franchise_fit),
            window_seasons=int(args.ratings_window_seasons),
            prior_season_weight=float(args.ratings_prior_season_weight),
            target=str(args.ratings_target),
            include_future=bool(args.ratings_include_future),
            ratings_min_season=args.ratings_min_season,
            strength_snapshots=args.strength_snapshots,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    if not args.model_in.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {args.model_in}")
    if not args.data_ml.exists():
        raise FileNotFoundError(f"Missing ML dataset: {args.data_ml}")
    if not args.data_schedule.exists():
        raise FileNotFoundError(f"Missing schedule dataset: {args.data_schedule}")
    if options.method == "composite" and not options.strength_snapshots.exists():
        raise FileNotFoundError(
            f"Missing strength snapshot file: {options.strength_snapshots}. Rebuild the data "
            "or rank with --method bradley_terry."
        )

    model = ml_model_core.load_model_checkpoint(args.model_in, args.model_kind)
    result = compute_power_rankings(
        model,
        model_kind=args.model_kind,
        data_ml=args.data_ml,
        data_schedule=args.data_schedule,
        season=args.season,
        through_week=args.through_week,
        include_postseason=bool(args.include_postseason),
        options=options,
    )

    write_ranking_outputs(
        result, out_dir=args.out_dir, season=args.season, through_week=args.through_week
    )
    log.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
