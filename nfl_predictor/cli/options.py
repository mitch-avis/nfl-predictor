"""Option handling shared by the command-line entry points."""

from __future__ import annotations

import argparse

# The model kinds a run records in its metadata; ``blended_margin_total`` is an older name.
MODEL_KINDS = ("margin_total", "blend")
_MODEL_KIND_ALIASES = {"blended_margin_total": "blend"}


def model_kind(value: str) -> str:
    """Return the canonical model kind for ``value``, mapping the old blend name to ``blend``."""
    return _MODEL_KIND_ALIASES.get(value, value)


def add_model_kind_option(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add ``--model-kind`` with the shared vocabulary (default ``margin_total``)."""
    parser.add_argument(
        "--model-kind",
        type=model_kind,
        choices=MODEL_KINDS,
        default="margin_total",
        help=f"{help_text} (margin_total or blend; blended_margin_total also means blend).",
    )


def market_modes(mode: str) -> list[tuple[str, bool, bool]]:
    """Resolve which market modes to evaluate.

    Returns ``(label, include_market, market_anchor)`` for ``features``, ``anchor`` or
    ``hybrid``; any other value (``all``) returns all three.
    """
    if mode == "features":
        return [("features", True, False)]
    if mode == "anchor":
        return [("anchor", False, True)]
    if mode == "hybrid":
        return [("hybrid", True, True)]
    return [("features", True, False), ("anchor", False, True), ("hybrid", True, True)]


def parse_feature_groups(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature group list into a tuple of stripped, non-empty names."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


def add_market_prob_options(parser: argparse.ArgumentParser) -> None:
    """Add the market-probability post-processing options (blend weight, clamp, source, method).

    ``--market-prob-blend`` is kept as a second spelling of ``--market-prob-weight``.
    """
    parser.add_argument(
        "--market-prob-weight",
        "--market-prob-blend",
        dest="market_prob_weight",
        type=float,
        default=0.0,
        help="Blend weight for the market's implied win probability (0=off, 1=market only).",
    )
    parser.add_argument(
        "--market-prob-clamp",
        type=float,
        default=0.0,
        help="Clamp model probability within +/- this delta of market (0=off).",
    )
    parser.add_argument(
        "--market-prob-source",
        choices=["raw", "novig"],
        default="raw",
        help="Market probability source for blending/clamping.",
    )
    parser.add_argument(
        "--market-prob-blend-method",
        choices=["prob", "logit"],
        default="prob",
        help="Blend method for market probabilities (prob or logit space).",
    )


def add_wf_window_options(parser: argparse.ArgumentParser) -> None:
    """Add the walk-forward evaluation window options; the bare spellings stay as aliases."""
    # Imported here so that commands without a walk-forward window never load the model code.
    from nfl_predictor.ml import walk_forward

    parser.add_argument(
        "--wf-eval-last-n-seasons",
        "--eval-last-n-seasons",
        dest="eval_last_n_seasons",
        type=int,
        default=3,
        help=(
            "Evaluate the last N seasons in the dataset (regular season only). The count "
            "includes a current season that has no completed week yet."
        ),
    )
    parser.add_argument(
        "--wf-start-week",
        type=int,
        default=3,
        help="Walk-forward start week.",
    )
    parser.add_argument(
        "--wf-calibration-weeks",
        "--calibration-weeks",
        dest="wf_calibration_weeks",
        type=int,
        default=walk_forward.DEFAULT_CALIBRATION_WEEKS,
        help="Number of prior weeks (same season) used for time-aware calibration.",
    )
    parser.add_argument(
        "--wf-exclude-incomplete-seasons",
        "--exclude-incomplete-seasons",
        dest="exclude_incomplete_seasons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude seasons whose regular season is incomplete in the dataset (useful when "
            "the current season is partial)."
        ),
    )


def add_feature_group_option(parser: argparse.ArgumentParser) -> None:
    """Add ``--disable-feature-groups`` for ablation runs."""
    parser.add_argument(
        "--disable-feature-groups",
        type=str,
        default=None,
        help=(
            "Comma-separated feature group names to drop for ablation comparisons "
            "(e.g. 'pbp' or 'pbp,other'). See constants.FEATURE_GROUP_COLUMN_MARKERS."
        ),
    )
