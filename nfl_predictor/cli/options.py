"""Option handling shared by the command-line entry points."""

from __future__ import annotations

import argparse

# The model kinds a run records in its metadata.
MODEL_KINDS = ("margin_total",)
# Kinds older runs recorded that can no longer be trained or loaded.
RETIRED_MODEL_KINDS = frozenset({"blend", "blended_margin_total"})
RETIRED_BLEND_MESSAGE = (
    "the blend model kind was retired (production submits the deterministic floor of a "
    "margin_total model); train a margin_total model instead"
)


def model_kind(value: str) -> str:
    """Return ``value``, or refuse a retired kind with the reason.

    Raises:
        argparse.ArgumentTypeError: If ``value`` names the retired blend kind.

    """
    if value in RETIRED_MODEL_KINDS:
        raise argparse.ArgumentTypeError(RETIRED_BLEND_MESSAGE)
    return value


def add_model_kind_option(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add ``--model-kind`` with the shared vocabulary (default ``margin_total``)."""
    parser.add_argument(
        "--model-kind",
        type=model_kind,
        choices=MODEL_KINDS,
        default="margin_total",
        help=f"{help_text} (margin_total, the only kind).",
    )


def parse_feature_groups(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated feature group list into a tuple of stripped, non-empty names."""
    if not raw:
        return ()
    return tuple(name.strip() for name in raw.split(",") if name.strip())


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
        help=(
            "Walk-forward: a positive value switches on the pooled calibrator frame for fitted "
            "calibrators (the previous two seasons plus the eval season's completed weeks); "
            "folds hold nothing out of the tree fit."
        ),
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
