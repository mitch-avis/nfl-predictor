"""Feature specification helpers for ML preprocessing."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nfl_predictor import constants
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_FEATURE_START_COLUMN = "away_rest"
DEFAULT_FEATURE_END_COLUMN = "home_moneyline"

MARKET_DERIVED_COLUMNS = (
    "market_home_margin",
    "market_total_line",
    "home_market_prob",
    "away_market_prob",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Feature metadata for model training and inference."""

    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    dropped_columns: list[str]
    id_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    feature_start: str
    feature_end: str
    metadata_columns: list[str]
    post_feature_columns: list[str]
    market_columns: list[str]


def _get_feature_range_columns(
    df: pd.DataFrame, feature_start: str, feature_end: str
) -> tuple[list[str], list[str], list[str]]:
    """Return feature range columns plus leading/trailing metadata groups."""
    columns = df.columns.tolist()
    if feature_start not in columns or feature_end not in columns:
        msg = f"Expected feature range columns '{feature_start}'..'{feature_end}' in dataset."
        raise ValueError(msg)
    start_idx = columns.index(feature_start)
    end_idx = columns.index(feature_end)
    if start_idx > end_idx:
        msg = f"Feature start column '{feature_start}' occurs after '{feature_end}'."
        raise ValueError(msg)
    feature_range = columns[start_idx : end_idx + 1]
    metadata_columns = columns[:start_idx]
    post_feature_columns = columns[end_idx + 1 :]
    return feature_range, metadata_columns, post_feature_columns


def _implied_prob_from_moneyline(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert moneyline values into implied win probabilities."""
    if isinstance(values, pd.Series):
        moneyline_series = pd.to_numeric(values, errors="coerce")
    else:
        moneyline_series = pd.to_numeric(pd.Series(values), errors="coerce")
    moneyline = moneyline_series.to_numpy(dtype=float)
    probs = np.full_like(moneyline, np.nan, dtype=float)
    neg_mask = moneyline < 0
    pos_mask = moneyline > 0
    probs[neg_mask] = -moneyline[neg_mask] / (-moneyline[neg_mask] + 100)
    probs[pos_mask] = 100 / (moneyline[pos_mask] + 100)
    return probs


def add_market_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived market columns exist on the input DataFrame."""
    df = df.copy()
    if "market_home_margin" not in df.columns:
        if "home_spread" in df.columns:
            df["market_home_margin"] = -pd.to_numeric(df["home_spread"], errors="coerce")
        elif "away_spread" in df.columns:
            df["market_home_margin"] = pd.to_numeric(df["away_spread"], errors="coerce")
    if "market_total_line" not in df.columns and "total_line" in df.columns:
        df["market_total_line"] = pd.to_numeric(df["total_line"], errors="coerce")
    if "home_market_prob" not in df.columns and "home_moneyline" in df.columns:
        df["home_market_prob"] = _implied_prob_from_moneyline(df["home_moneyline"])
    if "away_market_prob" not in df.columns and "away_moneyline" in df.columns:
        df["away_market_prob"] = _implied_prob_from_moneyline(df["away_moneyline"])
    return df


def get_market_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return market baseline margin and total arrays from input DataFrame."""
    df = add_market_transforms(df)
    if "market_home_margin" not in df.columns or "market_total_line" not in df.columns:
        msg = "Market anchor requested but spread/total columns are missing."
        raise ValueError(msg)
    baseline_margin = pd.to_numeric(df["market_home_margin"], errors="coerce").to_numpy(dtype=float)
    baseline_total = pd.to_numeric(df["market_total_line"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(baseline_margin).any() or np.isnan(baseline_total).any():
        msg = "Market anchor requested but spread/total contains missing values."
        raise ValueError(msg)
    return baseline_margin, baseline_total


def _drop_identifier_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove identifier columns (suffix _id) from a DataFrame."""
    id_columns = [col for col in df.columns if col.endswith("_id")]
    if not id_columns:
        return df, []
    return df.drop(columns=id_columns), id_columns


def _is_constant(series: pd.Series) -> bool:
    """Return whether a column's non-missing values are all equal (or there are none)."""
    values = series.dropna()
    return values.empty or bool((values == values.iloc[0]).all())


def _drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with zero variance."""
    constant_cols = [col for col in df.columns if _is_constant(df[col])]
    if not constant_cols:
        return df, []
    return df.drop(columns=constant_cols), constant_cols


def _drop_high_cardinality_columns(
    df: pd.DataFrame, max_cardinality_ratio: float
) -> tuple[pd.DataFrame, list[str]]:
    """Drop categorical columns with high cardinality ratios."""
    cat_cols = df.select_dtypes(include=["object", "str", "category"]).columns
    dropped: list[str] = []
    row_count = max(len(df), 1)
    for col in cat_cols:
        unique_ratio = df[col].nunique(dropna=True) / row_count
        if unique_ratio >= max_cardinality_ratio:
            dropped.append(col)
    if not dropped:
        return df, []
    return df.drop(columns=dropped), dropped


def validate_feature_groups(groups: Sequence[str]) -> None:
    """Check that every name in ``groups`` is a feature group, before any data is read.

    Raises:
        ValueError: if any name is not a key in ``constants.FEATURE_GROUP_COLUMN_MARKERS``; the
            message lists the unknown names and the valid ones.

    """
    valid_groups = constants.FEATURE_GROUP_COLUMN_MARKERS
    unknown = sorted({group for group in groups if group not in valid_groups})
    if unknown:
        msg = f"Unknown feature group(s): {unknown}. Valid groups: {sorted(valid_groups)}."
        raise ValueError(msg)


def resolve_feature_group_columns(
    columns: Sequence[str], groups: Sequence[str], *, warn_unmatched: bool = True
) -> list[str]:
    """Resolve which columns belong to the given feature groups: what disabling them removes.

    Every fit that leaves feature groups out drops exactly these columns: each walk-forward
    fold (``WalkForwardConfig.disabled_feature_groups``) and the final fit
    (``TrainingOptions.disabled_feature_groups``).

    A group that matches none of ``columns`` leaves nothing out, so a dataset build that lacks
    it trains differently from one that has it; each such group is logged as a warning, unless
    ``warn_unmatched`` is off for a caller resolving columns that may already have been dropped.

    A column belongs to a group when any marker string configured for that group in
    ``constants.FEATURE_GROUP_COLUMN_MARKERS`` is a *substring* of the column name (not a
    prefix), so a single marker can match ``away_``/``home_``/``opponent_`` prefixed variants
    and ``_diff`` suffixed variants of the same base name at once. A group whose marker tuple
    is empty matches zero columns.

    Raises:
        ValueError: if any requested group name is not a key in
            ``constants.FEATURE_GROUP_COLUMN_MARKERS`` (``validate_feature_groups``).

    """
    validate_feature_groups(groups)
    matched: set[str] = set()
    unmatched: list[str] = []
    for group in groups:
        markers = constants.FEATURE_GROUP_COLUMN_MARKERS[group]
        group_columns = [
            column for column in columns if any(marker in column for marker in markers)
        ]
        if not group_columns:
            unmatched.append(group)
        matched.update(group_columns)
    if unmatched and warn_unmatched:
        log.warning(
            "Feature group(s) %s match no column of the data, so leaving them out drops "
            "nothing; a dataset build without them trains differently from one with them.",
            sorted(set(unmatched)),
        )
    return sorted(matched)


def check_ratings_replace_strength(
    games: pd.DataFrame, groups: Sequence[str], *, stop: bool
) -> None:
    """Stop or warn when ``groups`` leave out the strength snapshot and no ported rating is there.

    Production leaves out ``strength`` because the ported team ratings (``ratings``) replace
    it; a dataset built before the port has the strength snapshot but no ratings column, and a
    build without play-by-play keeps the ratings columns null in every row, so either way the
    fit would train on neither. A ratings column counts only if some row has a value. Data with
    neither group gets no message: leaving ``strength`` out removes nothing there, and no group
    list would help. ``stop`` raises; otherwise the same message is logged as a warning and the
    run carries on (a backtest, which runs ablations on purpose).

    Raises:
        ValueError: if ``stop`` is set, ``groups`` include ``strength``, and ``games`` hold a
            column of the ``strength`` group but no ``ratings`` column with a value.

    """
    if "strength" not in groups:
        return
    columns = list(games.columns)
    if not resolve_feature_group_columns(columns, ["strength"], warn_unmatched=False):
        return
    ratings = resolve_feature_group_columns(columns, ["ratings"], warn_unmatched=False)
    if ratings and bool(games[ratings].notna().any(axis=None)):
        return
    msg = (
        "The feature groups left out include the strength snapshot (strength), but the data has "
        "no column of the ported ratings (ratings) with a value, so the model would train on "
        "neither. "
        "Rebuild the dataset with the ported ratings (`nfl-predictor data`), or pass a group "
        "list that keeps strength: `--disable-feature-groups own_elo,own_elo_wt`, or "
        "`disable_feature_groups: [own_elo, own_elo_wt]` in the weekly config."
    )
    if stop:
        raise ValueError(msg)
    log.warning(msg)


# The sets of market-informed groups this process has already noted, each noted once:
# every walk-forward fold builds its own spec.
_WARNED_MARKET_INFORMED: set[tuple[str, ...]] = set()


@dataclass(frozen=True, kw_only=True)
class FeatureSelection:
    """How a feature spec chooses its columns from a dataset.

    ``feature_start``/``feature_end`` bound the candidate columns; ``market_transform``
    replaces the raw lines with their derived columns; ``include_market`` keeps the market
    columns at all; pruning drops the known-weak columns unless ``disable_pruning``; and
    categoricals above ``max_cardinality_ratio`` distinct values per row are dropped.
    """

    include_market: bool
    max_cardinality_ratio: float
    feature_start: str = DEFAULT_FEATURE_START_COLUMN
    feature_end: str = DEFAULT_FEATURE_END_COLUMN
    market_transform: bool = False
    disable_pruning: bool = False


@dataclass(frozen=True)
class _MarketColumns:
    """The market columns of a feature range, and which of them the spec drops."""

    feature_range: list[str]
    derived: list[str]
    features: list[str]
    excluded: list[str]
    dropped_raw: list[str]


def _resolve_market_columns(
    df: pd.DataFrame, feature_range: list[str], selection: FeatureSelection
) -> _MarketColumns:
    """Return the market columns: derived ones join the range under the transform.

    Without market features every market column in the range is excluded, the raw lines and
    any derived column a dataset already carries there alike, whatever the transform setting.
    """
    derived = [col for col in MARKET_DERIVED_COLUMNS if col in df.columns]
    if selection.market_transform:
        feature_range = feature_range + [col for col in derived if col not in feature_range]
    raw = [col for col in feature_range if col in constants.LINES_COLUMNS]
    features = derived if selection.market_transform else raw
    dropped_raw = raw if selection.market_transform and raw else []
    excluded = (
        []
        if selection.include_market
        else [col for col in feature_range if col in raw or col in MARKET_DERIVED_COLUMNS]
    )
    return _MarketColumns(feature_range, derived, features, excluded, dropped_raw)


def _note_kept_market_informed_groups(columns: Sequence[str], selection: FeatureSelection) -> None:
    """Note that a run without game lines keeps the Elo groups' preseason win-total priors.

    A run without the market columns (``include_market`` off) means "no game lines": it still
    trains on the groups of ``constants.MARKET_INFORMED_FEATURE_GROUPS``, whose ratings carry a
    preseason win-total prior, unless it leaves them out too. That is the run's choice, so an
    info message names the groups it keeps and how to leave them out, once per set of groups in
    a process. ``columns`` are the spec's final feature columns, so a group whose columns all
    dropped out (all null or constant in the training rows) goes unnamed.
    """
    if selection.include_market:
        return
    kept = tuple(
        group
        for group in constants.MARKET_INFORMED_FEATURE_GROUPS
        if resolve_feature_group_columns(columns, [group], warn_unmatched=False)
    )
    if not kept or kept in _WARNED_MARKET_INFORMED:
        return
    _WARNED_MARKET_INFORMED.add(kept)
    log.info(
        "No game lines: the model trains without betting-line columns but keeps the preseason "
        "win-total priors in the feature group(s) %s; for a run with no market information at "
        "all, add %s to the groups left out: `--disable-feature-groups` (backtest, train; it "
        "replaces the default list) or `disable_feature_groups` in the weekly config.",
        list(kept),
        ",".join(kept),
    )


def _log_feature_spec(
    spec: FeatureSpec,
    selection: FeatureSelection,
    result_columns: list[str],
    market: _MarketColumns,
    pruned_columns: list[str],
) -> None:
    """Log, at debug level, every group of columns the spec left out and why."""
    if selection.market_transform:
        log.debug("Market feature transforms enabled: %s", market.derived)
    log.debug(
        "Dropped metadata columns (%d): %s", len(spec.metadata_columns), spec.metadata_columns
    )
    log.debug(
        "Dropped post-feature columns (%d): %s",
        len(spec.post_feature_columns),
        spec.post_feature_columns,
    )
    if result_columns:
        log.debug(
            "Target/result columns present (%d): %s",
            len(result_columns),
            result_columns,
        )
    dropped_market_columns = sorted(set(market.excluded + market.dropped_raw))
    if dropped_market_columns:
        log.debug(
            "Dropped market columns (%d): %s",
            len(dropped_market_columns),
            dropped_market_columns,
        )
    if selection.disable_pruning:
        log.debug("Feature pruning disabled.")
    elif pruned_columns:
        log.debug("Dropped pruned columns (%d): %s", len(pruned_columns), pruned_columns)
    if spec.id_columns:
        log.debug("Dropped identifier columns (%d): %s", len(spec.id_columns), spec.id_columns)
    if spec.constant_columns:
        log.debug(
            "Dropped constant columns (%d): %s", len(spec.constant_columns), spec.constant_columns
        )
    if spec.high_cardinality_columns:
        log.debug(
            "Dropped high-cardinality categoricals (%d, threshold=%.2f): %s",
            len(spec.high_cardinality_columns),
            selection.max_cardinality_ratio,
            spec.high_cardinality_columns,
        )


def build_feature_spec(df: pd.DataFrame, selection: FeatureSelection) -> FeatureSpec:
    """Build a FeatureSpec describing selected and dropped columns."""
    if selection.market_transform:
        df = add_market_transforms(df)

    feature_range, metadata_columns, post_feature_columns = _get_feature_range_columns(
        df, selection.feature_start, selection.feature_end
    )

    result_columns = [col for col in constants.RESULT_COLUMNS if col in df.columns]
    drop_columns = set(result_columns)

    market = _resolve_market_columns(df, feature_range, selection)
    drop_columns.update(market.dropped_raw)
    drop_columns.update(market.excluded)
    selected_columns = [col for col in market.feature_range if col not in drop_columns]

    pruned_columns: list[str] = []
    if not selection.disable_pruning:
        pruned_columns = [
            col for col in constants.PRUNED_FEATURE_COLUMNS if col in selected_columns
        ]
        if pruned_columns:
            drop_columns.update(pruned_columns)
            selected_columns = [col for col in selected_columns if col not in pruned_columns]

    feature_df = df[selected_columns].copy()

    feature_df, id_columns = _drop_identifier_columns(feature_df)
    feature_df, constant_columns = _drop_constant_columns(feature_df)
    feature_df, high_cardinality_columns = _drop_high_cardinality_columns(
        feature_df, selection.max_cardinality_ratio
    )
    _note_kept_market_informed_groups(feature_df.columns.tolist(), selection)

    categorical_columns = feature_df.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    numeric_columns = [col for col in feature_df.columns if col not in categorical_columns]

    spec = FeatureSpec(
        feature_columns=feature_df.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        dropped_columns=sorted(drop_columns),
        id_columns=id_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        feature_start=selection.feature_start,
        feature_end=selection.feature_end,
        metadata_columns=metadata_columns,
        post_feature_columns=post_feature_columns,
        market_columns=market.features,
    )
    _log_feature_spec(spec, selection, result_columns, market, pruned_columns)
    return spec


def apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Apply a FeatureSpec to reorder/select model features."""
    df = df.copy()
    if any(col in spec.feature_columns for col in MARKET_DERIVED_COLUMNS):
        df = add_market_transforms(df)
    missing = [col for col in spec.feature_columns if col not in df.columns]
    if missing:
        log.debug(
            "Missing %d feature columns in input data; filling with NaN: %s",
            len(missing),
            missing,
        )
    return df.reindex(columns=spec.feature_columns)


def build_preprocessor(spec: FeatureSpec, *, for_tree: bool = True) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric/categorical features."""
    encoder_params: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_params["sparse_output"] = for_tree
    else:
        encoder_params["sparse"] = for_tree

    numeric_steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        )
    ]
    if not for_tree:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(**encoder_params)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if spec.numeric_columns:
        transformers.append(("num", numeric_transformer, spec.numeric_columns))
    if spec.categorical_columns:
        transformers.append(("cat", categorical_transformer, spec.categorical_columns))
    if not transformers:
        msg = "No feature columns available after preprocessing."
        raise ValueError(msg)

    if for_tree:
        return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)
    return ColumnTransformer(transformers=transformers, remainder="drop")
