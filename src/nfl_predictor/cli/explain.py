"""SHAP analysis for trained models, using XGBoost's exact TreeSHAP (``pred_contribs``)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from nfl_predictor.ml import artifacts, feature_importance, ml_model_core
from nfl_predictor.ml.ml_model_xgb_utils import transform_matrix
from nfl_predictor.utils.logger import log

if TYPE_CHECKING:
    import xgboost as xgb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP analysis of one model head's features (XGBoost TreeSHAP)."
    )
    parser.add_argument(
        "--model-in",
        "--model-path",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to a saved model.joblib artifact.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="CSV dataset for feature extraction (same schema used for training).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for the SHAP report (JSON).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Max rows to sample for SHAP (default: 500).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for row sampling.",
    )
    parser.add_argument(
        "--target",
        choices=["margin", "total"],
        default="margin",
        help="Target model head to analyze (margin or total).",
    )
    return parser.parse_args()


def _select_model_component(model: object) -> tuple[ml_model_core.MarginTotalModel, str]:
    """Resolve the margin/total model to analyze; a saved blend model is refused."""
    if isinstance(model, ml_model_core.BlendedMarginTotalModel):
        msg = (
            "This is a blend model: the blend model kind was retired; "
            "explain a margin_total model instead."
        )
        raise TypeError(msg)
    if isinstance(model, ml_model_core.MarginTotalModel):
        return model, "margin_total"
    msg = f"Unsupported model type: {type(model).__name__}"
    raise TypeError(msg)


def _resolve_head(
    model: ml_model_core.MarginTotalModel, target: str
) -> tuple[xgb.XGBRegressor, str]:
    """Resolve the specific model head to analyze."""
    if target == "margin":
        return model.margin_model, "margin"
    if target == "total":
        return model.total_model, "total"
    msg = "Margin/total models support --target margin|total."
    raise ValueError(msg)


def main() -> int:
    """Write mean absolute SHAP per encoded column for one head, from XGBoost's own TreeSHAP."""
    args = _parse_args()

    if not args.model_path.exists():
        msg = f"Missing model: {args.model_path}"
        raise FileNotFoundError(msg)
    if not args.data_path.exists():
        msg = f"Missing dataset: {args.data_path}"
        raise FileNotFoundError(msg)

    model = joblib.load(args.model_path)
    base_model, model_kind = _select_model_component(model)
    head_model, target = _resolve_head(base_model, args.target)

    df = pd.read_csv(args.data_path)
    feature_df = ml_model_core.apply_feature_spec(df, base_model.feature_spec)
    matrix = transform_matrix(base_model.preprocessor, feature_df)
    x_matrix = np.asarray(matrix.todense()) if isinstance(matrix, spmatrix) else np.asarray(matrix)

    sample_size = max(int(args.sample_size), 1)
    if x_matrix.shape[0] > sample_size:
        rng = np.random.default_rng(int(args.random_seed))
        indices = rng.choice(x_matrix.shape[0], size=sample_size, replace=False)
        x_matrix = x_matrix[indices]

    feature_names = feature_importance.resolve_feature_names(
        base_model.preprocessor,
        head_model,
    )

    shap_values, _ = feature_importance.shap_values(head_model, x_matrix)
    mean_abs = np.abs(shap_values).mean(axis=0)

    if len(feature_names) != len(mean_abs):
        log.debug(
            "Feature name count mismatch for SHAP (names=%d, shap=%d); using f0..",
            len(feature_names),
            len(mean_abs),
        )
        feature_names = [f"f{i}" for i in range(len(mean_abs))]

    rows = [
        {"feature": name, "mean_abs_shap": float(value)}
        for name, value in zip(feature_names, mean_abs, strict=False)
    ]
    rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)

    output_path = (
        args.output_path
        if args.output_path is not None
        else args.model_path.parent / "shap_report.json"
    )
    payload = {
        "model_kind": model_kind,
        "component": "team",
        "target": target,
        "sample_size": int(x_matrix.shape[0]),
        "rows": rows,
    }
    artifacts.write_json(output_path, payload)
    log.info("Wrote SHAP report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
