"""Compare the web page's feature-importance number with total gain for one saved model.

The Model page shows, per base feature, the ``base_features.combined.gain`` value of the run's
``feature_importance.json``: XGBoost's ``importance_type="gain"`` (the *average* gain per split),
summed over every encoded column of the base feature and over the margin and total heads. For a
one-hot feature that sum runs over one average per category that the trees used, so it grows with
the number of categories rather than with how much the feature reduces the loss. This script
loads the model, recomputes that number, and ranks the same base features by total gain (average
gain times splits) and by split count, with each feature's encoded-column and split counts.

Run from the repository root:

    .venv/bin/python .agents/findings_2026_09_25/importance_aggregation.py \
        models/weekly_2026_week_03_full/model.joblib \
        > .agents/findings_2026_09_25/importance_aggregation_output.txt
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import joblib

from nfl_predictor.ml import feature_importance

IMPORTANCE_TYPES = ("gain", "total_gain", "weight")
LABELS = {
    "gain": "summed average gain (the web page's number)",
    "total_gain": "total gain",
    "weight": "splits",
}
TOP = 10


def main(argv: list[str]) -> int:
    """Print the three rankings for the model at ``argv[0]``."""
    model = joblib.load(Path(argv[0]))
    names = [str(name) for name in model.preprocessor.get_feature_names_out()]
    base_of = feature_importance._build_base_feature_map(model.preprocessor, names) or {}
    totals: dict[str, dict[str, float]] = {kind: defaultdict(float) for kind in IMPORTANCE_TYPES}
    encoded: dict[str, int] = defaultdict(int)
    used: dict[str, set[str]] = defaultdict(set)
    for name in names:
        encoded[base_of.get(name, name)] += 1
    for head in (model.margin_model, model.total_model):
        booster = head.get_booster()
        for kind in IMPORTANCE_TYPES:
            scores = booster.get_score(importance_type=kind)
            for index, name in enumerate(names):
                value = scores.get(name, scores.get(f"f{index}", 0.0))
                base = base_of.get(name, name)
                totals[kind][base] += float(value)
                if kind == "weight" and value:
                    used[base].add(name)

    features = sorted(totals["gain"])
    print(f"model: {argv[0]}")
    print(
        f"base features: {len(features)}; one-hot base features: "
        f"{sorted(f for f in features if encoded[f] > 1)}"
    )
    ranks = {
        kind: {f: r for r, f in enumerate(sorted(features, key=lambda f: -totals[kind][f]), 1)}
        for kind in IMPORTANCE_TYPES
    }
    for kind in IMPORTANCE_TYPES:
        print()
        print(f"top {TOP} by {LABELS[kind]}:")
        for feature in sorted(features, key=lambda f: -totals[kind][f])[:TOP]:
            print(
                f"  {feature}: {totals[kind][feature]:.2f}"
                f" (encoded columns {encoded[feature]}, used {len(used[feature])},"
                f" splits {totals['weight'][feature]:.0f})"
            )
    print()
    print("one-hot base features, rank by each measure (of the base features):")
    for feature in features:
        if encoded[feature] > 1:
            print(
                f"  {feature}: summed average gain {ranks['gain'][feature]},"
                f" total gain {ranks['total_gain'][feature]}, splits {ranks['weight'][feature]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
