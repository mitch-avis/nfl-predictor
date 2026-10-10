"""Context: the favorite-side gap by spread on real-moneyline seasons versus map-derived ones."""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy.special import ndtr
rows = []
for p in sorted(Path(sys.argv[1]).glob("fold_*.joblib")):
    d = joblib.load(p)
    f = d["predictions"][["season", "home_spread", "home_moneyline", "away_moneyline", "predicted_margin"]].copy()
    f["sigma"] = float(d["metrics"]["floor_sigma"])
    rows.append(f)
g = pd.concat(rows, ignore_index=True)
imp = lambda m: 100 / (m + 100) if m > 0 else -m / (-m + 100)
h = g["home_moneyline"].map(imp); a = g["away_moneyline"].map(imp)
g["p_ml"] = h / (h + a)
g["gap"] = ndtr(-g["home_spread"] / g["sigma"]) - g["p_ml"]
g = g[g["home_spread"] != 0]
# favorite-side gap: the curve's probability for the favorite minus the moneyline's
g["fav_gap"] = np.where(g["home_spread"] < 0, g["gap"], -g["gap"])
g["bucket"] = pd.cut(g["home_spread"].abs(), [0, 2.5, 3.5, 6.5, 7.5, 10.5, 30])
g["real"] = np.where(g["season"].isin([2022, 2024, 2025]), "real pairs", "map-derived")
print(g.pivot_table(index="bucket", columns="real", values="fav_gap", aggfunc=["mean", "count"], observed=True).round(4).to_string())
for name, sub in g.groupby("real"):
    means = sub.groupby("bucket", observed=True)["fav_gap"].transform("mean")
    r2 = 1 - ((sub["fav_gap"] - means) ** 2).sum() / ((sub["fav_gap"] - sub["fav_gap"].mean()) ** 2).sum()
    print(f"{name}: share of favorite-side gap variance explained by the spread bucket alone: {r2:.3f}")
