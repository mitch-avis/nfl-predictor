"""Informal device check for LightGBM and XGBoost on the real feature table (2026-09-24).

One split, no walk-forward: train on the 1999-2024 regular season, predict 2025, margin target,
the numeric features between ``away_rest`` and ``home_moneyline``, no market anchoring, 200
trees at learning rate 0.0165 and depth 5 (the shared defaults). Reports the median fit time of
three runs after a warm-up, and how closely GPU predictions match CPU predictions, with sampling
on (the production setting) and off (device differences alone), next to a CPU re-seed as the
yardstick for ordinary noise, and whether the same seed on the same device repeats exactly.
Holdout MAE from one split is not a model comparison.

Run from the repo root: ``.venv/bin/python .agents/m57/lightgbm_device_check.py``.
"""

import statistics
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

games = pd.read_csv("data/completed_games_ml.csv", low_memory=False)
games = games[games["game_type"].eq("REG")]
columns = list(games.columns)
features = columns[columns.index("away_rest") : columns.index("home_moneyline") + 1]
x_all = games[features].select_dtypes("number").astype(np.float32)
y_all = (games["home_score"] - games["away_score"]).astype(np.float32)
train = (games["season"] < 2025).to_numpy()
test = (games["season"] == 2025).to_numpy()
x_train, y_train = x_all[train].to_numpy(), y_all[train].to_numpy()
x_test, y_test = x_all[test].to_numpy(), y_all[test].to_numpy()
print(f"train {x_train.shape}, test {x_test.shape}")


def fit_lightgbm(device: str, sample: bool, seed: int) -> np.ndarray:
    """Fit LightGBM and return its 2025 predictions."""
    params = {
        "objective": "regression",
        "learning_rate": 0.0165,
        "max_depth": 5,
        "num_leaves": 31,
        "seed": seed,
        "device": device,
        "num_threads": 12,
        "verbosity": -1,
        "deterministic": True,
    }
    if sample:
        params.update(bagging_fraction=0.6354, bagging_freq=1, feature_fraction=0.6098)
    booster = lgb.train(params, lgb.Dataset(x_train, label=y_train), num_boost_round=200)
    return booster.predict(x_test)


def fit_xgboost(device: str, sample: bool, seed: int) -> np.ndarray:
    """Fit XGBoost and return its 2025 predictions."""
    sampling = {"subsample": 0.6354, "colsample_bytree": 0.6098} if sample else {}
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.0165,
        max_depth=5,
        random_state=seed,
        tree_method="hist",
        device=device,
        n_jobs=12,
        **sampling,
    )
    return model.fit(x_train, y_train).predict(x_test)


def compare(label: str, left: np.ndarray, right: np.ndarray) -> None:
    """Print how far two prediction vectors are apart."""
    gap = np.abs(left - right)
    corr = np.corrcoef(left, right)[0, 1]
    print(f"  {label:36s} max {gap.max():6.3f}  mean {gap.mean():.3f}  corr {corr:.5f}")


for name, fit in (("lightgbm", fit_lightgbm), ("xgboost", fit_xgboost)):
    print(name)
    for device in ("cpu", "cuda"):
        fit(device, True, 42)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            predictions = fit(device, True, 42)
            times.append(time.perf_counter() - start)
        mae = float(np.mean(np.abs(predictions - y_test)))
        print(f"  {device:4s} {statistics.median(times):6.2f} s/fit   holdout margin MAE {mae:.3f}")
    compare("cpu vs cuda, sampling on", fit("cpu", True, 42), fit("cuda", True, 42))
    compare("cpu vs cuda, sampling off", fit("cpu", False, 42), fit("cuda", False, 42))
    compare("cpu seed 42 vs seed 7 (noise yardstick)", fit("cpu", True, 42), fit("cpu", True, 7))
    for device in ("cpu", "cuda"):
        repeat = f"{device} same seed twice (reproducibility)"
        compare(repeat, fit(device, True, 42), fit(device, True, 42))
