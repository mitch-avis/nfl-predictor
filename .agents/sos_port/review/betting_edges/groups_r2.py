"""Round two: recompute every group of BETTING_EDGES.md independently, and check the season split.

Usage: <venv>/bin/python -I groups_r2.py <checkpoint_dir> <market_lines_metadata.json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

COLS = ["game_id", "season", "week", "game_type", "home_moneyline", "away_moneyline",
        "home_spread", "predicted_margin", "actual_home_win"]


def implied(ml: float) -> float:
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def boot(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(d)
    means = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(5000)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def stats(name: str, d: pd.DataFrame) -> None:
    n = len(d)
    k = math.ceil(n / 10)
    order = np.argsort(-d["edge"].abs().to_numpy(), kind="stable")[:k]
    top = d.iloc[order]
    y = d["actual_home_win"].to_numpy(float)
    lm = (y - d["p_ml"].to_numpy()) ** 2
    ls = (y - d["p_spread"].to_numpy()) ** 2
    lo, hi = boot(ls - lm)
    print(
        f"{name}: n={n} |edge|={d['edge'].abs().mean():.4f} |gap|={d['gap'].abs().mean():.4f} "
        f"|beyond|={d['beyond'].abs().mean():.4f} top={k} "
        f"share={float((top['gap'].abs() > top['beyond'].abs()).mean()):.4f} "
        f"sign={float((np.sign(top['gap']) == np.sign(top['edge'])).mean()):.4f} "
        f"corr_gap={np.corrcoef(d['edge'], d['gap'])[0, 1]:.4f} "
        f"corr_beyond={np.corrcoef(d['edge'], d['beyond'])[0, 1]:.4f} "
        f"brier_ml={lm.mean():.5f} brier_sp={ls.mean():.5f} "
        f"diff={(ls - lm).mean():+.5f} ({lo:+.5f} to {hi:+.5f})"
    )


def main() -> int:
    frames = []
    for path in sorted(Path(sys.argv[1]).glob("fold_*.joblib")):
        payload = joblib.load(path)
        f = payload["predictions"].loc[:, COLS].copy()
        f["sigma"] = float(payload["metrics"]["floor_sigma"])
        frames.append(f)
    g = pd.concat(frames, ignore_index=True)
    h = g["home_moneyline"].map(implied)
    a = g["away_moneyline"].map(implied)
    g["p_ml"] = h / (h + a)
    g["p_model"] = ndtr(g["predicted_margin"] / g["sigma"])
    g["p_spread"] = ndtr(-g["home_spread"] / g["sigma"])
    g["edge"] = g["p_model"] - g["p_ml"]
    g["gap"] = g["p_spread"] - g["p_ml"]
    g["beyond"] = g["p_model"] - g["p_spread"]

    # Season classification from the data alone: a season whose moneyline probability is a
    # single value within every spread cell is consistent with a spread-derived moneyline.
    cells = g.groupby(["season", "home_spread"])["p_ml"].nunique()
    data_derived = sorted(int(s) for s, v in cells.groupby("season").max().items() if v == 1)
    data_real = sorted(int(s) for s, v in cells.groupby("season").max().items() if v > 1)
    print(f"data: one moneyline per spread cell (derived-looking): {data_derived}")
    print(f"data: several moneylines per spread cell (real pairs): {data_real}")
    meta = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["seasons"]
    meta_real = sorted(
        int(s) for s, c in meta.items()
        if 2007 <= int(s) <= 2025
        and c["moneylines_from_map"] == 0 and c["moneylines_from_fixed_conversion"] == 0
    )
    print(f"metadata: seasons with no derived moneyline: {meta_real}")
    print(f"metadata 2023: {meta['2023']}")
    # The 2023 single-game cells: could any of them be a real pair the cell test cannot see?
    c23 = g[g["season"] == 2023].groupby("home_spread").size()
    print(f"2023 spread cells with a single game: {int((c23 == 1).sum())} games")
    # Does every 2023 game's moneyline probability equal the season map's value for its spread?
    # (only checkable on cells with two or more games; already 1 distinct value there)

    stats("All scored games", g)
    stats("Real moneyline pairs", g[g["season"].isin(meta_real)])
    stats("2025 alone", g[g["season"] == 2025])
    return 0


if __name__ == "__main__":
    sys.exit(main())
