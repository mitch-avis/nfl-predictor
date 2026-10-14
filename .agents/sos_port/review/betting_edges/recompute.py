"""Independent recomputation of the betting-edge decomposition from the fold checkpoints.

Written by the reviewer from the rule text alone (decisions/betting_edges.md, entry 1); it does
not import the implementer's script. Usage:

    <venv>/bin/python -I recompute.py <checkpoint_dir> [<run_metadata.json>]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

KEEP = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_abbr",
    "away_abbr",
    "home_moneyline",
    "away_moneyline",
    "home_spread",
    "away_spread",
    "predicted_margin",
    "deterministic_home_win_prob",
    "home_win_prob",
    "market_home_win_prob",
    "actual_home_win",
    "actual_margin",
    "home_score",
    "away_score",
    "floor_sigma",
]
ACTION_BOUNDS = np.array([0.02, 0.04, 0.07, 0.10])
ACTION_NAMES = np.array(["PASS", "LEAN", "SMALL", "MEDIUM", "STRONG"])
CONF_BOUNDS = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10])


def implied(ml: np.ndarray) -> np.ndarray:
    """American odds to raw implied probability; 0 or non-finite gives NaN."""
    ml = np.asarray(ml, dtype=float)
    out = np.full(ml.shape, np.nan)
    pos = np.isfinite(ml) & (ml > 0)
    neg = np.isfinite(ml) & (ml < 0)
    out[pos] = 100.0 / (ml[pos] + 100.0)
    out[neg] = -ml[neg] / (-ml[neg] + 100.0)
    return out


def action_label(abs_edge: np.ndarray) -> np.ndarray:
    """Label = first bound the edge is strictly below; at or above every bound is STRONG."""
    idx = np.searchsorted(ACTION_BOUNDS, abs_edge, side="right")
    idx = np.where(np.isfinite(abs_edge), idx, len(ACTION_BOUNDS))
    return ACTION_NAMES[idx]


def conf_score(abs_edge: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(CONF_BOUNDS, abs_edge, side="right")
    idx = np.where(np.isfinite(abs_edge), idx, len(CONF_BOUNDS))
    return idx + 1


def boot_loop_default_rng(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def boot_batch_default_rng(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def boot_legacy(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rs = np.random.RandomState(seed)
    n = len(d)
    means = np.array([d[rs.randint(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ckpt = Path(sys.argv[1])
    paths = sorted(ckpt.glob("fold_*.joblib"))
    print(f"fold files: {len(paths)}")
    frames = []
    fingerprints = set()
    sigma_row_mismatch = 0
    devices = Counter()
    early = Counter()
    best_iters = Counter()
    fallback_weeks = 0
    for path in paths:
        payload = joblib.load(path)
        fingerprints.add(payload["fingerprint"])
        metrics = payload["metrics"]
        sigma = float(metrics["floor_sigma"])
        fallback_weeks += bool(metrics.get("floor_sigma_fallback"))
        devices[metrics.get("xgb_device")] += 1
        for key, value in metrics.items():
            if key.endswith(".early_stopped"):
                early[(key, bool(value))] += 1
            if key == "margin_model.best_iteration":
                best_iters[int(value)] += 1
        pr = payload["predictions"]
        cols = [c for c in KEEP if c in pr.columns]
        f = pr.loc[:, cols].copy()
        if "floor_sigma" in f.columns:
            sigma_row_mismatch += int((f["floor_sigma"].astype(float) != sigma).sum())
        f["sigma_metric"] = sigma
        f["fold_season"] = int(payload["season"])
        f["fold_week"] = int(payload["week"])
        frames.append(f)
    g = pd.concat(frames, ignore_index=True)
    print(f"fingerprints: {sorted(fingerprints)}")
    print(f"devices: {dict(devices)}")
    print(f"margin_model.best_iteration counts: {dict(best_iters)}")
    print(f"early_stopped (key, value) counts: {dict(early)}")
    print(f"weeks on the fallback sigma: {fallback_weeks}")
    print(f"rows whose floor_sigma column differs from the fold metric: {sigma_row_mismatch}")
    print(f"rows: {len(g)}, unique game_id: {g['game_id'].nunique()}")
    print(f"season range: {g['season'].min()}-{g['season'].max()}, game_type: {dict(Counter(g['game_type']))}")
    print(f"fold season/week equals row season/week: {bool(((g['season'] == g['fold_season']) & (g['week'] == g['fold_week'])).all())}")

    sigma = g["sigma_metric"].to_numpy(float)
    margin = g["predicted_margin"].to_numpy(float)
    stored = g["deterministic_home_win_prob"].to_numpy(float)
    floor = ndtr(margin / sigma)
    print(f"floor check: max |Phi(margin/sigma) - deterministic_home_win_prob| = {np.max(np.abs(floor - stored)):.3e}")
    if "home_win_prob" in g.columns:
        print(f"  (home_win_prob vs deterministic: max diff {np.nanmax(np.abs(g['home_win_prob'].to_numpy(float) - stored)):.3e})")

    # outcome
    y = g["actual_home_win"].to_numpy(float)
    ties = (g["home_score"] == g["away_score"]).to_numpy()
    y_scores = (g["home_score"] > g["away_score"]).astype(float).to_numpy()
    print(f"outcome: actual_home_win equals (home_score > away_score): {bool(np.array_equal(y, y_scores))}; ties: {int(ties.sum())}, their actual_home_win values: {sorted(set(y[ties].tolist()))}")

    # sign convention, from the data
    hs = g["home_spread"].to_numpy(float)
    as_ = g["away_spread"].to_numpy(float)
    hml = g["home_moneyline"].to_numpy(float)
    aml = g["away_moneyline"].to_numpy(float)
    h_raw, a_raw = implied(hml), implied(aml)
    p_ml = h_raw / (h_raw + a_raw)
    ok = np.isfinite(hs) & np.isfinite(p_ml)
    print("sign convention:")
    print(f"  away_spread == -home_spread on {int(np.sum(as_[ok] == -hs[ok]))} of {int(ok.sum())} rows")
    fav = p_ml > 0.5
    dog = p_ml < 0.5
    print(f"  ML favorite at home (p_ml > 0.5): {int((ok & fav).sum())}; of them home_spread < 0: {int((ok & fav & (hs < 0)).sum())}, == 0: {int((ok & fav & (hs == 0)).sum())}, > 0: {int((ok & fav & (hs > 0)).sum())}")
    print(f"  ML underdog at home (p_ml < 0.5): {int((ok & dog).sum())}; of them home_spread > 0: {int((ok & dog & (hs > 0)).sum())}, == 0: {int((ok & dog & (hs == 0)).sum())}, < 0: {int((ok & dog & (hs < 0)).sum())}")
    print(f"  corr(-home_spread, p_ml) = {np.corrcoef(-hs[ok], p_ml[ok])[0, 1]:.4f}; corr(-home_spread, predicted_margin) = {np.corrcoef(-hs[ok], margin[ok])[0, 1]:.4f}")
    am = g["actual_margin"].to_numpy(float)
    slope, intercept = np.polyfit(-hs[ok], am[ok], 1)
    print(f"  OLS actual_margin ~ -home_spread: slope {slope:.3f}, intercept {intercept:.3f}; mean actual margin when home_spread<0: {am[ok & (hs < 0)].mean():.2f}, when >0: {am[ok & (hs > 0)].mean():.2f}")
    disagree = ok & ((fav & (hs > 0)) | (dog & (hs < 0)))
    if disagree.any():
        print("  rows where the spread and moneyline favor different sides:")
        print(g.loc[disagree, ["game_id", "home_spread", "home_moneyline", "away_moneyline"]].to_string())

    # cross-check p_ml against the report's own conversion and the walk-forward market column
    sys.path.insert(0, sys.argv[3]) if len(sys.argv) > 3 else None
    from nfl_predictor.reporting import betting_report as br  # noqa: PLC0415

    rep_h = np.array([br._moneyline_to_implied_prob(v) for v in hml])
    rep_a = np.array([br._moneyline_to_implied_prob(v) for v in aml])
    rep_novig = np.array([br._novig_pair(a, b)[0] for a, b in zip(rep_h, rep_a, strict=True)])
    both = np.isfinite(rep_novig) | np.isfinite(p_ml)
    print(f"p_ml: max |mine - report conversion| = {np.nanmax(np.abs(rep_novig[both] - p_ml[both])):.3e}; NaN pattern equal: {bool(np.array_equal(np.isfinite(rep_novig), np.isfinite(p_ml)))}")
    if "market_home_win_prob" in g.columns:
        mk = g["market_home_win_prob"].to_numpy(float)
        sel = np.isfinite(p_ml)
        print(f"p_ml vs fold market_home_win_prob: max diff {np.max(np.abs(mk[sel] - p_ml[sel])):.3e}")

    # scored set and decomposition
    window = g["season"].between(2007, 2025).to_numpy() & (g["game_type"] == "REG").to_numpy()
    p_spread = ndtr(-hs / sigma)
    scored = window & np.isfinite(p_ml) & np.isfinite(p_spread)
    print(f"in window: {int(window.sum())}; missing moneyline: {int((window & ~np.isfinite(p_ml)).sum())}; missing spread: {int((window & ~np.isfinite(hs)).sum())}; scored: {int(scored.sum())}")
    pm = floor[scored]
    pl = p_ml[scored]
    ps = p_spread[scored]
    ys = y[scored]
    edge = pm - pl
    gap = ps - pl
    beyond = pm - ps
    print(f"identity max |edge - (beyond + gap)|: {np.max(np.abs(edge - (beyond + gap))):.3e}")
    n = len(edge)
    print(f"N = {n}")
    print(f"mean |edge| = {np.abs(edge).mean():.6f}")
    print(f"mean |gap| = {np.abs(gap).mean():.6f}")
    print(f"mean |beyond| = {np.abs(beyond).mean():.6f}")
    k = math.ceil(n / 10)
    order = np.argsort(-np.abs(edge), kind="stable")
    top = order[:k]
    print(f"top decile size ceil(n/10) = {k}; |edge| at rank {k}: {np.abs(edge)[order[k - 1]]:.6f}, rank {k + 1}: {np.abs(edge)[order[k]]:.6f}")
    share_gap = float(np.mean(np.abs(gap[top]) > np.abs(beyond[top])))
    share_sign = float(np.mean(np.sign(gap[top]) == np.sign(edge[top])))
    print(f"top decile share |gap| > |beyond| = {share_gap:.6f} ({int(np.sum(np.abs(gap[top]) > np.abs(beyond[top])))}/{k})")
    print(f"top decile share sign(gap) == sign(edge) = {share_sign:.6f} ({int(np.sum(np.sign(gap[top]) == np.sign(edge[top])))}/{k})")
    print(f"top decile: mean |edge| {np.abs(edge[top]).mean():.4f}, mean |gap| {np.abs(gap[top]).mean():.4f}, mean |beyond| {np.abs(beyond[top]).mean():.4f}")
    print(f"corr(edge, gap) = {np.corrcoef(edge, gap)[0, 1]:.6f}")
    print(f"corr(edge, beyond) = {np.corrcoef(edge, beyond)[0, 1]:.6f}")
    print(f"corr(gap, beyond) = {np.corrcoef(gap, beyond)[0, 1]:.6f}")
    print(f"var(edge) = {edge.var():.3e}, var(gap) = {gap.var():.3e}, var(beyond) = {beyond.var():.3e}, 2cov(gap,beyond) = {2 * np.cov(gap, beyond, bias=True)[0, 1]:.3e}")
    rule_branch = "base labels on beyond (held for the user)" if share_gap > 0.5 else "keep labels on edge"
    print(f"RULE: share {share_gap:.4f} vs threshold > 0.5 -> {rule_branch}")

    loss_ml = (ys - pl) ** 2
    loss_sp = (ys - ps) ** 2
    d = loss_sp - loss_ml
    print(f"Brier p_ml = {loss_ml.mean():.6f}; Brier p_spread = {loss_sp.mean():.6f}; diff (spread - ml) = {d.mean():+.6f}")
    print(f"  bootstrap default_rng(0), loop of 5000: {boot_loop_default_rng(d, 5000, 0)}")
    print(f"  bootstrap default_rng(0), one batch:    {boot_batch_default_rng(d, 5000, 0)}")
    print(f"  bootstrap RandomState(0), loop:         {boot_legacy(d, 5000, 0)}")
    print(f"  (Brier of p_model on the same games: {((ys - pm) ** 2).mean():.6f})")

    # label-change count: action and confidence on |beyond| instead of the report's edge
    # the report's moneyline_edge_prob is max(|edge_home|, |edge_away|)
    pl_away = (a_raw / (h_raw + a_raw))[scored]
    for name, p_model in (("unrounded floor", pm), ("published 4-decimal clipped floor", np.clip(np.round(pm, 4), 0.0001, 0.9999))):
        e_home = p_model - pl
        e_away = (1.0 - p_model) - pl_away
        e_rep = np.maximum(np.abs(e_home), np.abs(e_away))
        b = np.abs(p_model - ps)
        lab_e = action_label(e_rep)
        lab_b = action_label(b)
        diff = lab_e != lab_b
        conf_diff = conf_score(e_rep) != conf_score(b)
        print(f"labels ({name}): action differs on {int(diff.sum())} of {n} ({diff.mean():.4f}); confidence 1-10 differs on {int(conf_diff.sum())}")
        print(f"   max | |edge_away| - |edge_home| | = {np.max(np.abs(np.abs(e_away) - np.abs(e_home))):.2e}")
        tab = pd.crosstab(pd.Series(lab_e, name="label on edge"), pd.Series(lab_b, name="label on beyond"))
        tab = tab.reindex(index=ACTION_NAMES, columns=ACTION_NAMES, fill_value=0)
        print(tab.to_string())
        rank = {v: i for i, v in enumerate(ACTION_NAMES)}
        down = int(sum(rank[a] > rank[c] for a, c in zip(lab_e, lab_b, strict=True)))
        up = int(sum(rank[a] < rank[c] for a, c in zip(lab_e, lab_b, strict=True)))
        print(f"   lower label on beyond: {down}; higher label on beyond: {up}")
        # how many top-decile games change label
        print(f"   top-decile games whose action changes: {int(diff[top].sum())} of {k}")
        side_flip = np.sign(e_home) != np.sign(p_model - ps)
        print(f"   games where beyond points to the other side than edge: {int(side_flip.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
