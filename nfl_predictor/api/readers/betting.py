"""Derive the betting report from a predictions file.

The report is recomputed from the predictions rather than read from the run's
``*_betting_report.csv`` so that a lines refresh followed by a re-predict never leaves a stale
report behind, and so the moneyline, spread, and total sections share one set of formulas
(the ones in the betting workbook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from nfl_predictor.api.readers import market
from nfl_predictor.api.readers.predictions import enrich, load_frame
from nfl_predictor.api.registry import project
from nfl_predictor.api.registry.betting import BETTING_COLUMNS
from nfl_predictor.api.schemas.common import TablePayload

BREAK_EVEN_110 = market.moneyline_to_prob(market.DEFAULT_SPREAD_TOTAL_ODDS) or 0.5238


def _f(value: object) -> float | None:
    """Return ``value`` as a float when finite."""
    return float(value) if market.is_number(value) else None


def _first(row: dict[str, Any], *keys: str) -> float | None:
    """Return the first finite value among ``keys``."""
    for key in keys:
        value = _f(row.get(key))
        if value is not None:
            return value
    return None


def _moneyline_block(row: dict[str, Any], p_home: float | None) -> dict[str, Any]:
    """Compute the moneyline recommendation columns."""
    ml_home = _f(row.get("home_moneyline"))
    ml_away = _f(row.get("away_moneyline"))
    raw_home = market.moneyline_to_prob(ml_home)
    raw_away = market.moneyline_to_prob(ml_away)
    out: dict[str, Any] = {
        "model_home_prob": p_home,
        "market_home_prob_raw": raw_home,
        "market_away_prob_raw": raw_away,
        "model_fair_home_moneyline": market.prob_to_moneyline(p_home),
        "model_fair_away_moneyline": market.prob_to_moneyline(1 - p_home)
        if p_home is not None
        else None,
        "moneyline_value_side": None,
        "moneyline_edge_prob": None,
        "moneyline_action": "PASS",
        "moneyline_confidence_1_10": 1,
        "moneyline_ev": None,
    }
    if p_home is None or raw_home is None or raw_away is None or ml_home is None or ml_away is None:
        return out
    edge_home = p_home - raw_home
    edge_away = (1 - p_home) - raw_away
    if edge_home >= edge_away:
        side, edge, prob, price = row.get("home_abbr"), edge_home, p_home, ml_home
    else:
        side, edge, prob, price = row.get("away_abbr"), edge_away, 1 - p_home, ml_away
    edge = max(0.0, edge)
    out.update(
        moneyline_value_side=side,
        moneyline_edge_prob=edge,
        moneyline_action=market.action_label(edge),
        moneyline_confidence_1_10=market.confidence_1_to_10(edge),
        moneyline_ev=market.expected_value(prob, price),
    )
    return out


def _spread_block(row: dict[str, Any]) -> dict[str, Any]:
    """Compute the spread recommendation columns."""
    mu = _first(row, "predicted_margin_raw", "predicted_margin")
    home_spread = _f(row.get("home_spread"))
    sigma = market.sigma_from_quantiles(
        row.get("predicted_margin_p10"),
        row.get("predicted_margin_p90"),
        market.DEFAULT_SIGMA_MARGIN,
    )
    out: dict[str, Any] = {
        "spread_sigma": sigma,
        "spread_p_home_cover": None,
        "spread_value_side": None,
        "spread_edge_prob": None,
        "spread_edge_points": None,
        "spread_action": "PASS",
        "spread_confidence_1_10": 1,
        "spread_ev": None,
    }
    if mu is None or home_spread is None:
        return out
    p_home_cover = market.prob_exceeds(-home_spread, mu, sigma)
    p_away_cover = 1.0 - p_home_cover
    edge_home = p_home_cover - BREAK_EVEN_110
    edge_away = p_away_cover - BREAK_EVEN_110
    if edge_home >= edge_away:
        side, edge, prob = row.get("home_abbr"), edge_home, p_home_cover
    else:
        side, edge, prob = row.get("away_abbr"), edge_away, p_away_cover
    edge = max(0.0, edge)
    out.update(
        spread_p_home_cover=p_home_cover,
        spread_value_side=side,
        spread_edge_prob=edge,
        spread_edge_points=mu + home_spread,
        spread_action=market.action_label(edge),
        spread_confidence_1_10=market.confidence_1_to_10(edge),
        spread_ev=market.expected_value(prob, market.DEFAULT_SPREAD_TOTAL_ODDS),
    )
    return out


def _total_block(row: dict[str, Any]) -> dict[str, Any]:
    """Compute the (informational) total columns."""
    mu = _first(row, "predicted_total_raw", "predicted_total")
    line = _f(row.get("total_line"))
    sigma = market.sigma_from_quantiles(
        row.get("predicted_total_p10"), row.get("predicted_total_p90"), market.DEFAULT_SIGMA_TOTAL
    )
    out: dict[str, Any] = {
        "total_p_over": None,
        "total_value_side": None,
        "total_edge_prob": None,
        "total_edge_points": None,
        "total_action": "PASS",
    }
    if mu is None or line is None:
        return out
    p_over = market.prob_exceeds(line, mu, sigma)
    edge_over = p_over - BREAK_EVEN_110
    edge_under = (1 - p_over) - BREAK_EVEN_110
    side, edge = ("OVER", edge_over) if edge_over >= edge_under else ("UNDER", edge_under)
    edge = max(0.0, edge)
    out.update(
        total_p_over=p_over,
        total_value_side=side,
        total_edge_prob=edge,
        total_edge_points=mu - line,
        total_action=market.action_label(edge),
    )
    return out


def betting_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return ``row`` plus every betting column."""
    p_home = _f(row.get("home_win_prob"))
    out = dict(row)
    out.update(_moneyline_block(row, p_home))
    out.update(_spread_block(row))
    out.update(_total_block(row))
    return out


def build_betting_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return the betting table for an enriched predictions frame."""
    rows = [betting_row(row) for row in df.to_dicts()]
    if not rows:
        return df
    return pl.DataFrame(rows, infer_schema_length=None)


def read_betting(path: Path) -> TablePayload:
    """Return the betting table derived from the predictions at ``path``, sorted by kickoff."""
    frame = enrich(load_frame(path))
    table = build_betting_frame(frame)
    if "game_datetime" in table.columns:
        table = table.sort(["game_datetime", "game_id"])
    return project(table, BETTING_COLUMNS)
