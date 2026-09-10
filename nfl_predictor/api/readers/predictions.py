"""Read a weekly predictions CSV into a table with market context and disagreement columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from nfl_predictor.api.readers import market
from nfl_predictor.api.readers.cache import cached
from nfl_predictor.api.registry import project
from nfl_predictor.api.registry.predictions import PICK_COLUMNS, PREDICTION_COLUMNS
from nfl_predictor.api.schemas.common import TablePayload


@dataclass(frozen=True)
class PredictionsSummary:
    """Headline numbers for a week's predictions."""

    games: int
    avg_confidence: float | None
    market_disagreements: int
    games_with_lines: int
    first_kickoff: str | None
    last_kickoff: str | None


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a CSV with a generous schema inference window."""
    return pl.read_csv(path, infer_schema_length=10000)


def load_frame(path: Path) -> pl.DataFrame:
    """Return the cached frame for ``path``."""
    frame: pl.DataFrame = cached(path, _read_csv)
    return frame


def _float(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite number."""
    return float(value) if market.is_number(value) else None


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    """Add market-implied probabilities and model-vs-market columns to one prediction row."""
    p_home = _float(row.get("home_win_prob"))
    ml_home = _float(row.get("home_moneyline"))
    ml_away = _float(row.get("away_moneyline"))
    raw_home = market.moneyline_to_prob(ml_home)
    raw_away = market.moneyline_to_prob(ml_away)
    novig_home, _novig_away = market.novig_pair(raw_home, raw_away)
    home_spread = _float(row.get("home_spread"))
    total_line = _float(row.get("total_line"))
    margin = _float(row.get("predicted_margin_raw"))
    if margin is None:
        margin = _float(row.get("predicted_margin"))
    total = _float(row.get("predicted_total_raw"))
    if total is None:
        total = _float(row.get("predicted_total"))
    market_margin = -home_spread if home_spread is not None else None
    favorite: str | None = None
    if novig_home is not None:
        favorite = str(row.get("home_abbr")) if novig_home >= 0.5 else str(row.get("away_abbr"))
    elif market_margin is not None and market_margin != 0:
        favorite = str(row.get("home_abbr")) if market_margin > 0 else str(row.get("away_abbr"))
    pick = row.get("predicted_winner")
    row["market_home_prob_novig"] = novig_home
    row["market_home_margin"] = market_margin
    row["edge_home_prob"] = (
        p_home - novig_home if p_home is not None and novig_home is not None else None
    )
    row["margin_vs_market"] = (
        margin - market_margin if margin is not None and market_margin is not None else None
    )
    row["total_vs_market"] = (
        total - total_line if total is not None and total_line is not None else None
    )
    row["market_favorite"] = favorite
    row["agrees_with_market"] = (pick == favorite) if favorite is not None and pick else None
    return row


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` with the derived market columns appended (row order preserved)."""
    rows = [enrich_row(row) for row in df.to_dicts()]
    if not rows:
        return df
    derived = [
        "market_home_prob_novig", "market_home_margin", "edge_home_prob", "margin_vs_market",
        "total_vs_market", "market_favorite", "agrees_with_market",
    ]  # fmt: skip
    extra = pl.DataFrame(
        {name: [row[name] for row in rows] for name in derived},
        schema={
            "market_home_prob_novig": pl.Float64,
            "market_home_margin": pl.Float64,
            "edge_home_prob": pl.Float64,
            "margin_vs_market": pl.Float64,
            "total_vs_market": pl.Float64,
            "market_favorite": pl.String,
            "agrees_with_market": pl.Boolean,
        },
    )
    return pl.concat([df.drop([c for c in derived if c in df.columns]), extra], how="horizontal")


def summarize(df: pl.DataFrame) -> PredictionsSummary:
    """Compute the headline numbers for an enriched frame."""
    rows = df.to_dicts()
    confidences = [
        r["confidence_strength"] for r in rows if market.is_number(r.get("confidence_strength"))
    ]
    kickoffs = sorted(str(r["game_datetime"]) for r in rows if r.get("game_datetime"))
    return PredictionsSummary(
        games=len(rows),
        avg_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        market_disagreements=sum(1 for r in rows if r.get("agrees_with_market") is False),
        games_with_lines=sum(1 for r in rows if r.get("market_home_prob_novig") is not None),
        first_kickoff=kickoffs[0] if kickoffs else None,
        last_kickoff=kickoffs[-1] if kickoffs else None,
    )


def read_predictions(path: Path) -> tuple[TablePayload, PredictionsSummary]:
    """Return the projected predictions table and its summary."""
    frame = enrich(load_frame(path))
    ordered = (
        frame.sort(["game_datetime", "game_id"]) if "game_datetime" in frame.columns else frame
    )
    return project(ordered, PREDICTION_COLUMNS), summarize(frame)


def read_picks(path: Path) -> TablePayload:
    """Return the confidence picks ordered from most to least confident."""
    frame = load_frame(path)
    if "confidence_rank" in frame.columns:
        frame = frame.sort("confidence_rank", descending=True)
    return project(frame, PICK_COLUMNS)


def picks_from_predictions(path: Path) -> TablePayload:
    """Derive the picks table from a predictions file when no picks file exists."""
    frame = load_frame(path)
    if "confidence_rank" in frame.columns:
        frame = frame.sort("confidence_rank", descending=True)
    return project(frame, PICK_COLUMNS)
