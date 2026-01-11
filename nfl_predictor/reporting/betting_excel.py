"""Excel betting report template generation.

This module writes a spreadsheet template intended to be updated with live
sportsbook lines (FanDuel or similar).

Design goals:
- Keep the workbook self-contained: users can paste live moneylines/spreads/totals
  and recommendations update via formulas.
- Avoid any claim of profitability; this is decision support.
- Use the same edge/action heuristics as scripts/betting_pipeline.py.

Implementation notes:
- We generate `.xlsx` via openpyxl. Writing `.xlsb` is not supported directly
  from pure-Python here; users can "Save As" in Excel to convert to `.xlsb` if
  required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from nfl_predictor.utils.logger import log

# For a Normal distribution, sigma can be estimated from p10/p90 via:
#   sigma ≈ (p90 - p10) / (2 * z0.90)
# where z0.90 ≈ 1.281551565545.
NORMAL_Z_P90 = 1.281551565545
P10_P90_TO_SIGMA_DENOM = 2 * NORMAL_Z_P90

# Fallback uncertainty when quantiles are missing.
# These match the defaults used on the Live sheet.
DEFAULT_SIGMA_MARGIN = 13
DEFAULT_SIGMA_TOTAL = 16

# Default market odds for spread/total when not provided.
DEFAULT_SPREAD_TOTAL_ODDS = -110


@dataclass(frozen=True)
class ExcelTemplateConfig:
    """Configuration for the generated workbook."""

    sheet_name: str = "Bets"
    instructions_sheet_name: str = "README"
    live_sheet_name: str = "Live"


def write_betting_template_xlsx(
    *,
    predictions: pd.DataFrame,
    out_path: Path,
    cfg: Optional[ExcelTemplateConfig] = None,
) -> Path:
    """Write an Excel template with live-odds inputs and formulas.

    Args:
        predictions: The predictions table (from predictions.csv). Must include at least:
            away_abbr, home_abbr, home_win_prob, predicted_margin, predicted_total,
            away_moneyline, home_moneyline.
        out_path: Output `.xlsx` path.
        cfg: Optional template config.

    Returns:
        The written path.
    """

    cfg = cfg or ExcelTemplateConfig()

    required = [
        "away_abbr",
        "home_abbr",
        "home_win_prob",
        "predicted_margin",
        "predicted_total",
    ]
    missing = [c for c in required if c not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openpyxl is required to write .xlsx templates. Install it via pip."
        ) from exc

    wb = openpyxl.Workbook()

    # README / instructions.
    ws_readme = wb.active
    if ws_readme is None:  # pragma: no cover
        ws_readme = wb.create_sheet(title=cfg.instructions_sheet_name)
    ws_readme.title = cfg.instructions_sheet_name
    ws_readme["A1"].value = "NFL Predictor Betting Template (Decision Support)"
    ws_readme["A1"].font = Font(bold=True, size=14)
    ws_readme["A3"].value = "How to use:"
    ws_readme["A3"].font = Font(bold=True)
    ws_readme["A4"].value = "1) Paste live moneylines/spread/total into the *Live* columns."
    ws_readme["A5"].value = "2) Recommendations update automatically (edges/actions)."
    ws_readme["A6"].value = "3) Convert to .xlsb via Excel Save As if desired."
    ws_readme["A8"].value = "Notes:"
    ws_readme["A8"].font = Font(bold=True)
    ws_readme["A9"].value = (
        "- Market raw prob is implied from a single offered line (includes vig). "
        "No-vig re-normalizes home/away to sum to 1."
    )
    ws_readme["A10"].value = (
        "- Action thresholds: PASS <2%, LEAN <4%, SMALL <7%, MEDIUM <10%, STRONG >=10%."
    )
    ws_readme["A11"].value = (
        "- Live recommendations use live odds + model probabilities. "
        "Spread/total cover probs are approximated from p10/p50/p90 quantiles when available."
    )
    ws_readme["A12"].value = (
        "- The Live tab is a lightweight heuristic: enter score + minutes remaining to get "
        "a state-adjusted probability/edge view. It is not a trained in-game model."
    )

    # Main sheet.
    ws = wb.create_sheet(title=cfg.sheet_name)

    # Column layout.
    columns = [
        # Identifiers
        "season",
        "week",
        "date",
        "game_id",
        "away_abbr",
        "home_abbr",
        # Model input (closing/opening lines captured in the dataset)
        "away_spread_model_input",
        "home_spread_model_input",
        "away_moneyline_model_input",
        "home_moneyline_model_input",
        "total_model_input",
        # Model outputs
        "model_away_prob",
        "model_home_prob",
        "predicted_away_score",
        "predicted_home_score",
        "predicted_margin",
        "predicted_total",
        # Optional diagnostic columns (quantiles)
        "predicted_margin_p10",
        "predicted_margin_p50",
        "predicted_margin_p90",
        "predicted_total_p10",
        "predicted_total_p50",
        "predicted_total_p90",
        # Live inputs (user editable)
        "away_spread_live",
        "away_spread_odds_live",
        "home_spread_live",
        "home_spread_odds_live",
        "away_moneyline_live",
        "home_moneyline_live",
        "total_live",
        "total_over_odds_live",
        "total_under_odds_live",
        # Quick deltas
        "spread_edge_points_home",
        # Market probabilities (from moneyline live)
        "market_away_prob_raw",
        "market_home_prob_raw",
        "market_away_prob_novig",
        "market_home_prob_novig",
        # Model vs market disagreement (no-vig)
        "edge_away_prob",
        "edge_home_prob",
        # Live betting recommendations (spread)
        "spread_value_side",
        "spread_edge_prob",
        "spread_action",
        "spread_confidence_1_10",
        "spread_ev",
        # Live betting recommendations (moneyline)
        "money_value_side",
        "money_edge_prob",
        "money_action",
        "money_confidence_1_10",
        "moneyline_ev",
        # Live betting recommendations (total)
        "total_value_side",
        "total_edge_prob",
        "total_action",
        "total_confidence_1_10",
        "total_ev",
        # Hidden helper columns to keep recommendation formulas simple.
        "spread_mu_margin",
        "spread_sigma_margin",
        "spread_p_home_cover",
        "spread_p_away_cover",
        "spread_implied_home_prob",
        "spread_implied_away_prob",
        "spread_edge_home",
        "spread_edge_away",
        "total_mu",
        "total_sigma",
        "total_p_over",
        "total_p_under",
        "total_implied_over_prob",
        "total_implied_under_prob",
        "total_edge_over",
        "total_edge_under",
    ]

    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # Build a display DataFrame with safe columns.
    df = predictions.copy()
    for c in ("season", "week", "date", "game_id"):
        if c not in df.columns:
            df[c] = pd.NA

    df["date"] = df["date"].astype(str)

    df["model_home_prob"] = df["home_win_prob"].astype(float)
    df["model_away_prob"] = 1.0 - df["model_home_prob"]

    # Model inputs: keep both sides explicit for easier live editing.
    df["home_spread_model_input"] = df.get("home_spread", pd.NA)
    df["away_spread_model_input"] = df.get("away_spread", pd.NA)
    if "away_spread_model_input" in df.columns and "home_spread_model_input" in df.columns:
        # If away is missing but home exists, fill away = -home (common convention).
        df["away_spread_model_input"] = df["away_spread_model_input"].fillna(
            -pd.to_numeric(df["home_spread_model_input"], errors="coerce")
        )

    df["home_moneyline_model_input"] = df.get("home_moneyline", pd.NA)
    df["away_moneyline_model_input"] = df.get("away_moneyline", pd.NA)
    df["total_model_input"] = df.get("total_line", pd.NA)

    # Predicted scores: prefer raw model outputs when present.
    df["predicted_home_score"] = df.get("predicted_home_score_raw", df.get("predicted_home_score"))
    df["predicted_away_score"] = df.get("predicted_away_score_raw", df.get("predicted_away_score"))
    if "predicted_home_score" not in df.columns or "predicted_away_score" not in df.columns:
        # Fallback from margin/total.
        margin = pd.to_numeric(df["predicted_margin"], errors="coerce")
        total = pd.to_numeric(df["predicted_total"], errors="coerce")
        df["predicted_home_score"] = (total + margin) / 2.0
        df["predicted_away_score"] = (total - margin) / 2.0

    # Predicted margin/total: prefer raw if present.
    df["predicted_margin"] = df.get("predicted_margin_raw", df.get("predicted_margin"))
    df["predicted_total"] = df.get("predicted_total_raw", df.get("predicted_total"))

    # Quantiles (optional): keep for probability approximations.
    for q in (
        "predicted_margin_p10",
        "predicted_margin_p50",
        "predicted_margin_p90",
        "predicted_total_p10",
        "predicted_total_p50",
        "predicted_total_p90",
    ):
        if q not in df.columns:
            df[q] = pd.NA

    # Default live inputs to model inputs when present.
    df["home_moneyline_live"] = df.get("home_moneyline", df["home_moneyline_model_input"])
    df["away_moneyline_live"] = df.get("away_moneyline", df["away_moneyline_model_input"])

    df["home_spread_live"] = df.get("home_spread", df["home_spread_model_input"])
    df["away_spread_live"] = df.get("away_spread", df["away_spread_model_input"])

    df["total_live"] = df.get("total_line", df["total_model_input"])

    # Live odds inputs.
    # Default spread/total odds to -110 to avoid N/A propagation in dependent formulas.
    df["home_spread_odds_live"] = DEFAULT_SPREAD_TOTAL_ODDS
    df["away_spread_odds_live"] = DEFAULT_SPREAD_TOTAL_ODDS
    df["total_over_odds_live"] = DEFAULT_SPREAD_TOTAL_ODDS
    df["total_under_odds_live"] = DEFAULT_SPREAD_TOTAL_ODDS

    # Computed outputs (filled by formulas).
    for c in (
        "market_home_prob_raw",
        "market_away_prob_raw",
        "market_home_prob_novig",
        "market_away_prob_novig",
        "edge_home_prob",
        "edge_away_prob",
        "spread_edge_points_home",
        "spread_value_side",
        "spread_edge_prob",
        "spread_action",
        "spread_confidence_1_10",
        "spread_ev",
        "money_value_side",
        "money_edge_prob",
        "money_action",
        "money_confidence_1_10",
        "moneyline_ev",
        "total_value_side",
        "total_edge_prob",
        "total_action",
        "total_confidence_1_10",
        "total_ev",
    ):
        if c not in df.columns:
            df[c] = pd.NA

    # Excel formula helpers.
    # Row numbers are 1-based in Excel; header is row 1.
    def r(i: int) -> int:
        return 2 + i

    # Map col name -> Excel column letter (A, B, ...)
    def col_letter(idx_0: int) -> str:
        # openpyxl has utils, but keep it simple and local.
        idx = idx_0 + 1
        letters = ""
        while idx:
            idx, rem = divmod(idx - 1, 26)
            letters = chr(65 + rem) + letters
        return letters

    col_index = {name: j for j, name in enumerate(columns)}

    def addr(row: int, col_name: str) -> str:
        return f"{col_letter(col_index[col_name])}{row}"

    records = df.to_dict(orient="records")
    for i, row in enumerate(records):
        excel_row = r(i)
        row_values: list[object] = []
        for c in columns:
            value = row.get(c, None)
            if value is pd.NA or (hasattr(pd, "isna") and pd.isna(value)):
                value = None
            row_values.append(value)
        ws.append(row_values)

        # Moneyline -> implied prob
        home_ml = addr(excel_row, "home_moneyline_live")
        away_ml = addr(excel_row, "away_moneyline_live")
        market_home_raw = addr(excel_row, "market_home_prob_raw")
        market_away_raw = addr(excel_row, "market_away_prob_raw")
        ws[market_home_raw].value = (
            f"=IF(OR(ISBLANK({home_ml}),{home_ml}=0),NA(),IF({home_ml}>0,100/({home_ml}+100),-({home_ml})/(-({home_ml})+100)))"
        )
        ws[market_away_raw].value = (
            f"=IF(OR(ISBLANK({away_ml}),{away_ml}=0),NA(),IF({away_ml}>0,100/({away_ml}+100),-({away_ml})/(-({away_ml})+100)))"
        )

        # No-vig normalization
        market_home_novig = addr(excel_row, "market_home_prob_novig")
        market_away_novig = addr(excel_row, "market_away_prob_novig")
        ws[market_home_novig].value = (
            f"=IF(OR(ISNA({market_home_raw}),ISNA({market_away_raw})),NA(),{market_home_raw}/({market_home_raw}+{market_away_raw}))"
        )
        ws[market_away_novig].value = (
            f"=IF(OR(ISNA({market_home_raw}),ISNA({market_away_raw})),NA(),{market_away_raw}/({market_home_raw}+{market_away_raw}))"
        )

        model_home = addr(excel_row, "model_home_prob")
        model_away = addr(excel_row, "model_away_prob")
        edge_home = addr(excel_row, "edge_home_prob")
        edge_away = addr(excel_row, "edge_away_prob")
        ws[edge_home].value = f"={model_home}-{market_home_novig}"
        ws[edge_away].value = f"={model_away}-{market_away_novig}"

        # --- Spread recommendation using live line + odds (approx prob from quantiles) ---
        spread_edge_points_home = addr(excel_row, "spread_edge_points_home")
        predicted_margin = addr(excel_row, "predicted_margin")
        home_spread_live = addr(excel_row, "home_spread_live")
        ws[spread_edge_points_home].value = (
            f"=IF(OR(ISBLANK({home_spread_live}),ISBLANK({predicted_margin})),NA(),{predicted_margin}+{home_spread_live})"
        )

        away_abbr = addr(excel_row, "away_abbr")
        home_abbr = addr(excel_row, "home_abbr")

        home_spread_odds = addr(excel_row, "home_spread_odds_live")
        away_spread_odds = addr(excel_row, "away_spread_odds_live")
        away_spread_live = addr(excel_row, "away_spread_live")

        # Estimate sigma from p10/p90 when available.
        margin_p10_cell = addr(excel_row, "predicted_margin_p10")
        margin_p50_cell = addr(excel_row, "predicted_margin_p50")
        margin_p90_cell = addr(excel_row, "predicted_margin_p90")

        spread_mu = addr(excel_row, "spread_mu_margin")
        spread_sigma = addr(excel_row, "spread_sigma_margin")
        spread_p_home = addr(excel_row, "spread_p_home_cover")
        spread_p_away = addr(excel_row, "spread_p_away_cover")
        spread_implied_home = addr(excel_row, "spread_implied_home_prob")
        spread_implied_away = addr(excel_row, "spread_implied_away_prob")
        spread_edge_home = addr(excel_row, "spread_edge_home")
        spread_edge_away = addr(excel_row, "spread_edge_away")

        ws[spread_mu].value = (
            f"=IF(ISBLANK({margin_p50_cell}),{predicted_margin},{margin_p50_cell})"
        )
        ws[spread_sigma].value = (
            f"=IF(OR(ISBLANK({margin_p10_cell}),ISBLANK({margin_p90_cell})),{DEFAULT_SIGMA_MARGIN},"
            f"({margin_p90_cell}-{margin_p10_cell})/({P10_P90_TO_SIGMA_DENOM}))"
        )

        # P(home covers) = P(margin > -home_spread)
        ws[spread_p_home].value = (
            f"=IF(OR(ISNA({spread_sigma}),{spread_sigma}<=0,ISBLANK({home_spread_live})),NA(),"
            f"1-NORMSDIST(((-{home_spread_live})-({spread_mu}))/({spread_sigma})))"
        )
        # P(away covers) = P(margin < away_spread)
        ws[spread_p_away].value = (
            f"=IF(OR(ISNA({spread_sigma}),{spread_sigma}<=0,ISBLANK({away_spread_live})),NA(),"
            f"NORMSDIST((({away_spread_live})-({spread_mu}))/({spread_sigma})))"
        )

        # Break-even probs from spread odds (vig included)
        ws[spread_implied_home].value = (
            f"=IF(OR(ISBLANK({home_spread_odds}),{home_spread_odds}=0),NA(),"
            f"IF({home_spread_odds}>0,100/({home_spread_odds}+100),-({home_spread_odds})/(-({home_spread_odds})+100)))"
        )
        ws[spread_implied_away].value = (
            f"=IF(OR(ISBLANK({away_spread_odds}),{away_spread_odds}=0),NA(),"
            f"IF({away_spread_odds}>0,100/({away_spread_odds}+100),-({away_spread_odds})/(-({away_spread_odds})+100)))"
        )

        ws[spread_edge_home].value = f"={spread_p_home}-{spread_implied_home}"
        ws[spread_edge_away].value = f"={spread_p_away}-{spread_implied_away}"

        spread_value_side = addr(excel_row, "spread_value_side")
        ws[spread_value_side].value = (
            f"=IF({spread_edge_home}>={spread_edge_away},{home_abbr},{away_abbr})"
        )

        spread_edge_prob = addr(excel_row, "spread_edge_prob")
        ws[spread_edge_prob].value = f"=MAX(0,{spread_edge_home},{spread_edge_away})"

        spread_action = addr(excel_row, "spread_action")
        ws[spread_action].value = (
            f'=IF({spread_edge_prob}<=0,"PASS",IF({spread_edge_prob}<0.02,"PASS",IF({spread_edge_prob}<0.04,"LEAN",IF({spread_edge_prob}<0.07,"SMALL",IF({spread_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        spread_conf = addr(excel_row, "spread_confidence_1_10")
        ws[spread_conf].value = (
            f"=IF({spread_edge_prob}<0.01,1,IF({spread_edge_prob}<0.02,2,IF({spread_edge_prob}<0.03,3,IF({spread_edge_prob}<0.04,4,IF({spread_edge_prob}<0.05,5,IF({spread_edge_prob}<0.06,6,IF({spread_edge_prob}<0.07,7,IF({spread_edge_prob}<0.08,8,IF({spread_edge_prob}<0.10,9,10)))))))))"
        )

        spread_ev = addr(excel_row, "spread_ev")
        home_spread_profit = (
            f"IF({home_spread_odds}>0,{home_spread_odds}/100,100/ABS({home_spread_odds}))"
        )
        away_spread_profit = (
            f"IF({away_spread_odds}>0,{away_spread_odds}/100,100/ABS({away_spread_odds}))"
        )
        ev_spread_home = f"({spread_p_home})*({home_spread_profit})-(1-({spread_p_home}))"
        ev_spread_away = f"({spread_p_away})*({away_spread_profit})-(1-({spread_p_away}))"
        ws[spread_ev].value = (
            f"=IF({spread_value_side}={home_abbr},{ev_spread_home},{ev_spread_away})"
        )

        # --- Moneyline recommendation using live odds ---
        # Edge vs offered price (break-even probability): model_prob - implied_prob_raw
        edge_home_price = f"({model_home}-{market_home_raw})"
        edge_away_price = f"({model_away}-{market_away_raw})"
        moneyline_value_side = addr(excel_row, "money_value_side")
        ws[moneyline_value_side].value = (
            f"=IF({edge_home_price}>={edge_away_price},{home_abbr},{away_abbr})"
        )

        moneyline_edge_prob = addr(excel_row, "money_edge_prob")
        ws[moneyline_edge_prob].value = f"=MAX(0,{edge_home_price},{edge_away_price})"

        moneyline_action = addr(excel_row, "money_action")
        ws[moneyline_action].value = (
            f'=IF({moneyline_edge_prob}<=0,"PASS",IF({moneyline_edge_prob}<0.02,"PASS",IF({moneyline_edge_prob}<0.04,"LEAN",IF({moneyline_edge_prob}<0.07,"SMALL",IF({moneyline_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        moneyline_conf = addr(excel_row, "money_confidence_1_10")
        ws[moneyline_conf].value = (
            f"=IF({moneyline_edge_prob}<0.01,1,IF({moneyline_edge_prob}<0.02,2,IF({moneyline_edge_prob}<0.03,3,IF({moneyline_edge_prob}<0.04,4,IF({moneyline_edge_prob}<0.05,5,IF({moneyline_edge_prob}<0.06,6,IF({moneyline_edge_prob}<0.07,7,IF({moneyline_edge_prob}<0.08,8,IF({moneyline_edge_prob}<0.10,9,10)))))))))"
        )

        # EV per $1 stake (expected profit; ignores pushes)
        moneyline_ev = addr(excel_row, "moneyline_ev")
        home_profit = f"IF({home_ml}>0,{home_ml}/100,100/ABS({home_ml}))"
        away_profit = f"IF({away_ml}>0,{away_ml}/100,100/ABS({away_ml}))"
        ev_home = f"({model_home})*({home_profit})-(1-({model_home}))"
        ev_away = f"({model_away})*({away_profit})-(1-({model_away}))"
        ws[moneyline_ev].value = f"=IF({moneyline_value_side}={home_abbr},{ev_home},{ev_away})"

        # --- Total recommendation using live line + odds (approx prob from quantiles) ---
        total_live = addr(excel_row, "total_live")
        total_over_odds = addr(excel_row, "total_over_odds_live")
        total_under_odds = addr(excel_row, "total_under_odds_live")

        over_odds_used = total_over_odds
        under_odds_used = total_under_odds

        total_p10_cell = addr(excel_row, "predicted_total_p10")
        total_p50_cell = addr(excel_row, "predicted_total_p50")
        total_p90_cell = addr(excel_row, "predicted_total_p90")

        predicted_total_cell = addr(excel_row, "predicted_total")
        total_mu = addr(excel_row, "total_mu")
        total_sigma = addr(excel_row, "total_sigma")
        total_p_over = addr(excel_row, "total_p_over")
        total_p_under = addr(excel_row, "total_p_under")
        total_implied_over = addr(excel_row, "total_implied_over_prob")
        total_implied_under = addr(excel_row, "total_implied_under_prob")
        total_edge_over = addr(excel_row, "total_edge_over")
        total_edge_under = addr(excel_row, "total_edge_under")

        ws[total_mu].value = (
            f"=IF(ISBLANK({total_p50_cell}),{predicted_total_cell},{total_p50_cell})"
        )
        ws[total_sigma].value = (
            f"=IF(OR(ISBLANK({total_p10_cell}),ISBLANK({total_p90_cell})),{DEFAULT_SIGMA_TOTAL},"
            f"({total_p90_cell}-{total_p10_cell})/({P10_P90_TO_SIGMA_DENOM}))"
        )

        ws[total_p_over].value = (
            f"=IF(OR(ISNA({total_sigma}),{total_sigma}<=0,ISBLANK({total_live})),NA(),"
            f"1-NORMSDIST((({total_live})-({total_mu}))/({total_sigma})))"
        )
        ws[total_p_under].value = f"=IF(ISNA({total_p_over}),NA(),1-({total_p_over}))"

        ws[total_implied_over].value = (
            f"=IF(OR(ISBLANK({over_odds_used}),{over_odds_used}=0),NA(),"
            f"IF({over_odds_used}>0,100/({over_odds_used}+100),-({over_odds_used})/(-({over_odds_used})+100)))"
        )
        ws[total_implied_under].value = (
            f"=IF(OR(ISBLANK({under_odds_used}),{under_odds_used}=0),NA(),"
            f"IF({under_odds_used}>0,100/({under_odds_used}+100),-({under_odds_used})/(-({under_odds_used})+100)))"
        )

        ws[total_edge_over].value = f"={total_p_over}-{total_implied_over}"
        ws[total_edge_under].value = f"={total_p_under}-{total_implied_under}"

        total_value_side = addr(excel_row, "total_value_side")
        ws[total_value_side].value = f'=IF({total_edge_over}>={total_edge_under},"OVER","UNDER")'

        total_edge_prob = addr(excel_row, "total_edge_prob")
        ws[total_edge_prob].value = f"=MAX(0,{total_edge_over},{total_edge_under})"

        total_action = addr(excel_row, "total_action")
        ws[total_action].value = (
            f'=IF({total_edge_prob}<=0,"PASS",IF({total_edge_prob}<0.02,"PASS",IF({total_edge_prob}<0.04,"LEAN",IF({total_edge_prob}<0.07,"SMALL",IF({total_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        total_conf = addr(excel_row, "total_confidence_1_10")
        ws[total_conf].value = (
            f"=IF({total_edge_prob}<0.01,1,IF({total_edge_prob}<0.02,2,IF({total_edge_prob}<0.03,3,IF({total_edge_prob}<0.04,4,IF({total_edge_prob}<0.05,5,IF({total_edge_prob}<0.06,6,IF({total_edge_prob}<0.07,7,IF({total_edge_prob}<0.08,8,IF({total_edge_prob}<0.10,9,10)))))))))"
        )

        total_ev = addr(excel_row, "total_ev")
        over_profit = f"IF({over_odds_used}>0,{over_odds_used}/100,100/ABS({over_odds_used}))"
        under_profit = f"IF({under_odds_used}>0,{under_odds_used}/100,100/ABS({under_odds_used}))"
        ev_over = f"({total_p_over})*({over_profit})-(1-({total_p_over}))"
        ev_under = f"({total_p_under})*({under_profit})-(1-({total_p_under}))"
        ws[total_ev].value = f'=IF({total_value_side}="OVER",{ev_over},{ev_under})'

        # --- Highlight model vs live input areas for usability ---
        # (Header row styled later; body fills applied per-cell for key inputs.)
        live_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        for col_name in (
            "away_spread_live",
            "away_spread_odds_live",
            "home_spread_live",
            "home_spread_odds_live",
            "away_moneyline_live",
            "home_moneyline_live",
            "total_live",
            "total_over_odds_live",
            "total_under_odds_live",
        ):
            j = col_index[col_name]
            ws.cell(row=excel_row, column=j + 1).fill = live_fill

    # Live sheet (in-game heuristic tab).
    ws_live = wb.create_sheet(title=cfg.live_sheet_name)

    live_columns = [
        # Identifiers (referenced from Bets)
        "season",
        "week",
        "date",
        "game_id",
        "away_abbr",
        "home_abbr",
        # User inputs
        "quarter",
        "minutes_remaining_in_quarter",
        "minutes_remaining",
        "away_score_live",
        "home_score_live",
        # Derived state
        "current_margin",
        "current_total",
        "w_time",
        # Pregame priors (from model)
        "pregame_mu_margin",
        "pregame_sigma_margin",
        "live_mu_margin",
        "live_sigma_margin",
        "live_home_win_prob",
        "live_away_win_prob",
        # In-play lines/odds (user editable; initialized to pregame values)
        "home_spread_live",
        "away_spread_live",
        "home_spread_odds_live",
        "away_spread_odds_live",
        "home_moneyline_live",
        "away_moneyline_live",
        "total_live",
        "total_over_odds_live",
        "total_under_odds_live",
        # Live spread probabilities + recommendations
        "live_p_home_cover",
        "live_p_away_cover",
        "live_spread_value_side",
        "live_spread_edge_prob",
        "live_spread_action",
        "live_spread_confidence_1_10",
        "live_spread_ev",
        # Live moneyline recommendations
        "live_money_value_side",
        "live_money_edge_prob",
        "live_money_action",
        "live_money_confidence_1_10",
        "live_moneyline_ev",
        # Live total probabilities + recommendations
        "pregame_mu_total",
        "pregame_sigma_total",
        "live_mu_total",
        "live_sigma_total",
        "live_p_over",
        "live_p_under",
        "live_total_value_side",
        "live_total_edge_prob",
        "live_total_action",
        "live_total_confidence_1_10",
        "live_total_ev",
        # Hidden helper columns to keep recommendation formulas simple.
        "live_spread_implied_home_prob",
        "live_spread_implied_away_prob",
        "live_spread_edge_home",
        "live_spread_edge_away",
        "live_total_implied_over_prob",
        "live_total_implied_under_prob",
        "live_total_edge_over",
        "live_total_edge_under",
    ]

    ws_live.append(live_columns)
    for cell in ws_live[1]:
        cell.font = Font(bold=True)

    live_col_index = {name: j for j, name in enumerate(live_columns)}

    # Heuristic constants (place to the right of the main table).
    const_label_col = col_letter(len(live_columns))
    const_value_col = col_letter(len(live_columns) + 1)
    ws_live[f"{const_label_col}1"].value = "DEFAULT_SIGMA_MARGIN"
    ws_live[f"{const_value_col}1"].value = DEFAULT_SIGMA_MARGIN
    ws_live[f"{const_label_col}2"].value = "DEFAULT_SIGMA_TOTAL"
    ws_live[f"{const_value_col}2"].value = DEFAULT_SIGMA_TOTAL

    def live_addr(row: int, col_name: str) -> str:
        return f"{col_letter(live_col_index[col_name])}{row}"

    sheet_bets = f"'{cfg.sheet_name}'"

    def bets_ref(bets_row: int, col_name: str) -> str:
        return f"={sheet_bets}!{addr(bets_row, col_name)}"

    # Input highlighting on Live tab.
    live_input_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    for i in range(len(records)):
        bets_row = r(i)
        live_row = r(i)
        ws_live.append([None] * len(live_columns))

        # Identifiers
        for name in ("season", "week", "date", "game_id", "away_abbr", "home_abbr"):
            ws_live[live_addr(live_row, name)].value = bets_ref(bets_row, name)

        # User inputs (defaults for quick live entry)
        ws_live[live_addr(live_row, "quarter")].value = 1
        ws_live[live_addr(live_row, "minutes_remaining_in_quarter")].value = 15
        ws_live[live_addr(live_row, "away_score_live")].value = 0
        ws_live[live_addr(live_row, "home_score_live")].value = 0
        for name in (
            "quarter",
            "minutes_remaining_in_quarter",
            "away_score_live",
            "home_score_live",
        ):
            ws_live[live_addr(live_row, name)].fill = live_input_fill

        quarter = live_addr(live_row, "quarter")
        minutes_in_quarter = live_addr(live_row, "minutes_remaining_in_quarter")
        minutes_remaining = live_addr(live_row, "minutes_remaining")
        ws_live[minutes_remaining].value = (
            f"=IF(OR(ISBLANK({quarter}),ISBLANK({minutes_in_quarter})),NA(),"
            f"MAX(0,MIN(60,(4-({quarter}))*15+({minutes_in_quarter}))))"
        )

        away_score_live = live_addr(live_row, "away_score_live")
        home_score_live = live_addr(live_row, "home_score_live")

        current_margin = live_addr(live_row, "current_margin")
        current_total = live_addr(live_row, "current_total")
        ws_live[current_margin].value = (
            f"=IF(OR(ISBLANK({home_score_live}),ISBLANK({away_score_live})),NA(),{home_score_live}-{away_score_live})"
        )
        ws_live[current_total].value = (
            f"=IF(OR(ISBLANK({home_score_live}),ISBLANK({away_score_live})),NA(),{home_score_live}+{away_score_live})"
        )

        w_time = live_addr(live_row, "w_time")
        ws_live[w_time].value = (
            f"=IF(ISBLANK({minutes_remaining}),NA(),MAX(0,MIN(1,{minutes_remaining}/60)))"
        )

        # Pregame margin mean/sigma
        pre_mu_margin = live_addr(live_row, "pregame_mu_margin")
        pre_sigma_margin = live_addr(live_row, "pregame_sigma_margin")
        bets_margin = f"{sheet_bets}!{addr(bets_row, 'predicted_margin')}"
        bets_margin_p10 = f"{sheet_bets}!{addr(bets_row, 'predicted_margin_p10')}"
        bets_margin_p50 = f"{sheet_bets}!{addr(bets_row, 'predicted_margin_p50')}"
        bets_margin_p90 = f"{sheet_bets}!{addr(bets_row, 'predicted_margin_p90')}"

        ws_live[pre_mu_margin].value = (
            f"=IF(ISBLANK({bets_margin_p50}),{bets_margin},{bets_margin_p50})"
        )
        ws_live[pre_sigma_margin].value = (
            "=IF(OR(ISBLANK({p10}),ISBLANK({p90})),${default_sigma},({p90}-{p10})/({denom}))"
        ).format(
            p10=bets_margin_p10,
            p90=bets_margin_p90,
            default_sigma=f"{const_value_col}$1",
            denom=str(P10_P90_TO_SIGMA_DENOM),
        )

        live_mu_margin = live_addr(live_row, "live_mu_margin")
        live_sigma_margin = live_addr(live_row, "live_sigma_margin")
        ws_live[live_mu_margin].value = (
            f"=IF(OR(ISNA({w_time}),ISNA({current_margin})),NA(),({w_time})*({pre_mu_margin})+(1-({w_time}))*({current_margin}))"
        )
        ws_live[live_sigma_margin].value = (
            f"=IF(OR(ISNA({w_time}),ISBLANK({pre_sigma_margin})),NA(),({pre_sigma_margin})*SQRT({w_time}))"
        )

        live_home_win = live_addr(live_row, "live_home_win_prob")
        live_away_win = live_addr(live_row, "live_away_win_prob")
        ws_live[live_home_win].value = (
            f"=IF(OR(ISNA({live_sigma_margin}),{live_sigma_margin}<=0),NA(),1-NORMSDIST((0-({live_mu_margin}))/({live_sigma_margin})))"
        )
        ws_live[live_away_win].value = f"=IF(ISNA({live_home_win}),NA(),1-({live_home_win}))"

        # Initialize in-play lines/odds to the pregame values from the input data,
        # but keep them editable (no cross-sheet formula dependency).
        rec = records[i]
        for name in (
            "home_spread_live",
            "away_spread_live",
            "home_spread_odds_live",
            "away_spread_odds_live",
            "home_moneyline_live",
            "away_moneyline_live",
            "total_live",
            "total_over_odds_live",
            "total_under_odds_live",
        ):
            cell = ws_live[live_addr(live_row, name)]
            val = rec.get(name)
            if val is pd.NA or (hasattr(pd, "isna") and pd.isna(val)):
                val = None

            if val is None and (name.endswith("_odds_live") or name.endswith("_moneyline_live")):
                val = DEFAULT_SPREAD_TOTAL_ODDS
            cell.value = val
            cell.fill = live_input_fill

        # Live spread probabilities
        home_spread_live = live_addr(live_row, "home_spread_live")
        away_spread_live = live_addr(live_row, "away_spread_live")
        p_home_cover = live_addr(live_row, "live_p_home_cover")
        p_away_cover = live_addr(live_row, "live_p_away_cover")
        ws_live[p_home_cover].value = (
            f"=IF(OR(ISNA({live_sigma_margin}),{live_sigma_margin}<=0,ISBLANK({home_spread_live})),NA(),"
            f"1-NORMSDIST(((-{home_spread_live})-({live_mu_margin}))/({live_sigma_margin})))"
        )
        ws_live[p_away_cover].value = (
            f"=IF(OR(ISNA({live_sigma_margin}),{live_sigma_margin}<=0,ISBLANK({away_spread_live})),NA(),"
            f"NORMSDIST((({away_spread_live})-({live_mu_margin}))/({live_sigma_margin})))"
        )

        # Spread edge/action/EV vs odds
        home_spread_odds = live_addr(live_row, "home_spread_odds_live")
        away_spread_odds = live_addr(live_row, "away_spread_odds_live")
        implied_home_spread = (
            f"IF(OR(ISBLANK({home_spread_odds}),{home_spread_odds}=0),NA(),"
            f"IF({home_spread_odds}>0,100/({home_spread_odds}+100),-({home_spread_odds})/(-({home_spread_odds})+100)))"
        )
        implied_away_spread = (
            f"IF(OR(ISBLANK({away_spread_odds}),{away_spread_odds}=0),NA(),"
            f"IF({away_spread_odds}>0,100/({away_spread_odds}+100),-({away_spread_odds})/(-({away_spread_odds})+100)))"
        )

        spread_implied_home = live_addr(live_row, "live_spread_implied_home_prob")
        spread_implied_away = live_addr(live_row, "live_spread_implied_away_prob")
        spread_edge_home = live_addr(live_row, "live_spread_edge_home")
        spread_edge_away = live_addr(live_row, "live_spread_edge_away")

        ws_live[spread_implied_home].value = f"={implied_home_spread}"
        ws_live[spread_implied_away].value = f"={implied_away_spread}"
        ws_live[spread_edge_home].value = f"={p_home_cover}-{spread_implied_home}"
        ws_live[spread_edge_away].value = f"={p_away_cover}-{spread_implied_away}"

        away_abbr = live_addr(live_row, "away_abbr")
        home_abbr = live_addr(live_row, "home_abbr")
        spread_value_side = live_addr(live_row, "live_spread_value_side")
        spread_edge_prob = live_addr(live_row, "live_spread_edge_prob")
        spread_action = live_addr(live_row, "live_spread_action")
        spread_conf = live_addr(live_row, "live_spread_confidence_1_10")
        spread_ev = live_addr(live_row, "live_spread_ev")

        ws_live[spread_value_side].value = (
            f"=IF({spread_edge_home}>={spread_edge_away},{home_abbr},{away_abbr})"
        )
        ws_live[spread_edge_prob].value = f"=MAX(0,{spread_edge_home},{spread_edge_away})"
        ws_live[spread_action].value = (
            f'=IF({spread_edge_prob}<=0,"PASS",IF({spread_edge_prob}<0.02,"PASS",IF({spread_edge_prob}<0.04,"LEAN",IF({spread_edge_prob}<0.07,"SMALL",IF({spread_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )
        ws_live[spread_conf].value = (
            f"=IF({spread_edge_prob}<0.01,1,IF({spread_edge_prob}<0.02,2,IF({spread_edge_prob}<0.03,3,IF({spread_edge_prob}<0.04,4,IF({spread_edge_prob}<0.05,5,IF({spread_edge_prob}<0.06,6,IF({spread_edge_prob}<0.07,7,IF({spread_edge_prob}<0.08,8,IF({spread_edge_prob}<0.10,9,10)))))))))"
        )
        home_spread_profit = (
            f"IF({home_spread_odds}>0,{home_spread_odds}/100,100/ABS({home_spread_odds}))"
        )
        away_spread_profit = (
            f"IF({away_spread_odds}>0,{away_spread_odds}/100,100/ABS({away_spread_odds}))"
        )
        ev_spread_home = f"({p_home_cover})*({home_spread_profit})-(1-({p_home_cover}))"
        ev_spread_away = f"({p_away_cover})*({away_spread_profit})-(1-({p_away_cover}))"
        ws_live[spread_ev].value = (
            f"=IF({spread_value_side}={home_abbr},{ev_spread_home},{ev_spread_away})"
        )

        # Live moneyline recommendations
        home_ml = live_addr(live_row, "home_moneyline_live")
        away_ml = live_addr(live_row, "away_moneyline_live")
        implied_home_ml = (
            f"IF(OR(ISBLANK({home_ml}),{home_ml}=0),NA(),"
            f"IF({home_ml}>0,100/({home_ml}+100),-({home_ml})/(-({home_ml})+100)))"
        )
        implied_away_ml = (
            f"IF(OR(ISBLANK({away_ml}),{away_ml}=0),NA(),"
            f"IF({away_ml}>0,100/({away_ml}+100),-({away_ml})/(-({away_ml})+100)))"
        )
        edge_home_ml = f"({live_home_win})-({implied_home_ml})"
        edge_away_ml = f"({live_away_win})-({implied_away_ml})"
        money_value_side = live_addr(live_row, "live_money_value_side")
        money_edge_prob = live_addr(live_row, "live_money_edge_prob")
        money_action = live_addr(live_row, "live_money_action")
        money_conf = live_addr(live_row, "live_money_confidence_1_10")
        money_ev = live_addr(live_row, "live_moneyline_ev")

        ws_live[money_value_side].value = (
            f"=IF({edge_home_ml}>={edge_away_ml},{home_abbr},{away_abbr})"
        )
        ws_live[money_edge_prob].value = f"=MAX(0,{edge_home_ml},{edge_away_ml})"
        ws_live[money_action].value = (
            f'=IF({money_edge_prob}<=0,"PASS",IF({money_edge_prob}<0.02,"PASS",IF({money_edge_prob}<0.04,"LEAN",IF({money_edge_prob}<0.07,"SMALL",IF({money_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )
        ws_live[money_conf].value = (
            f"=IF({money_edge_prob}<0.01,1,IF({money_edge_prob}<0.02,2,IF({money_edge_prob}<0.03,3,IF({money_edge_prob}<0.04,4,IF({money_edge_prob}<0.05,5,IF({money_edge_prob}<0.06,6,IF({money_edge_prob}<0.07,7,IF({money_edge_prob}<0.08,8,IF({money_edge_prob}<0.10,9,10)))))))))"
        )
        home_ml_profit = f"IF({home_ml}>0,{home_ml}/100,100/ABS({home_ml}))"
        away_ml_profit = f"IF({away_ml}>0,{away_ml}/100,100/ABS({away_ml}))"
        ev_ml_home = f"({live_home_win})*({home_ml_profit})-(1-({live_home_win}))"
        ev_ml_away = f"({live_away_win})*({away_ml_profit})-(1-({live_away_win}))"
        ws_live[money_ev].value = f"=IF({money_value_side}={home_abbr},{ev_ml_home},{ev_ml_away})"

        # Pregame total mean/sigma + live adjusted total
        pre_mu_total = live_addr(live_row, "pregame_mu_total")
        pre_sigma_total = live_addr(live_row, "pregame_sigma_total")
        bets_total = f"{sheet_bets}!{addr(bets_row, 'predicted_total')}"
        bets_total_p10 = f"{sheet_bets}!{addr(bets_row, 'predicted_total_p10')}"
        bets_total_p50 = f"{sheet_bets}!{addr(bets_row, 'predicted_total_p50')}"
        bets_total_p90 = f"{sheet_bets}!{addr(bets_row, 'predicted_total_p90')}"
        ws_live[pre_mu_total].value = (
            f"=IF(ISBLANK({bets_total_p50}),{bets_total},{bets_total_p50})"
        )
        ws_live[pre_sigma_total].value = (
            "=IF(OR(ISBLANK({p10}),ISBLANK({p90})),${default_sigma},({p90}-{p10})/({denom}))"
        ).format(
            p10=bets_total_p10,
            p90=bets_total_p90,
            default_sigma=f"{const_value_col}$2",
            denom=str(P10_P90_TO_SIGMA_DENOM),
        )

        live_mu_total = live_addr(live_row, "live_mu_total")
        live_sigma_total = live_addr(live_row, "live_sigma_total")
        ws_live[live_mu_total].value = (
            f"=IF(OR(ISNA({w_time}),ISNA({current_total})),NA(),({w_time})*({pre_mu_total})+(1-({w_time}))*({current_total}))"
        )
        ws_live[live_sigma_total].value = (
            f"=IF(OR(ISNA({w_time}),ISBLANK({pre_sigma_total})),NA(),({pre_sigma_total})*SQRT({w_time}))"
        )

        total_live = live_addr(live_row, "total_live")
        total_over_odds = live_addr(live_row, "total_over_odds_live")
        total_under_odds = live_addr(live_row, "total_under_odds_live")
        over_odds_used = total_over_odds
        under_odds_used = total_under_odds

        p_over = live_addr(live_row, "live_p_over")
        p_under = live_addr(live_row, "live_p_under")
        ws_live[p_over].value = (
            f"=IF(OR(ISNA({live_sigma_total}),{live_sigma_total}<=0,ISBLANK({total_live})),NA(),"
            f"1-NORMSDIST((({total_live})-({live_mu_total}))/({live_sigma_total})))"
        )
        ws_live[p_under].value = f"=IF(ISNA({p_over}),NA(),1-({p_over}))"

        implied_over = (
            f"IF(OR(ISBLANK({over_odds_used}),{over_odds_used}=0),NA(),"
            f"IF({over_odds_used}>0,100/({over_odds_used}+100),-({over_odds_used})/(-({over_odds_used})+100)))"
        )
        implied_under = (
            f"IF(OR(ISBLANK({under_odds_used}),{under_odds_used}=0),NA(),"
            f"IF({under_odds_used}>0,100/({under_odds_used}+100),-({under_odds_used})/(-({under_odds_used})+100)))"
        )

        total_implied_over = live_addr(live_row, "live_total_implied_over_prob")
        total_implied_under = live_addr(live_row, "live_total_implied_under_prob")
        total_edge_over = live_addr(live_row, "live_total_edge_over")
        total_edge_under = live_addr(live_row, "live_total_edge_under")

        ws_live[total_implied_over].value = f"={implied_over}"
        ws_live[total_implied_under].value = f"={implied_under}"
        ws_live[total_edge_over].value = f"={p_over}-{total_implied_over}"
        ws_live[total_edge_under].value = f"={p_under}-{total_implied_under}"

        total_value_side = live_addr(live_row, "live_total_value_side")
        total_edge_prob = live_addr(live_row, "live_total_edge_prob")
        total_action = live_addr(live_row, "live_total_action")
        total_conf = live_addr(live_row, "live_total_confidence_1_10")
        total_ev = live_addr(live_row, "live_total_ev")

        ws_live[total_value_side].value = (
            f'=IF({total_edge_over}>={total_edge_under},"OVER","UNDER")'
        )
        ws_live[total_edge_prob].value = f"=MAX(0,{total_edge_over},{total_edge_under})"
        ws_live[total_action].value = (
            f'=IF({total_edge_prob}<=0,"PASS",IF({total_edge_prob}<0.02,"PASS",IF({total_edge_prob}<0.04,"LEAN",IF({total_edge_prob}<0.07,"SMALL",IF({total_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )
        ws_live[total_conf].value = (
            f"=IF({total_edge_prob}<0.01,1,IF({total_edge_prob}<0.02,2,IF({total_edge_prob}<0.03,3,IF({total_edge_prob}<0.04,4,IF({total_edge_prob}<0.05,5,IF({total_edge_prob}<0.06,6,IF({total_edge_prob}<0.07,7,IF({total_edge_prob}<0.08,8,IF({total_edge_prob}<0.10,9,10)))))))))"
        )
        over_profit = f"IF({over_odds_used}>0,{over_odds_used}/100,100/ABS({over_odds_used}))"
        under_profit = f"IF({under_odds_used}>0,{under_odds_used}/100,100/ABS({under_odds_used}))"
        ev_over = f"({p_over})*({over_profit})-(1-({p_over}))"
        ev_under = f"({p_under})*({under_profit})-(1-({p_under}))"
        ws_live[total_ev].value = f'=IF({total_value_side}="OVER",{ev_over},{ev_under})'

    ws_live.freeze_panes = "G2"
    ws_live.auto_filter.ref = f"A1:{col_letter(len(live_columns) - 1)}1"

    # Basic formatting.
    # Freeze header row and the identifier columns through home_abbr.
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = f"A1:{col_letter(len(columns) - 1)}1"

    # Light column sizing for readability (best-effort).
    width_map = {
        "date": 12,
        "game_id": 18,
        "away_abbr": 10,
        "home_abbr": 10,
        "money_action": 12,
        "spread_action": 12,
        "total_action": 12,
    }
    for name, w in width_map.items():
        if name in col_index:
            ws.column_dimensions[col_letter(col_index[name])].width = w

    # Hide helper columns (they exist only to keep other formulas short).
    for name in (
        "spread_mu_margin",
        "spread_sigma_margin",
        "spread_p_home_cover",
        "spread_p_away_cover",
        "spread_implied_home_prob",
        "spread_implied_away_prob",
        "spread_edge_home",
        "spread_edge_away",
        "total_mu",
        "total_sigma",
        "total_p_over",
        "total_p_under",
        "total_implied_over_prob",
        "total_implied_under_prob",
        "total_edge_over",
        "total_edge_under",
    ):
        if name in col_index:
            ws.column_dimensions[col_letter(col_index[name])].hidden = True

    for name in (
        "live_spread_implied_home_prob",
        "live_spread_implied_away_prob",
        "live_spread_edge_home",
        "live_spread_edge_away",
        "live_total_implied_over_prob",
        "live_total_implied_under_prob",
        "live_total_edge_over",
        "live_total_edge_under",
    ):
        if name in live_col_index:
            ws_live.column_dimensions[col_letter(live_col_index[name])].hidden = True

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    log.info("Wrote Excel template to %s", out_path)
    return out_path
