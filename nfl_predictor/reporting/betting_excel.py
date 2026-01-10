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


@dataclass(frozen=True)
class ExcelTemplateConfig:
    """Configuration for the generated workbook."""

    sheet_name: str = "Bets"
    instructions_sheet_name: str = "README"


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
        # Live inputs (user editable)
        "away_spread_live",
        "away_spread_odds_live",
        "home_spread_live",
        "home_spread_odds_live",
        "away_moneyline_live",
        "home_moneyline_live",
        "total_live",
        "total_live_odds",
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
        # Live betting recommendations (moneyline)
        "value_side",
        "edge_prob",
        "action",
        "confidence_1_10",
        "moneyline_ev",
        # Live betting recommendations (spread)
        "spread_value_side",
        "spread_edge_prob",
        "spread_action",
        "spread_ev",
        # Live betting recommendations (total)
        "total_value_side",
        "total_edge_prob",
        "total_action",
        "total_ev",
        # Optional diagnostic columns (quantiles)
        "predicted_margin_p10",
        "predicted_margin_p50",
        "predicted_margin_p90",
        "predicted_total_p10",
        "predicted_total_p50",
        "predicted_total_p90",
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

    # Live odds inputs (default blank).
    df["home_spread_odds_live"] = pd.NA
    df["away_spread_odds_live"] = pd.NA
    df["total_live_odds"] = pd.NA

    # Computed outputs (filled by formulas).
    for c in (
        "market_home_prob_raw",
        "market_away_prob_raw",
        "market_home_prob_novig",
        "market_away_prob_novig",
        "edge_home_prob",
        "edge_away_prob",
        "spread_edge_points_home",
        "value_side",
        "edge_prob",
        "action",
        "confidence_1_10",
        "moneyline_ev",
        "spread_value_side",
        "spread_edge_prob",
        "spread_action",
        "spread_ev",
        "total_value_side",
        "total_edge_prob",
        "total_action",
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

        # --- Moneyline recommendation using live odds ---
        # Edge vs offered price (break-even probability): model_prob - implied_prob_raw
        edge_home_price = f"({model_home}-{market_home_raw})"
        edge_away_price = f"({model_away}-{market_away_raw})"
        moneyline_value_side = addr(excel_row, "value_side")
        away_abbr = addr(excel_row, "away_abbr")
        home_abbr = addr(excel_row, "home_abbr")
        ws[moneyline_value_side].value = (
            f"=IF({edge_home_price}>={edge_away_price},{home_abbr},{away_abbr})"
        )

        moneyline_edge_prob = addr(excel_row, "edge_prob")
        ws[moneyline_edge_prob].value = f"=MAX(0,{edge_home_price},{edge_away_price})"

        moneyline_action = addr(excel_row, "action")
        ws[moneyline_action].value = (
            f'=IF({moneyline_edge_prob}<=0,"PASS",IF({moneyline_edge_prob}<0.02,"PASS",IF({moneyline_edge_prob}<0.04,"LEAN",IF({moneyline_edge_prob}<0.07,"SMALL",IF({moneyline_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        moneyline_conf = addr(excel_row, "confidence_1_10")
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

        # --- Spread recommendation using live line + odds (approx prob from quantiles) ---
        spread_edge_points_home = addr(excel_row, "spread_edge_points_home")
        predicted_margin = addr(excel_row, "predicted_margin")
        home_spread_live = addr(excel_row, "home_spread_live")
        ws[spread_edge_points_home].value = (
            f"=IF(OR(ISBLANK({home_spread_live}),ISBLANK({predicted_margin})),NA(),{predicted_margin}+{home_spread_live})"
        )

        home_spread_odds = addr(excel_row, "home_spread_odds_live")
        away_spread_odds = addr(excel_row, "away_spread_odds_live")
        away_spread_live = addr(excel_row, "away_spread_live")

        # Estimate sigma from p10/p90 when available.
        margin_p10_cell = addr(excel_row, "predicted_margin_p10")
        margin_p50_cell = addr(excel_row, "predicted_margin_p50")
        margin_p90_cell = addr(excel_row, "predicted_margin_p90")

        mu_margin = f"IF(ISBLANK({margin_p50_cell}),{predicted_margin},{margin_p50_cell})"
        sigma_margin = (
            f"IF(OR(ISBLANK({margin_p10_cell}),ISBLANK({margin_p90_cell})),NA(),"
            f"({margin_p90_cell}-{margin_p10_cell})/(2*1.281551565545))"
        )

        # P(home covers) = P(margin > -home_spread)
        p_home_cover = (
            f"IF(OR(ISNA({sigma_margin}),{sigma_margin}<=0,ISBLANK({home_spread_live})),NA(),"
            f"1-NORM.S.DIST(((-{home_spread_live})-({mu_margin}))/({sigma_margin}),TRUE))"
        )
        # P(away covers) = P(margin < away_spread)
        p_away_cover = (
            f"IF(OR(ISNA({sigma_margin}),{sigma_margin}<=0,ISBLANK({away_spread_live})),NA(),"
            f"NORM.S.DIST((({away_spread_live})-({mu_margin}))/({sigma_margin}),TRUE))"
        )

        # Break-even probs from spread odds
        implied_home_spread = (
            f"IF(OR(ISBLANK({home_spread_odds}),{home_spread_odds}=0),NA(),"
            f"IF({home_spread_odds}>0,100/({home_spread_odds}+100),-({home_spread_odds})/(-({home_spread_odds})+100)))"
        )
        implied_away_spread = (
            f"IF(OR(ISBLANK({away_spread_odds}),{away_spread_odds}=0),NA(),"
            f"IF({away_spread_odds}>0,100/({away_spread_odds}+100),-({away_spread_odds})/(-({away_spread_odds})+100)))"
        )

        edge_home_spread = f"({p_home_cover})-({implied_home_spread})"
        edge_away_spread = f"({p_away_cover})-({implied_away_spread})"

        spread_value_side = addr(excel_row, "spread_value_side")
        ws[spread_value_side].value = (
            f"=IF({edge_home_spread}>={edge_away_spread},{home_abbr},{away_abbr})"
        )

        spread_edge_prob = addr(excel_row, "spread_edge_prob")
        ws[spread_edge_prob].value = f"=MAX(0,{edge_home_spread},{edge_away_spread})"

        spread_action = addr(excel_row, "spread_action")
        ws[spread_action].value = (
            f'=IF({spread_edge_prob}<=0,"PASS",IF({spread_edge_prob}<0.02,"PASS",IF({spread_edge_prob}<0.04,"LEAN",IF({spread_edge_prob}<0.07,"SMALL",IF({spread_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        spread_ev = addr(excel_row, "spread_ev")
        home_spread_profit = (
            f"IF({home_spread_odds}>0,{home_spread_odds}/100,100/ABS({home_spread_odds}))"
        )
        away_spread_profit = (
            f"IF({away_spread_odds}>0,{away_spread_odds}/100,100/ABS({away_spread_odds}))"
        )
        ev_spread_home = f"({p_home_cover})*({home_spread_profit})-(1-({p_home_cover}))"
        ev_spread_away = f"({p_away_cover})*({away_spread_profit})-(1-({p_away_cover}))"
        ws[spread_ev].value = (
            f"=IF({spread_value_side}={home_abbr},{ev_spread_home},{ev_spread_away})"
        )

        # --- Total recommendation using live line + odds (approx prob from quantiles) ---
        total_live = addr(excel_row, "total_live")
        total_odds = addr(excel_row, "total_live_odds")

        total_p10_cell = addr(excel_row, "predicted_total_p10")
        total_p50_cell = addr(excel_row, "predicted_total_p50")
        total_p90_cell = addr(excel_row, "predicted_total_p90")

        predicted_total_cell = addr(excel_row, "predicted_total")
        mu_total = f"IF(ISBLANK({total_p50_cell}),{predicted_total_cell},{total_p50_cell})"
        sigma_total = (
            f"IF(OR(ISBLANK({total_p10_cell}),ISBLANK({total_p90_cell})),NA(),"
            f"({total_p90_cell}-{total_p10_cell})/(2*1.281551565545))"
        )

        p_over = (
            f"IF(OR(ISNA({sigma_total}),{sigma_total}<=0,ISBLANK({total_live})),NA(),"
            f"1-NORM.S.DIST((({total_live})-({mu_total}))/({sigma_total}),TRUE))"
        )
        p_under = f"IF(ISNA({p_over}),NA(),1-({p_over}))"

        implied_total = (
            f"IF(OR(ISBLANK({total_odds}),{total_odds}=0),NA(),"
            f"IF({total_odds}>0,100/({total_odds}+100),-({total_odds})/(-({total_odds})+100)))"
        )
        edge_over = f"({p_over})-({implied_total})"
        edge_under = f"({p_under})-({implied_total})"

        total_value_side = addr(excel_row, "total_value_side")
        ws[total_value_side].value = f'=IF({edge_over}>={edge_under},"OVER","UNDER")'

        total_edge_prob = addr(excel_row, "total_edge_prob")
        ws[total_edge_prob].value = f"=MAX(0,{edge_over},{edge_under})"

        total_action = addr(excel_row, "total_action")
        ws[total_action].value = (
            f'=IF({total_edge_prob}<=0,"PASS",IF({total_edge_prob}<0.02,"PASS",IF({total_edge_prob}<0.04,"LEAN",IF({total_edge_prob}<0.07,"SMALL",IF({total_edge_prob}<0.10,"MEDIUM","STRONG")))))'
        )

        total_ev = addr(excel_row, "total_ev")
        total_profit = f"IF({total_odds}>0,{total_odds}/100,100/ABS({total_odds}))"
        ev_over = f"({p_over})*({total_profit})-(1-({p_over}))"
        ev_under = f"({p_under})*({total_profit})-(1-({p_under}))"
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
            "total_live_odds",
        ):
            j = col_index[col_name]
            ws.cell(row=excel_row, column=j + 1).fill = live_fill

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
        "action": 12,
        "spread_action": 12,
        "total_action": 12,
    }
    for name, w in width_map.items():
        if name in col_index:
            ws.column_dimensions[col_letter(col_index[name])].width = w

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    log.info("Wrote Excel template to %s", out_path)
    return out_path
