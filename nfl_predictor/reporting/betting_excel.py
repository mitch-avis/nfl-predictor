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
        from openpyxl.styles import Font
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
    ws_readme["A9"].value = "- Edges use no-vig probabilities derived from the two moneylines."
    ws_readme["A10"].value = (
        "- Action thresholds: PASS <2%, LEAN <4%, SMALL <7%, MEDIUM <10%, STRONG >=10%."
    )

    # Main sheet.
    ws = wb.create_sheet(title=cfg.sheet_name)

    # Column layout.
    columns = [
        "season",
        "week",
        "game_id",
        "away_abbr",
        "home_abbr",
        "model_home_prob",
        "model_away_prob",
        "predicted_margin",
        "predicted_total",
        "home_moneyline_model_input",
        "away_moneyline_model_input",
        "home_moneyline_live",
        "away_moneyline_live",
        "market_home_prob_raw",
        "market_away_prob_raw",
        "market_home_prob_novig",
        "market_away_prob_novig",
        "edge_home_prob",
        "edge_away_prob",
        "value_side",
        "edge_prob",
        "action",
        "confidence_1_10",
        "home_spread_live",
        "total_line_live",
        "spread_edge_points_home",
    ]

    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    # Build a display DataFrame with safe columns.
    df = predictions.copy()
    for c in ("season", "week", "game_id"):
        if c not in df.columns:
            df[c] = pd.NA

    df["model_home_prob"] = df["home_win_prob"].astype(float)
    df["model_away_prob"] = 1.0 - df["model_home_prob"]
    # Default live inputs to model inputs when present.
    df["home_moneyline_model_input"] = df.get("home_moneyline", pd.NA)
    df["away_moneyline_model_input"] = df.get("away_moneyline", pd.NA)
    df["home_moneyline_live"] = df.get("home_moneyline", pd.NA)
    df["away_moneyline_live"] = df.get("away_moneyline", pd.NA)
    df["home_spread_live"] = df.get("home_spread", pd.NA)
    df["total_line_live"] = df.get("total_line", pd.NA)

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

        # Value side (choose side with non-negative edge)
        away_abbr = addr(excel_row, "away_abbr")
        home_abbr = addr(excel_row, "home_abbr")
        value_side = addr(excel_row, "value_side")
        ws[value_side].value = f"=IF({edge_home}>=0,{home_abbr},{away_abbr})"

        # Edge magnitude
        edge_prob = addr(excel_row, "edge_prob")
        ws[edge_prob].value = f"=MAX(ABS({edge_home}),ABS({edge_away}))"

        # Action thresholds
        action = addr(excel_row, "action")
        ws[action].value = (
            f'=IF({edge_prob}<0.02,"PASS",IF({edge_prob}<0.04,"LEAN",IF({edge_prob}<0.07,"SMALL",IF({edge_prob}<0.10,"MEDIUM","STRONG"))))'
        )

        # Confidence 1..10 (mirrors betting_pipeline thresholds)
        conf = addr(excel_row, "confidence_1_10")
        ws[conf].value = (
            f"=IF({edge_prob}<0.01,1,IF({edge_prob}<0.02,2,IF({edge_prob}<0.03,3,IF({edge_prob}<0.04,4,IF({edge_prob}<0.05,5,IF({edge_prob}<0.06,6,IF({edge_prob}<0.07,7,IF({edge_prob}<0.08,8,IF({edge_prob}<0.10,9,10)))))))))"
        )

        # Spread edge points (model margin + home spread)
        spread_edge = addr(excel_row, "spread_edge_points_home")
        predicted_margin = addr(excel_row, "predicted_margin")
        home_spread_live = addr(excel_row, "home_spread_live")
        ws[spread_edge].value = (
            f"=IF(OR(ISBLANK({home_spread_live}),ISBLANK({predicted_margin})),NA(),{predicted_margin}+{home_spread_live})"
        )

    # Basic formatting.
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{col_letter(len(columns) - 1)}1"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    log.info("Wrote Excel template to %s", out_path)
    return out_path
