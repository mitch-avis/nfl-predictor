"""Tests for the Excel betting template generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_predictor.reporting.betting_excel import write_betting_template_xlsx


def test_write_betting_template_xlsx_creates_workbook(tmp_path: Path) -> None:
    """Writes a workbook with expected sheets and formulas."""

    df = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 19,
                "game_id": "2025_19_X_Y",
                "away_abbr": "X",
                "home_abbr": "Y",
                "home_win_prob": 0.62,
                "predicted_margin": 3.5,
                "predicted_total": 45.0,
                "home_moneyline": -135,
                "away_moneyline": 115,
                "home_spread": -2.5,
                "total_line": 44.5,
            }
        ]
    )

    out = tmp_path / "template.xlsx"
    write_betting_template_xlsx(predictions=df, out_path=out)
    assert out.exists()

    import openpyxl

    wb = openpyxl.load_workbook(out)
    assert "README" in wb.sheetnames
    assert "Bets" in wb.sheetnames

    ws = wb["Bets"]
    # Row 2, columns include market_home_prob_raw formula.
    headers = [c.value for c in ws[1]]
    col_idx = headers.index("market_home_prob_raw") + 1
    cell = ws.cell(row=2, column=col_idx)
    assert isinstance(cell.value, str)
    assert cell.value.startswith("=IF(")
