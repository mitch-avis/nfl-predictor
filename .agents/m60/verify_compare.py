"""Check that ``nfl-predictor compare`` reproduces the task 55.8 independent rescore exactly.

The independent review of the nine season-weighting arms
(``models/wf_m55_8_review/INDEPENDENT_REVIEW.md``) wrote every number it reports to
``models/wf_m55_8_review/independent_rescore.json`` with its own script, which shares no code with
the compare command. This script recomputes each of those numbers through
``nfl_predictor.reporting.run_comparison`` from the same fold checkpoints and requires exact
float equality: every arm's window metrics, and every paired contrast (single-seed, two-seed, the
seed floor and the ladder contrasts) in every window and column.

Run from the repository root:

    .venv/bin/python .agents/m60/verify_compare.py > .agents/m60/verify_compare_output.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nfl_predictor.reporting import run_comparison

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "models" / "wf_m55_8_review" / "independent_rescore.json"
ARMS = {
    "unw_s42": "wf_m55_8_2020_2025_unweighted",
    "hl4_s42": "wf_m55_8_2020_2025_half_life4",
    "hl8_s42": "wf_m55_8_2020_2025_half_life8",
    "hl16_s42": "wf_m55_8_2020_2025_half_life16",
    "hl32_s42": "wf_m55_8_2020_2025_half_life32",
    "unw_s7": "wf_m55_8_2020_2025_unweighted_seed7",
    "hl4_s7": "wf_m55_8_2020_2025_half_life4_seed7",
    "hl16_s7": "wf_m55_8_2020_2025_half_life16_seed7",
    "hl32_s7": "wf_m55_8_2020_2025_half_life32_seed7",
}
# The contrasts the review computed, as (candidate, reference) arm pairs, one pair per seed.
CONTRASTS: dict[str, list[tuple[str, str]]] = {
    "hl4_s42 - unw_s42": [("hl4_s42", "unw_s42")],
    "hl8_s42 - unw_s42": [("hl8_s42", "unw_s42")],
    "hl16_s42 - unw_s42": [("hl16_s42", "unw_s42")],
    "hl32_s42 - unw_s42": [("hl32_s42", "unw_s42")],
    "hl4_s7 - unw_s7": [("hl4_s7", "unw_s7")],
    "hl16_s7 - unw_s7": [("hl16_s7", "unw_s7")],
    "hl32_s7 - unw_s7": [("hl32_s7", "unw_s7")],
    "seed floor unw s7 - s42": [("unw_s7", "unw_s42")],
    "seed floor hl4 s7 - s42": [("hl4_s7", "hl4_s42")],
    "seed floor hl16 s7 - s42": [("hl16_s7", "hl16_s42")],
    "seed floor hl32 s7 - s42": [("hl32_s7", "hl32_s42")],
    "TWO-SEED hl4 - unw": [("hl4_s42", "unw_s42"), ("hl4_s7", "unw_s7")],
    "TWO-SEED hl16 - unw": [("hl16_s42", "unw_s42"), ("hl16_s7", "unw_s7")],
    "TWO-SEED hl32 - unw": [("hl32_s42", "unw_s42"), ("hl32_s7", "unw_s7")],
    "TWO-SEED hl16 - hl32": [("hl16_s42", "hl32_s42"), ("hl16_s7", "hl32_s7")],
    "ladder unw_s42 - hl4_s42": [("unw_s42", "hl4_s42")],
    "ladder hl8_s42 - hl4_s42": [("hl8_s42", "hl4_s42")],
    "ladder hl16_s42 - hl4_s42": [("hl16_s42", "hl4_s42")],
    "ladder hl32_s42 - hl4_s42": [("hl32_s42", "hl4_s42")],
}
METRIC_KEYS = (*run_comparison.LOSS_COLUMNS, "market_brier", "pool", "det_minus_market_brier")


def main() -> int:
    """Recompute every reviewed number and print each mismatch; exit 1 if any."""
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    runs = {
        arm: run_comparison.load_run(run_comparison.resolve_run(ROOT / "models" / run_dir))
        for arm, run_dir in ARMS.items()
    }
    checked = 0
    mismatches: list[str] = []

    def check(where: str, ours: object, theirs: object) -> None:
        """Record one comparison, exact for every float."""
        nonlocal checked
        checked += 1
        ours_list = list(ours) if isinstance(ours, tuple) else ours
        if ours_list != theirs:
            mismatches.append(f"{where}: compare {ours_list!r} != review {theirs!r}")

    for name, arm_pairs in CONTRASTS.items():
        report = run_comparison.compare_runs(
            [runs[candidate] for candidate, _ in arm_pairs],
            [runs[reference] for _, reference in arm_pairs],
        )
        for window, expected in review["contrasts"][name].items():
            got = report["windows"][window]
            for column, values in expected.items():
                check(f"{name} | {window} | {column}", got["contrast"][column], values)
            for arm in {arm for pair in arm_pairs for arm in pair}:
                label = runs[arm].run.label
                for key in METRIC_KEYS:
                    check(
                        f"{arm} | {window} | {key}",
                        got["runs"][label][key],
                        review["metrics"][arm][window][key],
                    )
    print(f"values compared: {checked}")
    print(f"contrasts: {len(CONTRASTS)}, arms: {len(ARMS)}, windows: 4")
    print(f"mismatches: {len(mismatches)}")
    for line in mismatches:
        print(line)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
