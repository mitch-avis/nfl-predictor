"""Compare two walk-forward runs game by game, rescored from their fold checkpoints.

``nfl-predictor compare --candidate <run> --reference <run>`` rescores both runs' fold
checkpoints and reports, for week 1, week 2, weeks 3-18 and all weeks, each run's deterministic
Brier, log loss, pick accuracy, margin and total MAE, confidence-pool points and market Brier,
and the paired candidate-minus-reference differences with bootstrap intervals. A run is a run
directory with a ``metadata.json`` (which names its checkpoint directory and adds provenance and
the configuration differences) or a checkpoint directory under ``models/wf_checkpoints/``.

Give ``--candidate`` and ``--reference`` once per seed, in the same seed order, to combine seeds:
the per-game differences are averaged over the seed pairs before bootstrapping. The definitions
are in ``nfl_predictor.reporting.run_comparison``.

Example:
    nfl-predictor compare \
      --candidate models/wf_m55_8_2020_2025_half_life16 \
      --candidate models/wf_m55_8_2020_2025_half_life16_seed7 \
      --reference models/wf_m55_8_2020_2025_unweighted \
      --reference models/wf_m55_8_2020_2025_unweighted_seed7 \
      --out-json models/wf_m55_8_review/compare_hl16.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nfl_predictor.reporting import run_comparison
from nfl_predictor.utils.logger import log


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the compare command's options."""
    parser = argparse.ArgumentParser(
        description="Paired comparison of walk-forward runs from their fold checkpoints."
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="Candidate run directory or checkpoint directory; repeat once per seed.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        help="Reference run directory or checkpoint directory; repeat once per seed, in order.",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=run_comparison.DEFAULT_RESAMPLES,
        help="Bootstrap resamples (default: %(default)s).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=run_comparison.DEFAULT_BOOTSTRAP_SEED,
        help="Seed of the bootstrap's random generator (default: %(default)s).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Also write the full comparison, at full precision, to this JSON file.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Also write the printed tables to this Markdown file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison; return 0, or 2 when the runs cannot be compared."""
    args = _parse_args(argv)
    try:
        candidates = [
            run_comparison.load_run(run_comparison.resolve_run(path)) for path in args.candidate
        ]
        references = [
            run_comparison.load_run(run_comparison.resolve_run(path)) for path in args.reference
        ]
        report = run_comparison.compare_runs(
            candidates, references, resamples=args.resamples, seed=args.bootstrap_seed
        )
    except ValueError as error:
        log.error("compare: %s", error)
        return 2
    lines = run_comparison.format_report(report)
    for line in lines:
        log.info("%s", line)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        log.info("Wrote %s", args.out_json)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Wrote %s", args.out_md)
    return 0
