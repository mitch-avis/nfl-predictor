"""Count exact coin flips in reviewed walk-forward runs and rescore picks under both tie rules.

Run from the repository root, naming run directories (read only):

    .venv/bin/python .agents/sos_port/decisions/compare_tie_check.py \
        /home/mitch/workspace/nfl-predictor/models/step4_reference/adopted_seed42 ...

For each run it reads the fold checkpoints and prints the games, how many deterministic and
market probabilities are exactly 0.5 or missing, and, over all weeks, the pick accuracy and pool
points ``compare`` reports for the run under the old rule (``p > 0.5`` home, ``p < 0.5`` away,
an exact 0.5 never right) and the new one (``metrics.picks_home``: ``p >= 0.5`` home). It also
scores the run's market probability as a pick, the way the step-4 review scored the betting lines
with ``compare``'s convention. It writes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nfl_predictor.ml.metrics import confidence_ranks, picks_home
from nfl_predictor.reporting import run_comparison

EVEN = 0.5


def _predictions(run_dir: Path) -> pd.DataFrame:
    """Return a run's fold prediction rows, sorted by game id."""
    run = run_comparison.resolve_run(run_dir)
    frames = [
        joblib.load(path)["predictions"] for path in sorted(run.checkpoint_dir.glob("fold_*.joblib"))
    ]
    return pd.concat(frames, ignore_index=True).sort_values("game_id").reset_index(drop=True)


def _correct(p: np.ndarray, margin: np.ndarray, *, new_rule: bool) -> np.ndarray:
    """Return 1.0 for a right pick under the new or the old tie rule, else 0.0."""
    if new_rule:
        home = picks_home(p)
        return ((home & (margin > 0)) | (~home & (margin < 0))).astype(float)
    return (((p > EVEN) & (margin > 0)) | ((p < EVEN) & (margin < 0))).astype(float)


def _pool(frame: pd.DataFrame, p: np.ndarray, correct: np.ndarray) -> float:
    """Return the all-weeks pool points, ranked as ``compare`` ranks them."""
    ranks = confidence_ranks(
        p,
        tiebreaker=frame["game_id"].to_numpy(),
        groups=(frame["season"].to_numpy(), frame["week"].to_numpy()),
    )
    return float((ranks * correct).sum())


def main() -> None:
    """Print one block per run."""
    for name in sys.argv[1:]:
        run_dir = Path(name)
        frame = _predictions(run_dir)
        margin = frame["actual_margin"].to_numpy(float)
        print(f"== {run_dir.parent.name}/{run_dir.name}: {len(frame)} games")
        for column, label in (
            ("deterministic_home_win_prob", "model"),
            ("market_home_win_prob", "market"),
        ):
            p = frame[column].to_numpy(float)
            old = _correct(p, margin, new_rule=False)
            new = _correct(p, margin, new_rule=True)
            print(
                f"  {label}: exactly 0.5 {int((p == EVEN).sum())}, missing "
                f"{int(np.isnan(p).sum())}; pick accuracy old {old.mean():.4f} new {new.mean():.4f}"
                f"; pool points old {_pool(frame, p, old):.1f} new {_pool(frame, p, new):.1f}"
            )
        scored = run_comparison.per_game_scores(frame)
        print(
            f"  compare as it now scores the model: pick accuracy "
            f"{scored['correct'].mean():.4f}, pool points {scored['pool'].sum():.1f}"
        )


if __name__ == "__main__":
    main()
