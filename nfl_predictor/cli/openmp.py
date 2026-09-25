"""The OpenMP wait policy for commands that run walk-forward fits.

XGBoost's OpenMP threads spin while they wait for each other by default. When other work
shares the machine, a preempted thread stalls the rest, and a walk-forward week has been
measured at several times its idle duration. The passive policy sleeps instead, which costs
a little on an idle machine and avoids the stalls. OpenMP reads the variable once, when its
library loads, so this must run before XGBoost is imported. It changes scheduling only, never
results.
"""

from __future__ import annotations

import os

WAIT_POLICY_VARIABLE = "OMP_WAIT_POLICY"


def prefer_passive_wait_policy() -> str:
    """Set ``OMP_WAIT_POLICY=PASSIVE`` unless the environment already sets it; return the value.

    An operator's value is kept, so ``OMP_WAIT_POLICY=`` (empty) keeps the library default on a
    dedicated machine.
    """
    return os.environ.setdefault(WAIT_POLICY_VARIABLE, "PASSIVE")
