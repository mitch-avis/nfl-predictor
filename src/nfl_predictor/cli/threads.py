"""One Polars thread for every command unless the caller sets ``POLARS_MAX_THREADS``.

The ETL and the readers build thousands of small frames (per season, week and team), where a
Polars pool sized to every core spends its time waking threads instead of working. Polars reads
the variable once, when its pool starts, so this must run before a command module imports Polars;
the front door calls it before resolving any command. An operator's value is kept, so
``POLARS_MAX_THREADS=24`` restores a full pool. It changes scheduling only; XGBoost and the BLAS
libraries keep their own thread settings.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import MutableMapping

POLARS_THREAD_VARIABLE = "POLARS_MAX_THREADS"


def prefer_one_polars_thread(environ: MutableMapping[str, str] | None = None) -> str:
    """Set ``POLARS_MAX_THREADS=1`` unless ``environ`` already sets it; return the value.

    Args:
        environ: The environment to update; the process environment when omitted.

    Returns:
        The value in effect after the call.
    """
    target = os.environ if environ is None else environ
    return target.setdefault(POLARS_THREAD_VARIABLE, "1")
