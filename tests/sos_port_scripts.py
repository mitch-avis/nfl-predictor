"""Load the scripts under ``.agents/sos_port`` as modules, for their unit tests.

The scripts import their siblings by bare name (``import elo_eval``), as they do when run from
their own directory, so the loader appends that directory to ``sys.path`` before importing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".agents" / "sos_port"


def load_script(name: str) -> ModuleType:
    """Return the script ``name`` (its file name without ``.py``) as an imported module."""
    directory = str(SCRIPTS_DIR)
    if directory not in sys.path:
        sys.path.append(directory)
    return importlib.import_module(name)
