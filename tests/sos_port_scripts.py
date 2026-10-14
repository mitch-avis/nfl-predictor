"""Import the audit scripts under ``.agents/sos_port/`` so their tests can call them.

The scripts are checked-in records that run from the repository root and import their siblings
by name, so the directory goes on ``sys.path`` (at the end, so it shadows nothing) before the
import.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

SOS_PORT_DIR = Path(__file__).resolve().parents[1] / ".agents" / "sos_port"


def load_sos_port_script(name: str) -> ModuleType:
    """Return the module of the script ``.agents/sos_port/{name}.py``."""
    directory = str(SOS_PORT_DIR)
    if directory not in sys.path:
        sys.path.append(directory)
    return importlib.import_module(name)
