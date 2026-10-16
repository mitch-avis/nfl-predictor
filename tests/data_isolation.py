"""Keep the test suite out of the checkout's ``data/`` and ``models/`` and off the network.

``install`` runs once per test process, from ``tests/conftest.py`` before any test module imports
the package:

- It points ``constants.DATA_PATH`` and the two cache directories under it at a temporary
  directory for the whole session, so every path a module copies from them at import time (the
  lines refresh's and the week builder's data directory, the default strength-snapshot file and
  the dataclass defaults built from it) starts out redirected too. ``conftest.py`` points the
  constants at a fresh directory for each test on top of that; the import-time copies keep the
  shared session directory, not the per-test one.
- It points ``constants.MODELS_PATH`` at a second temporary directory for the session, which
  ``conftest.py`` again replaces for each test. Code reads that constant when it runs, so run
  directories, walk-forward checkpoints and the web app's run index for this checkout all follow
  it.
- It refuses every TCP connection and every lookup of a host name other than ``localhost``, for
  the whole session, module- and session-scoped fixtures included. A refusal raises ``OSError``
  (``socket.gaierror`` for a lookup), as on an offline machine, so code that degrades on a
  network failure still runs and an offline machine never waits on DNS; the attempt is recorded
  and fails the test (or the session) in ``conftest.py``.

It covers the paths built from ``constants.DATA_PATH`` and ``constants.MODELS_PATH`` only. Out of
scope, because none of them reads those constants, so redirecting them would need a package
change; tests that touch them pass their own paths:

- ``api.settings.Settings()`` built without a root, which fills its data, state and database
  paths from ``constants.ROOT_DIR`` in ``model_post_init`` (the checkout's ``data/web/app.db``,
  the database the running web app uses); its models directory follows ``MODELS_PATH``;
- the floor-sigma reference runs (``constants.FLOOR_SIGMA_REFERENCE_RUNS``, by default
  ``models/qb_elos_measure/new_file_seed*``), which ``ml/floor_sigma.py`` resolves against
  ``constants.ROOT_DIR``: the default of ``nfl-predictor train``'s
  ``--floor-sigma-reference-runs`` and the weekly run's ``floor_sigma_reference_runs``, read when
  a model is trained;
- the relative argparse defaults in ``cli/rankings.py`` (``Path("data/...")`` for
  ``--data-ml``, ``--data-schedule`` and ``--out-dir``), resolved against the current directory.
"""

from __future__ import annotations

import atexit
import ipaddress
import shutil
import socket
import sys
import tempfile
from pathlib import Path, PurePath
from typing import Any

import pytest

from nfl_predictor import constants

# The checkout's data and models directories, worked out from the root so a second import cannot
# mistake the redirected values for them.
PACKAGED_DATA_DIR = Path(constants.ROOT_DIR) / "data"
PACKAGED_MODELS_DIR = Path(constants.ROOT_DIR) / "models"
# The constants that hold a data path, with the part of the data directory each one names.
CONSTANT_DATA_PATHS: dict[str, tuple[str, ...]] = {
    "DATA_PATH": (),
    "NFLREADPY_CACHE_DIR": ("cache", "nflreadpy"),
    "NFELO_CACHE_DIR": ("cache", "nfelo"),
}
SESSION_DATA_DIR = Path(tempfile.mkdtemp(prefix="nfl-predictor-tests-data-"))
atexit.register(shutil.rmtree, SESSION_DATA_DIR, ignore_errors=True)
SESSION_MODELS_DIR = Path(tempfile.mkdtemp(prefix="nfl-predictor-tests-models-"))
atexit.register(shutil.rmtree, SESSION_MODELS_DIR, ignore_errors=True)
# The session-wide patches; never undone, since they must outlive every fixture.
_SESSION_PATCH = pytest.MonkeyPatch()
_installed: list[bool] = []

# Every refused connection or lookup, in order, until a test or the session takes them.
NETWORK_ATTEMPTS: list[str] = []
_NETWORK_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_LOCAL_HOSTS = frozenset({"", "localhost", b"", b"localhost"})
_REAL_CONNECT = socket.socket.connect
_REAL_GETADDRINFO = socket.getaddrinfo


def _refused_connect(sock: socket.socket, address: Any) -> None:
    """Refuse and record a TCP connection; let Unix sockets through."""
    if sock.family in _NETWORK_FAMILIES:
        NETWORK_ATTEMPTS.append(f"connect {address!r}")
        msg = f"network access is blocked in tests: {address!r}"
        raise OSError(msg)
    _REAL_CONNECT(sock, address)


def _needs_dns(host: Any) -> bool:
    """Return whether looking ``host`` up would ask a name server."""
    if host is None or host in _LOCAL_HOSTS:
        return False
    text = host.decode() if isinstance(host, bytes) else str(host)
    try:
        ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return True
    return False


def _refused_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    """Refuse and record a host-name lookup; numeric and local addresses resolve as usual."""
    if _needs_dns(host):
        NETWORK_ATTEMPTS.append(f"resolve {host!r}")
        msg = f"name resolution is blocked in tests: {host!r}"
        raise socket.gaierror(socket.EAI_NONAME, msg)
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def install() -> None:
    """Redirect the data and models constants to session directories and refuse the network."""
    if _installed:
        return
    for name, parts in CONSTANT_DATA_PATHS.items():
        _SESSION_PATCH.setattr(constants, name, SESSION_DATA_DIR.joinpath(*parts))
    _SESSION_PATCH.setattr(constants, "MODELS_PATH", SESSION_MODELS_DIR)
    _SESSION_PATCH.setattr(socket.socket, "connect", _refused_connect)
    _SESSION_PATCH.setattr(socket, "getaddrinfo", _refused_getaddrinfo)
    _installed.append(True)


def take_network_attempts() -> list[str]:
    """Return the recorded refusals and forget them."""
    attempts = list(NETWORK_ATTEMPTS)
    NETWORK_ATTEMPTS.clear()
    return attempts


def packaged_path(
    value: object, redirected_dirs: tuple[Path, ...], packaged_dir: Path = PACKAGED_DATA_DIR
) -> Path | None:
    """Return the checkout path a redirected path stands for, or ``None`` for any other value.

    ``packaged_dir`` is the checkout directory the ``redirected_dirs`` stand in for.
    """
    if not isinstance(value, PurePath):
        return None
    path = Path(value)
    for directory in redirected_dirs:
        if path.is_relative_to(directory):
            return packaged_dir / path.relative_to(directory)
    return None


def restore_packaged_paths(
    patch: pytest.MonkeyPatch,
    redirected_dirs: tuple[Path, ...],
    packaged_dir: Path = PACKAGED_DATA_DIR,
) -> None:
    """Point every loaded package module's redirected path back at the checkout's."""
    for name, module in list(sys.modules.items()):
        if name != "nfl_predictor" and not name.startswith("nfl_predictor."):
            continue
        for attribute, value in list(vars(module).items()):
            packaged = packaged_path(value, redirected_dirs, packaged_dir)
            if packaged is not None:
                patch.setattr(module, attribute, packaged)
