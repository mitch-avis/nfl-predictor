"""A small cache keyed by a file's identity (path, size, mtime).

Rereads are free until the file changes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_ENTRIES = 64

_entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_lock = threading.Lock()


def file_key(path: Path) -> tuple[str, int, int]:
    """Return ``(path, size, mtime_ns)`` for ``path``; raises ``FileNotFoundError`` when absent."""
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def cached(path: Path, loader: Callable[[Path], Any], *tags: Any) -> Any:
    """Return ``loader(path)``, memoized until the file's size or mtime changes.

    ``tags`` distinguish different loaders applied to the same file.
    """
    key = (*file_key(path), *tags)
    with _lock:
        if key in _entries:
            _entries.move_to_end(key)
            return _entries[key]
    value = loader(path)
    with _lock:
        _entries[key] = value
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)
    return value


def clear() -> None:
    """Drop every cached entry (tests)."""
    with _lock:
        _entries.clear()
