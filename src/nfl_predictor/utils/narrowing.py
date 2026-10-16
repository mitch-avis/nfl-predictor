"""Type parsed JSON and unpickled values as the containers the reading code expects.

``json.loads`` and ``joblib.load`` return untyped values, and an ``isinstance`` check on one
cannot carry a container's type arguments, so the type checker would treat every key, element
and value read from it as unknown. These helpers do the ``isinstance`` check and give the
result the type the callers read it as, returning the same object (never a copy).
"""

from typing import Any, cast


def as_str_dict(value: object) -> dict[str, Any] | None:
    """Return ``value`` as a string-keyed dict when it is a dict, else ``None``.

    The keys are not checked: JSON objects always have string keys, and so do the payloads
    this project pickles. The values are whatever the file held.
    """
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


def as_list(value: object) -> list[Any] | None:
    """Return ``value`` when it is a list (a tuple is not), else ``None``."""
    if isinstance(value, list):
        return cast("list[Any]", value)
    return None
