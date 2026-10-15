"""Typed stand-ins for the callables tests swap in with ``monkeypatch.setattr``.

``monkeypatch.setattr`` accepts any value, so a ``lambda *_args, **_kwargs: value`` gives strict
pyright no type for its parameters. ``returning(value)`` is the typed form of that lambda. A stub
whose arity matters (it should fail when called with the wrong arguments) is better written as a
typed lambda against a declared ``Callable`` type, or as a small typed function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def returning[T](value: T) -> Callable[..., T]:
    """Return a callable that accepts any arguments and returns ``value`` on every call.

    The same object comes back each time: when the code under test mutates the result and each
    call needs a fresh one, build it inside a typed function instead.
    """

    def stub(*_args: object, **_kwargs: object) -> T:
        return value

    return stub
