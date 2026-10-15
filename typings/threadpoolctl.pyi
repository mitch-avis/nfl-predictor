from types import TracebackType
from typing import NotRequired, Self, TypedDict

class _LibraryInfo(TypedDict):
    user_api: str
    internal_api: str
    prefix: str
    filepath: str
    version: str | None
    num_threads: int
    threading_layer: NotRequired[str]
    architecture: NotRequired[str]

def threadpool_info(debugging_info: bool = False) -> list[_LibraryInfo]: ...

class threadpool_limits:
    def __init__(
        self,
        limits: int | dict[str, int] | str | None = None,
        user_api: str | None = None,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None: ...
