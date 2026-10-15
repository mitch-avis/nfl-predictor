from os import PathLike
from typing import IO, Any, Literal

def dump(
    value: object,
    filename: str | PathLike[str] | IO[bytes],
    compress: bool | int | tuple[str, int] = 0,
    protocol: int | None = None,
) -> list[str]: ...
def load(
    filename: str | PathLike[str] | IO[bytes],
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
    ensure_native_byte_order: bool | Literal["auto"] = "auto",
) -> Any: ...
