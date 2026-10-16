from typing import Literal

class OneHotEncoder:
    def __init__(
        self,
        *,
        handle_unknown: Literal["error", "ignore", "infrequent_if_exist", "warn"] = "error",
        sparse_output: bool = True,
    ) -> None: ...

class StandardScaler:
    def __init__(
        self, *, copy: bool = True, with_mean: bool = True, with_std: bool = True
    ) -> None: ...
