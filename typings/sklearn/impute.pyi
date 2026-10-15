from collections.abc import Callable
from typing import Literal

import numpy as np

class SimpleImputer:
    def __init__(
        self,
        *,
        missing_values: float | str | None = ...,
        strategy: Literal["mean", "median", "most_frequent", "constant"]
        | Callable[[np.ndarray], float] = "mean",
        fill_value: float | str | None = None,
        copy: bool = True,
        add_indicator: bool = False,
        keep_empty_features: bool = False,
    ) -> None: ...
