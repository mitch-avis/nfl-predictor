from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

def brier_score_loss(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    pos_label: float | str | None = None,
    labels: ArrayLike | None = None,
    scale_by_half: bool | Literal["auto"] = "auto",
) -> float: ...
def log_loss(
    y_true: ArrayLike,
    y_proba: ArrayLike | None = None,
    *,
    normalize: bool = True,
    sample_weight: ArrayLike | None = None,
    labels: ArrayLike | None = None,
) -> float: ...
@overload
def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    multioutput: Literal["raw_values"],
) -> np.ndarray: ...
@overload
def mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    multioutput: Literal["uniform_average"] | ArrayLike = "uniform_average",
) -> float: ...
