from collections.abc import Callable, Sequence
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import sparray, spmatrix

# A transformer is any object with fit and transform (sklearn checks at fit time), or one of
# the keywords "drop" and "passthrough".
type _TransformerSpec = object
type _ColumnSelector = (
    str | int | slice | list[str] | list[int] | np.ndarray | Callable[[pd.DataFrame], list[str]]
)
type _Output = np.ndarray | spmatrix | sparray | pd.DataFrame

class ColumnTransformer:
    transformers: list[tuple[str, _TransformerSpec, _ColumnSelector]]
    transformers_: list[tuple[str, _TransformerSpec, _ColumnSelector]]
    sparse_threshold: float
    feature_names_in_: npt.NDArray[np.object_]
    def __init__(
        self,
        transformers: Sequence[tuple[str, _TransformerSpec, _ColumnSelector]],
        *,
        remainder: Literal["drop", "passthrough"] | _TransformerSpec = "drop",
        sparse_threshold: float = 0.3,
        n_jobs: int | None = None,
        verbose_feature_names_out: bool = True,
    ) -> None: ...
    def fit(self, X: pd.DataFrame | ArrayLike, y: ArrayLike | None = None) -> Self: ...
    def fit_transform(self, X: pd.DataFrame | ArrayLike, y: ArrayLike | None = None) -> _Output: ...
    def transform(self, X: pd.DataFrame | ArrayLike) -> _Output: ...
    def get_feature_names_out(
        self, input_features: ArrayLike | None = None
    ) -> npt.NDArray[np.object_]: ...
