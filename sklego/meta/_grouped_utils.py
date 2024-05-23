from __future__ import annotations

from typing import List

import narwhals as nw
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils.validation import _ensure_no_complex_data


def parse_X_y(X, y, groups, check_X=True, **kwargs) -> nw.DataFrame:
    """Converts X, y to narwhals dataframe.

    If it is not a supported dataframe, it uses pandas constructor as a fallback.

    Additionally, data checks are performed.
    """
    # Check raw X
    _data_format_checks(X)

    # Convert X to Narwhals frame
    X = nw.from_native(X, strict=False, eager_only=True)
    if not isinstance(X, nw.DataFrame):
        X = nw.from_native(pd.DataFrame(X))

    # Check groups and feaures values
    if groups is not None:
        _validate_groups_values(X, groups)

        if check_X:
            check_array(X.drop(groups), **kwargs)

    # Convert y and assign it to the frame
    n_samples = X.shape[0]
    native_space = nw.get_native_namespace(X)

    y_native = native_space.Series([None] * n_samples) if y is None else native_space.Series(y)
    return X.with_columns(__sklego_target__=nw.from_native(y_native, allow_series=True))


def _validate_groups_values(X: nw.DataFrame, groups: List[int] | List[str]) -> None:
    X_cols = X.columns
    unexisting_cols = [g for g in groups if g not in X_cols]

    if len(unexisting_cols):
        raise ValueError(f"The following groups are not available in X: {unexisting_cols}")

    if X.select(nw.col(groups).is_null().any()).to_numpy().squeeze().any():
        raise ValueError("Groups values have NaN")


def _data_format_checks(X):
    """Checks that X is not sparse nor has complex dtype"""
    _ensure_no_complex_data(X)

    if issparse(X):  # sklearn.validation._ensure_sparse_format to complicated
        msg = "Estimator does not work on sparse matrices"
        raise ValueError(msg)
