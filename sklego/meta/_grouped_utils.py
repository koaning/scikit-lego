from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils.validation import _ensure_no_complex_data

from sklego.common import as_list


def constant_shrinkage(group_sizes: list, alpha: float) -> np.ndarray:
    r"""
    The augmented prediction for each level is the weighted average between its prediction and the augmented
    prediction for its parent.

    Let $\hat{y}_i$ be the prediction at level $i$, with $i=0$ being the root, than the augmented prediction
    $\hat{y}_i^* = \alpha \hat{y}_i + (1 - \alpha) \hat{y}_{i-1}^*$, with $\hat{y}_0^* = \hat{y}_0$.
    """
    return np.array(
        [alpha ** (len(group_sizes) - 1)]
        + [
            alpha ** (len(group_sizes) - 1 - i) * (1 - alpha)
            for i in range(1, len(group_sizes) - 1)
        ]
        + [(1 - alpha)]
    )


def relative_shrinkage(group_sizes: list) -> np.ndarray:
    """Weigh each group according to it's size"""
    return np.array(group_sizes)


def min_n_obs_shrinkage(group_sizes: list, min_n_obs) -> np.ndarray:
    """Use only the smallest group with a certain amount of observations"""
    if min_n_obs > max(group_sizes):
        raise ValueError(
            f"There is no group with size greater than or equal to {min_n_obs}"
        )

    res = np.zeros(len(group_sizes))
    res[np.argmin(np.array(group_sizes) >= min_n_obs) - 1] = 1
    return res


def _split_groups_and_values(
    X, groups, name="", min_value_cols=1, check_X=True, **kwargs
) -> Tuple[pd.DataFrame, np.ndarray]:
    _data_format_checks(X, name=name)
    _shape_check(X, min_value_cols)

    try:
        if isinstance(X, pd.DataFrame):
            X_group = X.loc[:, as_list(groups)]
            X_value = X.drop(columns=groups).values
        else:
            X_group = pd.DataFrame(X[:, as_list(groups)])
            pos_indexes = range(X.shape[1])
            X_value = np.delete(X, [pos_indexes[g] for g in as_list(groups)], axis=1)
    except (KeyError, IndexError):
        raise ValueError(f"Could not drop groups {groups} from columns of X")

    X_group = _check_grouping_columns(X_group, **kwargs)

    if check_X:
        X_value = check_array(X_value, **kwargs)

    return X_group, X_value


def _data_format_checks(X, name):
    _ensure_no_complex_data(X)

    if issparse(X):  # sklearn.validation._ensure_sparse_format to complicated
        raise ValueError(f"The estimator {name} does not work on sparse matrices")


def _shape_check(X, min_value_cols):
    if min_value_cols > 1:
        if X.ndim == 1 or X.shape[1] < 2:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of {min_value_cols} is required."
            )
    else:
        if X.ndim == 2 and X.shape[1] < 1:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of {min_value_cols} is required."
            )


def _check_grouping_columns(X_group, **kwargs) -> pd.DataFrame:
    """Do basic checks on grouping columns"""
    # Do regular checks on numeric columns
    X_group_num = X_group.select_dtypes(include="number")
    if X_group_num.shape[1]:
        check_array(X_group_num, **kwargs)

    # Only check missingness in object columns
    if X_group.select_dtypes(exclude="number").isnull().any(axis=None):
        raise ValueError("X has NaN values")

    # The grouping part we always want as a DataFrame with range index
    return X_group.reset_index(drop=True)
