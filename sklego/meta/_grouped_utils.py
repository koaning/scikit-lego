from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.utils.validation import _ensure_no_complex_data

from sklego.common import as_list


def split_groups_and_values(
    X, groups, name="", **kwargs
) -> Tuple[pd.DataFrame, np.ndarray]:
    general_checks(X, name=name)

    try:
        if isinstance(X, pd.DataFrame):
            X_group = X.loc[:, as_list(groups)]
            X_value = X.drop(columns=groups).values
        else:
            X_group = pd.DataFrame(X[:, as_list(groups)])
            X_value = np.delete(X, as_list(groups), axis=1)
    except (KeyError, IndexError):
        raise ValueError(f"Could not drop groups {groups} from columns of X")

    X_group = check_grouping_columns(X_group, **kwargs)
    X_value = check_value_columns(X_value, **kwargs)

    return X_group, X_value


def general_checks(X, name):
    _ensure_no_complex_data(X)

    if issparse(X):  # sklearn.validation._ensure_sparse_format to complicated
        raise ValueError(f"The estimator {name} does not work on sparse matrices")

    if X.ndim == 1 or X.shape[1] < 2:
        raise ValueError(
            f"0 feature(s) (shape={X.shape}) while a minimum of 2 is required."
        )


def check_value_columns(X_value, **kwargs) -> np.ndarray:
    """Do basic checks on the value columns"""
    return check_array(X_value, **kwargs)


def check_grouping_columns(X_group, **kwargs) -> pd.DataFrame:
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
