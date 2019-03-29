import inspect
import logging
import numpy as np
import pandas as pd
import datetime as dt

from functools import wraps
from sklego.common import as_list
from scipy.ndimage.interpolation import shift


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(inspect.stack()[1].function)

        tic = dt.datetime.now()
        result_df = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ''.join(', {} = {!r}'.format(*item) for item in list(func_args.items())[1:])

        logger.info(
            f"[ {func.__name__}(df{func_args_str}) ] "
            f"n_obs={result_df.shape[0]} n_col={result_df.shape[1]} time={time_taken}")
        return result_df
    return wrapper


def add_lags(X, cols, lags, drop_na=True):
    """
    Appends lag column(s).

    :param X: array-like, shape=(n_columns, n_samples,) training data.
    :param cols: column name(s) or index (indices).
    :param lags: the amount of lag for each col.
    :param drop_na: remove rows that contain NA values.
    :returns: ``pd.DataFrame | np.ndarray`` with only the selected cols.

    :Example:

    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2, 3],
    ...                    [4, 5, 6],
    ...                    [7, 8, 9]],
    ...                    columns=['a', 'b', 'c'],
    ...                    index=[1, 2, 3])

    >>> add_lags(df, 'a', [1]) # doctest: +NORMALIZE_WHITESPACE
       a  b  c  a1
    1  1  2  3  4.0
    2  4  5  6  7.0

    >>> add_lags(df, ['a', 'b'], 2) # doctest: +NORMALIZE_WHITESPACE
       a  b  c  a2   b2
    1  1  2  3  7.0  8.0

    >>> import numpy as np
    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])

    >>> add_lags(X, 0, [1])
    array([[1, 2, 3, 4],
           [4, 5, 6, 7]])

    >>> add_lags(X, 1, [-1, 1])
    array([[4, 5, 6, 2, 8]])
    """

    # A single lag will be put in a list
    lags = as_list(lags)

    # Now we can iterate over the list to determine
    # whether it is a list of integers
    if not all(isinstance(x, int) for x in lags):
        raise ValueError("lags must be a list of type: " + str(int))

    # The keys of the allowed_inputs dict contain the allowed
    # types, and the values contain the associated handlers
    allowed_inputs = {
        pd.core.frame.DataFrame: _add_lagged_pandas_columns,
        np.ndarray: _add_lagged_numpy_columns,
    }

    # Choose the correct handler based on the input class
    for allowed_input, handler in allowed_inputs.items():
        if isinstance(X, allowed_input):
            return handler(X, cols, lags, drop_na)

    # Otherwise, raise a ValueError
    allowed_input_names = list(allowed_inputs.keys())
    raise ValueError("X type should be one of:", allowed_input_names)


def _add_lagged_numpy_columns(X, cols, lags, drop_na):
    """
    Append a lag columns.

    :param df: the input ``np.ndarray``.
    :param cols: column index / indices.
    :param drop_na: remove rows that contain NA values.
    :returns: ``np.ndarray`` with the concatenated lagged cols.
    """

    cols = as_list(cols)

    if not all([isinstance(col, int) for col in cols]):
        raise ValueError("Matrix columns are indexed by integers")

    if not all([col < X.shape[1] for col in cols]):
        raise KeyError("The column does not exist")

    combos = (
        shift(X[:, col], -lag, cval=np.NaN)
        for col in cols
        for lag in lags
    )

    # In integer-based ndarrays, NaN values are represented as
    # -9223372036854775808, so we convert back and forth from
    # original to float and back to original dtype
    original_type = X.dtype
    X = np.asarray(X, dtype=float)
    ans = np.column_stack((X, *combos))

    # Remove rows that contain NA values when drop_na is truthy
    if drop_na:
        ans = ans[~np.isnan(ans).any(axis=1)]

    # Change dtype back to its original
    ans = np.asarray(ans, dtype=original_type)
    return ans


def _add_lagged_pandas_columns(df, cols, lags, drop_na):
    """
    Append a lag columns.

    :param df: the input ``pd.DataFrame``.
    :param cols: column name(s).
    :param drop_na: remove rows that contain NA values.
    :returns: ``pd.DataFrame`` with the concatenated lagged cols.
    """

    cols = as_list(cols)

    # Indexes are not supported as pandas column names may be
    # integers themselves, introducing unexpected behaviour
    if not all([col in df.columns.values for col in cols]):
        raise KeyError("The column does not exist")

    combos = (
        df[col].shift(-lag).rename(col + str(lag))
        for col in cols
        for lag in lags
    )

    ans = pd.concat([df, *combos], axis=1)

    # Remove rows that contain NA values when drop_na is truthy
    if drop_na:
        ans = ans.dropna()

    return ans
