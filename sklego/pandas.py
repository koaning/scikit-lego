import inspect
import logging
import numpy as np
import pandas as pd
import datetime as dt

from functools import wraps
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
  
  
def _as_list(val):
    """
    Helper function, always returns a list.

    :param val: a list or a single value.
    :returns: the input value as a list.

    :Example:

    >>> _as_list('test')
    ['test']

    >>> _as_list(['test1', 'test2'])
    ['test1', 'test2']
    """

    if not isinstance(val, list):
        return [val]
    return val


def _negate_lags(lags):
    """
    Helper function, negates an int or iterable of ints.

    :param lags: iterable of ints or int.
    :returns: iterable of ints that are negated.

    :Example:

    >>> _negate_lags(3)
    [-3]

    >>> _negate_lags([3])
    [-3]
    """

    def negate(x):
        return -x

    lags = _as_list(lags)

    if not all(isinstance(x, int) for x in lags):
        raise ValueError("Must be list of ints")

    return [-x for x in lags]


def add_lag(X, cols, lags):
    """
    Appends lag column(s).

    :param X: array-like, shape=(n_columns, n_samples,) training data.
    :param cols: an iterable of column names or a single column name string.
    :param lags: iterable of ints or int.
    :returns: ``pd.DataFrame | np.ndarray`` with only the selected columns

    :Example:

    >>> import pandas as pd
    >>> df = pd.DataFrame([[1, 2, 3],
    ...                    [4, 5, 6],
    ...                    [7, 8, 9]],
    ...                    columns=['a', 'b', 'c'],
    ...                    index=[1, 2, 3])

    >>> add_lag(df, 'a', [1]) # doctest: +NORMALIZE_WHITESPACE
    a  b  c  a-1
    1  1  2  3  4.0
    2  4  5  6  7.0

    >>> add_lag(df, ['a', 'b'], 2) # doctest: +NORMALIZE_WHITESPACE
       a  b  c  a-2  b-2
    1  1  2  3  7.0  8.0

    >>> import numpy as np
    >>> X = np.array([[-4, 2],
    ...               [-2, 0],
    ...               [4, -6]])

    >>> add_lag(X, [0, 1], 1)
    array([[-4,  2, -2,  0],
           [-2,  0,  4, -6],
           [ 4, -6,  0,  0]])

    >>> add_lag(X, 1, [-1, 1])
    array([[-4,  2,  0,  0],
           [-2,  0,  2, -6],
           [ 4, -6,  0,  0]])
    """

    lags = _negate_lags(lags)

    allowed_inputs = {
        pd.core.frame.DataFrame: _add_lagged_pandas_columns,
        np.ndarray: _add_lagged_numpy_columns,
    }

    for allowed_input, handler in allowed_inputs.items():
        if isinstance(X, allowed_input):
            return handler(X, cols, lags)

    allowed_input_names = list(allowed_inputs.keys())
    raise ValueError("X type should be one of:", allowed_input_names)


def _add_lagged_numpy_columns(X, cols, lags):
    """
    Append a lag columns, removes all NA values.

    :param df: the input ``np.ndarray``.
    :param cols: an iterable of column names, or a single column name.
    :returns: ``np.ndarray`` with the concatenated lagged columns.

    :Example:
    >>> import numpy as np
    >>> X = np.array([[-4, 2],
    ...               [-2, 0],
    ...               [4, -6]])

    >>> _add_lagged_numpy_columns(X, ['test'], 1)
    Traceback (most recent call last):
        ...
    ValueError: Matrix columns are indexed by integers

    >>> _add_lagged_numpy_columns(X, ['test'], 1)
    Traceback (most recent call last):
        ...
    ValueError: Matrix columns are indexed by integers

    >>> _add_lagged_numpy_columns(X, [15], 1)
    Traceback (most recent call last):
        ...
    KeyError: 'The column does not exist'
    """

    cols = _as_list(cols)

    if not all([isinstance(col, int) for col in cols]):
        raise ValueError("Matrix columns are indexed by integers")

    if not all([col < X.shape[1] for col in cols]):
        raise KeyError("The column does not exist")

    combos = (shift(X[:, col], lag) for col in cols for lag in lags)

    return np.column_stack((X, *combos))


def _add_lagged_pandas_columns(df, cols, lags):
    """
    Append a lag columns, removes all NA values.

    :param df: the input ``pd.DataFrame``.
    :param cols: an iterable of column names, or a single column name.
    :returns: ``pd.DataFrame`` with the concatenated lagged columns.

    :Example:

    >>> df = pd.DataFrame([[1, 2, 3],
    ...                    [4, 5, 6],
    ...                    [7, 8, 9]],
    ...                    columns=['a', 'b', 'c'],
    ...                    index=[1, 2, 3])

    >>> _add_lagged_pandas_columns(df, ['last_name'], 1)
    Traceback (most recent call last):
        ...
    KeyError: 'The column does not exist'
    """

    cols = _as_list(cols)

    if not all([col in df.columns.values for col in cols]):
        raise KeyError("The column does not exist")

    combos = [
        df[col].shift(lag).rename(col + str(lag))
        for col in cols
        for lag in lags
    ]

    return pd.concat([df, *combos], axis=1).dropna()
