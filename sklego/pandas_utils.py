import datetime as dt
import inspect
from functools import partial, wraps

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift

from sklego.common import as_list


def _get_shape_delta(old_shape, new_shape):
    diffs = [
        ("+" if new > old else "") + str(new - old)
        for new, old in zip(new_shape, old_shape)
    ]

    return f"delta=({', '.join(diffs)})"


def log_step(
    func=None,
    *,
    time_taken=True,
    shape=True,
    shape_delta=False,
    names=False,
    dtypes=False,
    print_fn=print,
    display_args=True,
    log_error=True,
):
    """
    Decorates a function that transforms a pandas dataframe to add automated logging statements

    :param func: callable, function to log, defaults to None
    :param time_taken: bool, log the time it took to run a function, defaults to True
    :param shape: bool, log the shape of the output result, defaults to True
    :param shape_delta: bool, log the difference in shape of input and output, defaults to False
    :param names: bool, log the names of the columns of the result, defaults to False
    :param dtypes: bool, log the dtypes of the results, defaults to False
    :param print_fn: callable, print function (e.g. print or logger.info), defaults to print
    :param print_args: bool, whether or not to print the arguments given to the function.
    :param log_error: bool, whether to add the Exception message to the log if the function fails, defaults to True.
    :returns: the result of the function

    :Example:
    >>> @log_step
    ... def remove_outliers(df, min_obs=5):
    ...     pass

    >>> @log_step(print_fn=logging.info, shape_delta=True)
    ... def remove_outliers(df, min_obs=5):
    ...     pass

    """

    if func is None:
        return partial(
            log_step,
            time_taken=time_taken,
            shape=shape,
            shape_delta=shape_delta,
            names=names,
            dtypes=dtypes,
            print_fn=print_fn,
            display_args=display_args,
            log_error=log_error,
        )

    names = False if dtypes else names

    @wraps(func)
    def wrapper(*args, **kwargs):
        if shape_delta:
            old_shape = args[0].shape
        tic = dt.datetime.now()

        optional_strings = []
        try:
            result = func(*args, **kwargs)
            optional_strings = [
                f"time={dt.datetime.now() - tic}" if time_taken else None,
                f"n_obs={result.shape[0]}, n_col={result.shape[1]}" if shape else None,
                _get_shape_delta(old_shape, result.shape) if shape_delta else None,
                f"names={result.columns.to_list()}" if names else None,
                f"dtypes={result.dtypes.to_dict()}" if dtypes else None,
            ]
            return result
        except Exception as exc:
            optional_strings = [
                f"time={dt.datetime.now() - tic}" if time_taken else None,
                "FAILED" + (f" with error: {exc}" if log_error else "")
            ]
            raise
        finally:
            combined = " ".join([s for s in optional_strings if s])

            if display_args:

                func_args = inspect.signature(func).bind(*args, **kwargs).arguments
                func_args_str = "".join(
                    ", {} = {!r}".format(*item) for item in list(func_args.items())[1:]
                )
                print_fn(f"[{func.__name__}(df{func_args_str})] " + combined,)
            else:
                print_fn(f"[{func.__name__}]" + combined,)

    return wrapper


def log_step_extra(
    *log_functions, print_fn=print, **log_func_kwargs,
):
    """
    Decorates a function that transforms a pandas dataframe to add automated logging statements

    :param *log_functions: callable(s), functions that take the output of the decorated function and turn it into a log.
                                        Note that the output of each log_function is casted to string using `str()`
    :param print_fn: callable, print function (e.g. print or logger.info), defaults to print
    :param **log_func_kwargs: keyword arguments to be passed to log_functions
    :returns: the result of the function

    :Example:
    >>> @log_step_extra(lambda d: d["some_column"].value_counts())
    ... def remove_outliers(df, min_obs=5):
    ...     pass

    """
    if not log_functions:
        raise ValueError("Supply at least one log_function for log_step_extra")

    def _log_step_extra(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            func_args_str = "".join(
                ", {} = {!r}".format(*item) for item in list(func_args.items())[1:]
            )

            try:
                extra_logs = " ".join(
                    [str(log_f(result, **log_func_kwargs)) for log_f in log_functions]
                )
            except TypeError:
                raise ValueError(
                    f"All log functions should be callable, got {[type(log_f) for log_f in log_functions]}"
                )

            print_fn(f"[{func.__name__}(df{func_args_str})] " + extra_logs,)

            return result

        return wrapper

    return _log_step_extra


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

    combos = (shift(X[:, col], -lag, cval=np.NaN) for col in cols for lag in lags)

    # In integer-based ndarrays, NaN values are represented as
    # -9223372036854775808, so we convert back and forth from
    # original to float and back to original dtype
    original_type = X.dtype
    X = np.asarray(X, dtype=float)
    answer = np.column_stack((X, *combos))

    # Remove rows that contain NA values when drop_na is truthy
    if drop_na:
        answer = answer[~np.isnan(answer).any(axis=1)]

    # Change dtype back to its original
    answer = np.asarray(answer, dtype=original_type)
    return answer


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
        df[col].shift(-lag).rename(col + str(lag)) for col in cols for lag in lags
    )

    answer = pd.concat([df, *combos], axis=1)

    # Remove rows that contain NA values when drop_na is truthy
    if drop_na:
        answer = answer.dropna()

    return answer
