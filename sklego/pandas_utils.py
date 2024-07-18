import datetime as dt
import inspect
from functools import partial, wraps

import narwhals.stable.v1 as nw
import numpy as np
from scipy.ndimage import shift

from sklego.common import as_list


def _get_shape_delta(old_shape, new_shape):
    """Returns a string with the difference in shape between old and new."""
    diffs = [("+" if new > old else "") + str(new - old) for new, old in zip(new_shape, old_shape)]

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
    """Decorates a function that transforms a pandas dataframe to add automated logging statements.

    Parameters
    ----------
    func : Callable | None, default=None
        The function to decorate with logs. If None, returns a partial function with the given arguments.
    time_taken : bool, default=True
        Whether or not to log the time it took to run a function.
    shape : bool, default=True
        Whether or not to log the shape of the output result.
    shape_delta : bool, default=False
        Whether or not to log the difference in shape of input and output.
    names : bool, default=False
        Whether or not to log the names of the columns of the result.
    dtypes : bool, default=False
        Whether or not to log the dtypes of the result.
    print_fn : Callable, default=print
        Print function to use (e.g. `print` or `logger.info`)
    display_args : bool, default=True
        Whether or not to display the arguments given to the function.
    log_error : bool, default=True
        Whether or not to add the Exception message to the log if the function fails.

    Returns
    -------
    Callable
        The decorated function.

    Examples
    --------
    ```py
    @log_step
    def remove_outliers(df, min_obs=5):
        pass

    @log_step(print_fn=logging.info, shape_delta=True)
    def remove_outliers(df, min_obs=5):
        pass
    ```
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
                "FAILED" + (f" with error: {exc}" if log_error else ""),
            ]
            raise
        finally:
            combined = " ".join([s for s in optional_strings if s])

            if display_args:
                func_args = inspect.signature(func).bind(*args, **kwargs).arguments
                func_args_str = "".join(", {} = {!r}".format(*item) for item in list(func_args.items())[1:])
                print_fn(
                    f"[{func.__name__}(df{func_args_str})] " + combined,
                )
            else:
                print_fn(
                    f"[{func.__name__}]" + combined,
                )

    return wrapper


def log_step_extra(
    *log_functions,
    print_fn=print,
    **log_func_kwargs,
):
    """Decorates a function that transforms a pandas dataframe to add automated logging statements.

    Parameters
    ----------
    *log_functions : List[Callable]
        Functions that take the output of the decorated function and turn it into a log.
        Note that the output of each log_function is casted to string using `str()`.
    print_fn: Callable, default=print
        Print function (e.g. `print` or `logger.info`).
    **log_func_kwargs: dict
        Keyword arguments to be passed to `log_functions`

    Returns
    -------
    Callable
        The decorated function.

    Examples
    --------
    ```py
    @log_step_extra(lambda d: d["some_column"].value_counts())
    def remove_outliers(df, min_obs=5):
        pass
    ```
    """
    if not log_functions:
        raise ValueError("Supply at least one log_function for log_step_extra")

    def _log_step_extra(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            func_args_str = "".join(", {} = {!r}".format(*item) for item in list(func_args.items())[1:])

            try:
                extra_logs = " ".join([str(log_f(result, **log_func_kwargs)) for log_f in log_functions])
            except TypeError:
                raise ValueError(
                    f"All log functions should be callable, got {[type(log_f) for log_f in log_functions]}"
                )

            print_fn(
                f"[{func.__name__}(df{func_args_str})] " + extra_logs,
            )

            return result

        return wrapper

    return _log_step_extra


def add_lags(X, cols, lags, drop_na=True):
    """Appends lag column(s).

    Parameters
    ----------
    X : array-like
        Data to be lagged.
    cols : str | int | List[str] | List[int]
        Column name(s) or index (indices).
    lags : int | List[int]
        The amount of lag for each col.
    drop_na : bool, default=True
        Whether or not to remove rows that contain NA values.

    Returns
    -------
    DataFrame | np.ndarray
        With only the selected cols.

    Raises
    ------
    ValueError
        If the input is not a supported DataFrame.

    Notes
    -----
    Native cross-dataframe support is achieved using
    [Narwhals](https://narwhals-dev.github.io/narwhals/){:target="_blank"}.
    Supported dataframes are:

    - pandas
    - Polars (eager or lazy)
    - Modin
    - cuDF

    See [Narwhals docs](https://narwhals-dev.github.io/narwhals/extending/){:target="_blank"} for an up-to-date list
    (and to learn how you can add your dataframe library to it!).

    Examples
    --------
    ```py
    import pandas as pd
    df = pd.DataFrame([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
            columns=["a", "b", "c"],
            index=[1, 2, 3]
    )

    add_lags(df, "a", [1]) # doctest: +NORMALIZE_WHITESPACE
    '''
        a  b  c  a1
    1  1  2  3  4.0
    2  4  5  6  7.0
    '''

    add_lags(df, ["a", "b"], 2) # doctest: +NORMALIZE_WHITESPACE
    '''
        a  b  c  a2   b2
    1  1  2  3  7.0  8.0
    '''

    import numpy as np
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    add_lags(X, 0, [1])
    # array([[1, 2, 3, 4],
    #        [4, 5, 6, 7]])

    add_lags(X, 1, [-1, 1])
    # array([[4, 5, 6, 2, 8]])
    ```
    """

    # A single lag will be put in a list
    lags = as_list(lags)

    # Now we can iterate over the list to determine
    # whether it is a list of integers
    if not all(isinstance(x, int) for x in lags):
        raise ValueError("lags must be a list of type: " + str(int))

    # The keys of the allowed_inputs dict contain the allowed
    # types, and the values contain the associated handlers
    X = nw.from_native(X, strict=False)
    allowed_inputs = {
        nw.DataFrame: _add_lagged_dataframe_columns,
        nw.LazyFrame: _add_lagged_dataframe_columns,
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
    """Append a lag columns.

    Parameters
    ----------
    X : np.ndarray
        Data to be lagged.
    cols : int | List[int]
        Column index / indices.
    lags : int | List[int]
        The amount of lag for each col.
    drop_na : bool
        Whether or not to remove rows that contain NA values.

    Returns
    -------
    np.ndarray
        Array with concatenated lagged cols.
    """

    cols = as_list(cols)

    if not all([isinstance(col, int) for col in cols]):
        raise ValueError("Matrix columns are indexed by integers")

    if not all([col < X.shape[1] for col in cols]):
        raise KeyError("The column does not exist")

    combos = (shift(X[:, col], -lag, cval=np.nan) for col in cols for lag in lags)

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


def _add_lagged_dataframe_columns(df, cols, lags, drop_na):
    """Append a lag columns.

    Parameters
    ----------
    df : narwhals.DataFrame | narwhals.LazyFrame
        Data to be lagged.
    cols : str | List[str]
        Column name / names.
    lags : int | List[int]
        The amount of lag for each col.
    drop_na : bool
        Whether or not to remove rows that contain NA values.

    Returns
    -------
    DataFrame
        Dataframe with concatenated lagged cols.
    """

    cols = as_list(cols)

    if not all([col in df.columns for col in cols]):
        raise KeyError("The column does not exist")

    answer = df.with_columns(nw.col(col).shift(-lag).alias(col + str(lag)) for col in cols for lag in lags)

    # Remove rows that contain null values when drop_na is truthy
    if drop_na:
        answer = answer.drop_nulls()

    return nw.to_native(answer)
