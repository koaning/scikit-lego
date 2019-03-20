import pandas as pd


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

    if isinstance(lags, int):
        return [negate(lags)]

    if hasattr(lags, '__iter__'):
        assert all(isinstance(x, int) for x in lags), "Must be list of ints"
        return list(map(negate, lags))


def _expand_columns(cols):
    """
    Helper function, returns list of one or more columns
    from column or list of columns.

    :param cols: an iterable of column names or a single column name string.
    :returns: an iterable of column names.

    :Example:

    >>> _expand_columns('test')
    ['test']

    >>> _expand_columns(['test1', 'test2'])
    ['test1', 'test2']
    """

    if isinstance(cols, str):
        return [cols]

    if hasattr(cols, '__iter__'):
        return cols


def add_lag(X, cols, lags):
    """
    Appends lag column(s).

    :param X: array-like, shape=(n_columns, n_samples,) training data.
    :param cols: an iterable of column names or a single column name string.
    :param lags: iterable of ints or int.
    :returns: ``pd.DataFrame`` with only the selected columns

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
    """

    lags = _negate_lags(lags)

    # Check the X instance of pandas DataFrame
    if isinstance(X, pd.core.frame.DataFrame):
        return _add_lagged_pandas_columns(X, cols, lags)

    raise ValueError("Please pass a pandas DataFrame")


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
    KeyError
    """

    cols = _expand_columns(cols)

    if not all([col in df.columns.values for col in cols]):
        raise KeyError

    combos = [
        df[col].shift(lag).rename(col + str(lag))
        for col in cols
        for lag in lags
    ]

    return pd.concat([df, *combos], axis=1).dropna()
