import pandas as pd


def negate_lags(lags):
    """
    Helper function, negates an int or list of ints.

    >>> negate_lags(3)
    [-3]

    >>> negate_lags([3])
    [-3]
    """

    def negate(x):
        return -x

    if isinstance(lags, int):
        return [negate(lags)]

    if hasattr(lags, '__iter__'):
        assert all(isinstance(x, int) for x in lags), "Must be list of ints"
        return list(map(negate, lags))


def expand_columns(cols):
    """
    Helper function, returns list of one or more columns
    from column or list of columns.

    >>> expand_columns('test')
    ['test']

    >>> expand_columns(['test1', 'test2'])
    ['test1', 'test2']
    """

    if isinstance(cols, str):
        return [cols]

    if hasattr(cols, '__iter__'):
        return cols


def add_lag(X, cols, lags):
    """
    Append a shifted column.

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

    lags = negate_lags(lags)

    # Check the X instance
    if isinstance(X, pd.core.frame.DataFrame):
        return add_lagged_pandas_columns(X, cols, lags)

    # If X can not be handled
    raise NotImplementedError("Please pass a pandas DataFrame")


def add_lagged_pandas_column(df, col, lag):
    """
    Return generator of each lag for col.
    """
    return df[col].shift(lag).rename(col + str(lag))


def add_lagged_pandas_columns(df, cols, lags):
    """
    Append a shifted column.
    """

    cols = expand_columns(cols)
    combos = [
        add_lagged_pandas_column(df, col, lag)
        for col in cols
        for lag in lags
    ]

    return pd.concat([df, *combos], axis=1).dropna()
