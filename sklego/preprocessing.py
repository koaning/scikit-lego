from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class TimeSeriesFeatureAdder(TransformerMixin, BaseEstimator):
    """
    Add common time-series features to data frame in your scikit-learn pipeline.
    Currently, it returns the day of the week.

    :param date_col: name of the date column (default= 'date')
    :param make_dummies: boolean whether the days of week will be added as dummy variables or as
    one column. In case of one column, Mondays = 0 and Sundays = 6. (default= false).

    """

    def __init__(self, date_col="date", make_dummies=False):
        self.date_col = date_col
        self.make_dummies = make_dummies

    def fit(self, X, y):
        _ = X.get(self.date_col,
                  f"Date column {self.date_col} doesn't exist. Please provide another one.")
        if isinstance(_, str):
            print(_)

        # TODO: check that column is date time, below doesn't work
        # assert isinstance(X[date_col], np.datetime64)
        return self

    def transform(self, X, y):
        X = self._get_days_of_week(X, self.date_col)

        return X

    def _get_days_of_week(self, dataf: pd.DataFrame, date_col: str = "date"):
        """
        The day of the week with Monday=0, Sunday=6.

        :param dataf: data frame that contains a date column. (type pandas.DataFrame)
        :returns pandas.DataFrame with an extra column day_of_week
        """
        df_with_days_of_week = dataf.assign(day_of_week=lambda d: d[date_col].dt.dayofweek)
        if self.make_dummies:
            df_with_days_of_week = pd.get_dummies(df_with_days_of_week, columns=["day_of_week"],
                                                  prefix="day", dtype=int)

        return df_with_days_of_week


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Allows selecting specific columns from a pandas DataFrame by name. Can be useful in a sklearn Pipeline.
    :param columns: column name ``str`` or list of column names to be selected
    .. note::
        Raises a ``TypeError`` if input provided is not a DataFrame
        Raises a ``ValueError`` if columns provided are not in the input DataFrame
    :Example:
    >>> # Selecting a single column from a pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'name': ['Swen', 'Victor', 'Alex'],
    ...     'length': [1.82, 1.85, 1.80],
    ...     'shoesize': [42, 44, 45]
    ... })
    >>> ColumnSelector(['length']).fit_transform(df)
       length
    0    1.82
    1    1.85
    2    1.80
    >>> # Selecting multiple columns from a pandas DataFrame
    >>> ColumnSelector(['length', 'shoesize']).fit_transform(df)
       length  shoesize
    0    1.82        42
    1    1.85        44
    2    1.80        45
    >>> # Selecting non-existent columns returns in a KeyError
    >>> ColumnSelector(['weight']).fit_transform(df)
    Traceback (most recent call last):
        ...
    KeyError: "['weight'] column(s) not in DataFrame"
    >>> # How to use the ColumnSelector in a sklearn Pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('select', ColumnSelector(['length'])),
    ...     ('scale', StandardScaler()),
    ... ])
    >>> pipe.fit_transform(df)
    array([[-0.16222142],
           [ 1.29777137],
           [-1.13554995]])
    """

    def __init__(self, columns: list):
        # if the columns parameter is not a list, make it into a list
        if not isinstance(columns, list):
            columns = [columns]

        self.columns = columns

    def fit(self, X, y=None):
        """
        Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame
        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """

        self._check_X_for_type(X)
        self._check_column_names(X)
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns
        :param X: ``pd.DataFrame`` on which we apply the column selection
        :returns: ``pd.DataFrame`` with only the selected columns
        """
        if self.columns:
            return X[self.columns]
        return X

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
        non_existent_columns = set(self.columns).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f'{list(non_existent_columns)} column(s) not in DataFrame')

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")
