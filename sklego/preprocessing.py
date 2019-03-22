from operator import attrgetter
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PandasTypeSelector(BaseEstimator, TransformerMixin):
    """
    Select columns in a pandas dataframe based on their dtype

    :param include: types to be included in the dataframe
    :param exclude: types to be exluded in the dataframe
    """
    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """
        Saves the column names for check during transform

        :param X: pandas dataframe to select dtypes out of
        :param y: not used in this class
        """
        self._check_X_for_type(X)
        self.type_columns_ = list(X.select_dtypes(include=self.include, exclude=self.exclude))

        if len(self.type_columns_) == 0:
            raise ValueError(f'Provided type(s) results in empty dateframe')

        return self

    def transform(self, X):
        """
        Transforms pandas dataframe by (de)selecting columns based on their dtype

        :param X: pandas dataframe to select dtypes for
        """
        check_is_fitted(self, 'type_columns_')

        self._check_X_for_type(X)

        transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

        if set(list(transformed_df)) != set(self.type_columns_):
            raise ValueError(f'Columns were not equal during fit and transform')

        return transformed_df

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")


class DateTimeFeatures(TransformerMixin, BaseEstimator):
    """
    Add common time-series features to data frame in your scikit-learn pipeline.
    All time features supported by pandas.dt can be added (e.g. weekday, weekofyear, year,
    quarter, month, day, hour and minute. By default it will generate all the time features. They
    can be individually excluded.

    :param date_col: name of the date column (default= 'date')
    :param date_features: tuple of dt features

    """

    def __init__(self, date_col="date", date_features=("weekday", "weekofyear", "year", "quarter",
                                                       "month", "day", "hour", "minute")):

        self.date_col = date_col
        self.date_features = date_features

    def fit(self, X, y):
        _ = X.get(self.date_col,
                  f"Date column {self.date_col} doesn't exist. Please provide another one.")
        if isinstance(_, str):
            print(_)

        bad_date_features = [attr for attr in self.date_features if not hasattr(pd.Series.dt, attr)]
        if len(bad_date_features) >= 1:
            raise ValueError(f"Date feature "
                             f"{bad_date_features} "
                             f"doesn't exist")
        # TODO: check that column is date time, below doesn't work
        # assert isinstance(X[date_col], np.datetime64)
        return self

    def transform(self, X, y):
        return self._get_time_features(X)

    def _get_time_features(self, dataf: pd.DataFrame):
        time_attr = pd.DataFrame(attrgetter(*self.date_features)(dataf[self.date_col].dt)).T
        time_attr.columns = self.date_features
        return pd.concat([dataf, time_attr], axis=1)


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
