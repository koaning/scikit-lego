import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Allows dropping specific columns from a pandas DataFrame by name. Can be useful in a sklearn Pipeline.

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
    >>> ColumnDropper(['name']).fit_transform(df)
       length  shoesize
    0    1.82        42
    1    1.85        44
    2    1.80        45

    >>> # Selecting multiple columns from a pandas DataFrame
    >>> ColumnDropper(['length', 'shoesize']).fit_transform(df)
         name
    0    Swen
    1  Victor
    2    Alex


    >>> # Selecting non-existent columns returns in a KeyError
    >>> ColumnDropper(['weight']).fit_transform(df)
    Traceback (most recent call last):
        ...
    KeyError: "['weight'] column(s) not in DataFrame"

    >>> # How to use the ColumnSelector in a sklearn Pipeline
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('select', ColumnDropper(['name', 'shoesize'])),
    ...     ('scale', StandardScaler()),
    ... ])
    >>> pipe.fit_transform(df)
    array([[-0.16222142],
           [ 1.29777137],
           [-1.13554995]])
    """

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        """
        Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """
        self.columns_ = as_list(self.columns)
        self._check_X_for_type(X)
        self._check_column_names(X)
        self.feature_names_ = list(X.drop(columns=self.columns_).columns)
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :returns: ``pd.DataFrame`` with only the selected columns
        """
        check_is_fitted(self, ["feature_names_"])
        self._check_X_for_type(X)
        if self.columns_:
            return X.drop(columns=self.columns_)
        return X

    def get_feature_names(self):
        return self.feature_names_

    def _check_column_length(self):
        """Check if all columns are dropped"""
        if len(self.feature_names_) == 0:
            raise ValueError(
                f"Dropping {self.columns_} would result in an empty output DataFrame"
            )

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
        non_existent_columns = set(self.columns_).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")


class PandasTypeSelector(BaseEstimator, TransformerMixin):
    """
    Select columns in a pandas dataframe based on their dtype

    :param include: types to be included in the dataframe
    :param exclude: types to be excluded in the dataframe
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
        self.X_dtypes_ = X.dtypes
        self.feature_names_ = list(
            X.select_dtypes(include=self.include, exclude=self.exclude).columns
        )

        if len(self.feature_names_) == 0:
            raise ValueError("Provided type(s) results in empty dateframe")

        return self

    def get_feature_names(self, *args, **kwargs):
        return self.feature_names_

    def transform(self, X):
        """
        Transforms pandas dataframe by (de)selecting columns based on their dtype
        :param X: pandas dataframe to select dtypes for
        """
        check_is_fitted(self, ["X_dtypes_", "feature_names_"])

        try:
            if (self.X_dtypes_ != X.dtypes).any():
                raise ValueError(
                    f"Column dtypes were not equal during fit and transform. Fit types: \n"
                    f"{self.X_dtypes_}\n"
                    f"transform: \n"
                    f"{X.dtypes}"
                )
        except ValueError as e:
            raise ValueError("Columns were not equal during fit and transform") from e

        self._check_X_for_type(X)
        transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)

        return transformed_df

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")


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
        self.columns = columns

    def fit(self, X, y=None):
        """
        Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """
        self.columns_ = as_list(self.columns)
        self._check_X_for_type(X)
        self._check_column_length()
        self._check_column_names(X)
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :returns: ``pd.DataFrame`` with only the selected columns
        """
        self._check_X_for_type(X)
        if self.columns:
            return X[self.columns_]
        return X

    def get_feature_names(self):
        return self.columns_

    def _check_column_length(self):
        """Check if no column is selected"""
        if len(self.columns_) == 0:
            raise ValueError(
                "Expected columns to be at least of length 1, found length of 0 instead"
            )

    def _check_column_names(self, X):
        """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
        non_existent_columns = set(self.columns_).difference(X.columns)
        if len(non_existent_columns) > 0:
            raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")

    @staticmethod
    def _check_X_for_type(X):
        """Checks if input of the Selector is of the required dtype"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")
