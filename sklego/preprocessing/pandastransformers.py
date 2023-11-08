import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


class ColumnDropper(BaseEstimator, TransformerMixin):
    """The `ColumnDropper` transformer allows dropping specific columns from a pandas DataFrame by name.
    Can be useful in a sklearn Pipeline.

    Parameters
    ----------
    columns : str | list[str]
        Column name(s) to be selected.

    Attributes
    ----------
    feature_names_ : list[str]
        The names of the features to keep during transform.

    Examples
    --------
    ```py
    # Selecting a single column from a pandas DataFrame
    import pandas as pd
    from sklego.preprocessing import ColumnDropper

    df = pd.DataFrame({
        "name": ["Swen", "Victor", "Alex"],
        "length": [1.82, 1.85, 1.80],
        "shoesize": [42, 44, 45]
    })
    ColumnDropper(["name"]).fit_transform(df)
    '''
       length  shoesize
    0    1.82        42
    1    1.85        44
    2    1.80        45
    '''

    # Selecting multiple columns from a pandas DataFrame
    ColumnDropper(["length", "shoesize"]).fit_transform(df)
    '''
         name
    0    Swen
    1  Victor
    2    Alex
    '''

    # Selecting non-existent columns returns in a KeyError
    ColumnDropper(["weight"]).fit_transform(df)
    # Traceback (most recent call last):
    #     ...
    # KeyError: "['weight'] column(s) not in DataFrame"

    # How to use the ColumnSelector in a sklearn Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("select", ColumnDropper(["name", "shoesize"])),
        ("scale", StandardScaler()),
    ])
    pipe.fit_transform(df)
    # array([[-0.16222142],
    #        [ 1.29777137],
    #        [-1.13554995]])
    ```

    !!! warning

        - Raises a `TypeError` if input provided is not a DataFrame.
        - Raises a `ValueError` if columns provided are not in the input DataFrame.
    """

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer by storing the column names to keep during `.transform()` step.

        Checks:

        1. If input is a `pd.DataFrame` object
        2. If column names are in such DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnDropper
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        KeyError
            If one or more of the columns provided doesn't exist in the input DataFrame.
        ValueError
            If dropping the specified columns would result in an empty output DataFrame.
        """
        self.columns_ = as_list(self.columns)
        self._check_X_for_type(X)
        self._check_column_names(X)
        self.feature_names_ = X.columns.drop(self.columns_).tolist()
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.

        Returns
        -------
        pd.DataFrame
            The data with the specified columns dropped.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        """
        check_is_fitted(self, ["feature_names_"])
        self._check_X_for_type(X)
        if self.columns_:
            return X.drop(columns=self.columns_)
        return X

    def get_feature_names(self):
        """Alias for `.feature_names_` attribute"""
        return self.feature_names_

    def _check_column_length(self):
        """Check if all columns are dropped"""
        if len(self.feature_names_) == 0:
            raise ValueError(f"Dropping {self.columns_} would result in an empty output DataFrame")

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
    """The `PandasTypeSelector` transformer allows to select columns in a pandas DataFrame based on their type.
    Can be useful in a sklearn Pipeline.

    It uses
    [pandas.DataFrame.select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
    method.

    Parameters
    ----------
    include : scalar or list-like
        Column type(s) to be selected
    exclude : scalar or list-like
        Column type(s) to be excluded from selection

    Attributes
    ----------
    feature_names_ : list[str]
        The names of the features to keep during transform.
    X_dtypes_ : pd.Series
        The dtypes of the columns in the input DataFrame.

    !!! warning

        Raises a `TypeError` if input provided is not a DataFrame.
    """

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """Fit the transformer by saving the column names to keep during transform.

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : PandasTypeSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        ValueError
            If provided type(s) results in empty dataframe.
        """
        self._check_X_for_type(X)
        self.X_dtypes_ = X.dtypes
        self.feature_names_ = list(X.select_dtypes(include=self.include, exclude=self.exclude).columns)

        if len(self.feature_names_) == 0:
            raise ValueError("Provided type(s) results in empty dataframe")

        return self

    def get_feature_names(self, *args, **kwargs):
        """Alias for `.feature_names_` attribute"""
        return self.feature_names_

    def transform(self, X):
        """Returns a pandas DataFrame with columns (de)selected based on their dtype.

        Parameters
        ----------
        X : pd.DataFrame
            The data to select dtype for.

        Returns
        -------
        pd.DataFrame
            The data with the specified columns selected.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        ValueError
            If column dtypes were not equal during fit and transform.
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
    """The `ColumnSelector` transformer allows selecting specific columns from a pandas DataFrame by name.
    Can be useful in a sklearn Pipeline.

    Parameters
    ----------
    columns : str | list[str]
        Column name(s) to be selected.

    Attributes
    ----------
    columns_ : list[str]
        The names of the features to keep during transform.

    Examples
    --------
    ```py
    # Selecting a single column from a pandas DataFrame
    import pandas as pd
    from sklego.preprocessing import ColumnSelector

    df = pd.DataFrame({
        "name": ["Swen", "Victor", "Alex"],
        "length": [1.82, 1.85, 1.80],
        "shoesize": [42, 44, 45]
    })
    ColumnSelector(["length"]).fit_transform(df)
    '''
        length
    0    1.82
    1    1.85
    2    1.80
    '''

    # Selecting multiple columns from a pandas DataFrame
    ColumnSelector(["length", "shoesize"]).fit_transform(df)
    '''
       length  shoesize
    0    1.82        42
    1    1.85        44
    2    1.80        45
    '''

    # Selecting non-existent columns returns in a KeyError
    ColumnSelector(["weight"]).fit_transform(df)
    # Traceback (most recent call last):
    #     ...
    # KeyError: "['weight'] column(s) not in DataFrame"

    # How to use the ColumnSelector in a sklearn Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("select", ColumnSelector(["length"])),
        ("scale", StandardScaler()),
    ])
    pipe.fit_transform(df)
    # array([[-0.16222142],
    #        [ 1.29777137],
    #        [-1.13554995]])
    ```

    !!! warning

        Raises a `TypeError` if input provided is not a DataFrame.

        Raises a `ValueError` if columns provided are not in the input DataFrame.
    """

    def __init__(self, columns: list):
        # if the columns parameter is not a list, make it into a list
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer by storing the column names to keep during transform.

        Checks:

        1. If input is a `pd.DataFrame` object
        2. If column names are in such DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.
        y : pd.Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        KeyError
            If one or more of the columns provided doesn't exist in the input DataFrame.
        ValueError
            If dropping the specified columns would result in an empty output DataFrame.
        """
        self.columns_ = as_list(self.columns)
        self._check_X_for_type(X)
        self._check_column_length()
        self._check_column_names(X)
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The data on which we apply the column selection.

        Returns
        -------
        pd.DataFrame
            The data with the specified columns dropped.

        Raises
        ------
        TypeError
            If `X` is not a `pd.DataFrame` object.
        """
        self._check_X_for_type(X)
        if self.columns:
            return X[self.columns_]
        return X

    def get_feature_names(self):
        """Alias for `.columns_` attribute"""
        return self.columns_

    def _check_column_length(self):
        """Check if no column is selected"""
        if len(self.columns_) == 0:
            raise ValueError("Expected columns to be at least of length 1, found length of 0 instead")

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
