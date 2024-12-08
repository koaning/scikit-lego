from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v1 as nw
from narwhals.dependencies import get_pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklego.common import as_list


def _nw_match_dtype(dtype, selection):
    if selection == "number":
        return dtype in (
            nw.Int64,
            nw.Int32,
            nw.Int16,
            nw.Int8,
            nw.UInt64,
            nw.UInt32,
            nw.UInt16,
            nw.UInt8,
            nw.Float64,
            nw.Float32,
        )
    if selection == "bool":
        return dtype == nw.Boolean
    if selection == "string":
        return dtype == nw.String
    if selection == "category":
        return dtype == nw.Categorical
    msg = f"Expected {{'number', 'bool', 'string', 'category'}}, got: {selection}, which is not (yet!) supported."
    raise ValueError(msg)


def _nw_select_dtypes(include: str | list[str], exclude: str | list[str], schema: dict[str, Any]):
    if not include and not exclude:
        raise ValueError("Must provide at least one of `include` or `exclude`")

    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    exclude = exclude or []

    if include:
        feature_names = [
            name
            for name, dtype in schema.items()
            if any(_nw_match_dtype(dtype, _include) for _include in include)
            and not any(_nw_match_dtype(dtype, _exclude) for _exclude in exclude)
        ]
    else:
        feature_names = [
            name for name, dtype in schema.items() if not any(_nw_match_dtype(dtype, _exclude) for _exclude in exclude)
        ]
    return feature_names


class ColumnDropper(TransformerMixin, BaseEstimator):
    """The `ColumnDropper` transformer allows dropping specific columns from a DataFrame by name.
    Can be useful in a sklearn Pipeline.

    Parameters
    ----------
    columns : str | list[str]
        Column name(s) to be selected.

    Attributes
    ----------
    feature_names_ : list[str]
        The names of the features to keep during transform.

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

    # Dropping multiple columns from a pandas DataFrame
    ColumnDropper(["length", "shoesize"]).fit_transform(df)
    '''
         name
    0    Swen
    1  Victor
    2    Alex
    '''

    # Dropping non-existent columns results in a KeyError
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

    Raises
    ------
    TypeError
        If input provided is not a DataFrame.
    KeyError
        If columns provided are not in the input DataFrame.
    ValueError
        If dropping the specified columns would result in an empty output DataFrame.
    """

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer by storing the column names to keep during `.transform()` step.

        Checks:

        1. If input is a supported DataFrame
        2. If column names are in such DataFrame

        Parameters
        ----------
        X : DataFrame
            The data on which we apply the column selection.
        y : Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnDropper
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame.
        KeyError
            If one or more of the columns provided doesn't exist in the input DataFrame.
        ValueError
            If dropping the specified columns would result in an empty output DataFrame.
        """
        self.columns_ = as_list(self.columns)
        X = nw.from_native(X)
        self._check_column_names(X)
        self.feature_names_ = [x for x in X.columns if x not in self.columns_]
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a DataFrame with only the specified columns.

        Parameters
        ----------
        X : DataFrame
            The data on which we apply the column selection.

        Returns
        -------
        DataFrame
            The data with the specified columns dropped.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame object.
        """
        check_is_fitted(self, ["feature_names_"])
        X = nw.from_native(X)
        if self.columns_:
            return nw.to_native(X.drop(self.columns_))
        return nw.to_native(X)

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


class TypeSelector(TransformerMixin, BaseEstimator):
    """The `TypeSelector` transformer allows to select columns in a DataFrame based on their type.
    Can be useful in a sklearn Pipeline.

    - For pandas, it uses
      [pandas.DataFrame.select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
      method.
    - For non-pandas dataframes (e.g. Polars), the following inputs are allowed:

        - 'number'
        - 'string'
        - 'bool'
        - 'category'

    !!! info "New in version 0.9.0"

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
    X_dtypes_ : Series | dict[str, DType]
        The dtypes of the columns in the input DataFrame.

    !!! warning

        Raises a `TypeError` if input provided is not a DataFrame.

    Examples
    --------
    ```py
    import pandas as pd
    from sklego.preprocessing import TypeSelector

    df = pd.DataFrame({
        "name": ["Swen", "Victor", "Alex"],
        "length": [1.82, 1.85, 1.80],
        "shoesize": [42, 44, 45]
    })

    #Excluding single column
    TypeSelector(exclude="int64").fit_transform(df)
    #	name	length
    #0	Swen	1.82
    #1	Victor	1.85
    #2	Alex	1.80

    #Including multiple columns
    TypeSelector(include=["int64", "object"]).fit_transform(df)
    #	name	shoesize
    #0	Swen	42
    #1	Victor	44
    #2	Alex	45
    ```
    """

    def __init__(self, include=None, exclude=None):
        self.include = include
        self.exclude = exclude

    def fit(self, X, y=None):
        """Fit the transformer by saving the column names to keep during transform.

        Parameters
        ----------
        X : DataFrame
            The data on which we apply the column selection.
        y : Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : TypeSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame.
        ValueError
            If provided type(s) results in empty dataframe.
        """
        if (pd := get_pandas()) is not None and isinstance(X, pd.DataFrame):
            self.X_dtypes_ = X.dtypes
            self.feature_names_ = list(X.select_dtypes(include=self.include, exclude=self.exclude).columns)
        else:
            X = nw.from_native(X)
            self.X_dtypes_ = X.schema
            self.feature_names_ = _nw_select_dtypes(include=self.include, exclude=self.exclude, schema=self.X_dtypes_)

        if len(self.feature_names_) == 0:
            raise ValueError("Provided type(s) results in empty dataframe")

        return self

    def get_feature_names(self, *args, **kwargs):
        """Alias for `.feature_names_` attribute"""
        return self.feature_names_

    def transform(self, X):
        """Returns a DataFrame with columns (de)selected based on their dtype.

        Parameters
        ----------
        X : DataFrame
            The data to select dtype for.

        Returns
        -------
        DataFrame
            The data with the specified columns selected.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame.
        ValueError
            If column dtypes were not equal during fit and transform.
        """
        check_is_fitted(self, ["X_dtypes_", "feature_names_"])

        if (pd := get_pandas()) is not None and isinstance(X, pd.DataFrame):
            try:
                if (self.X_dtypes_ != X.dtypes).any():
                    raise ValueError(
                        f"Column dtypes were not equal during fit and transform. Fit types: \n"
                        f"{self.X_dtypes_}\n"
                        f"transform: \n"
                        f"{X.dtypes}"
                    )
            except ValueError as e:
                raise ValueError("Column dtypes were not equal during fit and transform") from e
            transformed_df = X.select_dtypes(include=self.include, exclude=self.exclude)
        else:
            X = nw.from_native(X)
            X_schema = X.schema
            if self.X_dtypes_ != X_schema:
                raise ValueError(
                    f"Column dtypes were not equal during fit and transform. Fit types: \n"
                    f"{self.X_dtypes_}\n"
                    f"transform: \n"
                    f"{X.schema}"
                )
            transformed_df = X.select(
                _nw_select_dtypes(include=self.include, exclude=self.exclude, schema=X_schema)
            ).pipe(nw.to_native)

        return transformed_df


class PandasTypeSelector(TypeSelector):
    """
    !!! warning "Deprecated since version 0.9.0, please use TypeSelector instead"
    """

    def __init__(self, include=None, exclude=None):
        warnings.warn(
            "PandasTypeSelector is deprecated and will be removed in a future version. "
            "Please use `from sklego.preprocessing import TypeSelector` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(include=include, exclude=exclude)


class ColumnSelector(TransformerMixin, BaseEstimator):
    """The `ColumnSelector` transformer allows selecting specific columns from a DataFrame by name.
    Can be useful in a sklearn Pipeline.

    Parameters
    ----------
    columns : str | list[str]
        Column name(s) to be selected.

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

    df_pd = pd.DataFrame({
        "name": ["Swen", "Victor", "Alex"],
        "length": [1.82, 1.85, 1.80],
        "shoesize": [42, 44, 45]
    })

    ColumnSelector(["length"]).fit_transform(df_pd)
    '''
        length
    0    1.82
    1    1.85
    2    1.80
    '''

    # Selecting multiple columns from a polars DataFrame
    import polars as pl
    from sklego.preprocessing import ColumnSelector

    df_pl = pl.DataFrame({
        "name": ["Swen", "Victor", "Alex"],
        "length": [1.82, 1.85, 1.80],
        "shoesize": [42, 44, 45]
    })

    ColumnSelector(["length", "shoesize"]).fit_transform(df_pl)
    '''
    shape: (3, 2)
    ┌────────┬──────────┐
    │ length ┆ shoesize │
    │ ---    ┆ ---      │
    │ f64    ┆ i64      │
    ╞════════╪══════════╡
    │ 1.82   ┆ 42       │
    │ 1.85   ┆ 44       │
    │ 1.8    ┆ 45       │
    └────────┴──────────┘
    '''


    # Selecting non-existent columns results in a KeyError
    ColumnSelector(["weight"]).fit_transform(df_pd)
    # Traceback (most recent call last):
    #     ...
    # KeyError: "['weight'] column(s) not in DataFrame"

    # How to use the ColumnSelector in a sklearn Pipeline
    import polars as pl
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklego.preprocessing import ColumnSelector

    pipe = Pipeline([
        ("select", ColumnSelector(["length"])),
        ("scale", StandardScaler()),
    ])
    pipe.fit_transform(df_pl)
    # array([[-0.16222142],
    #        [ 1.29777137],
    #        [-1.13554995]])
    ```

    Raises
    ------
    TypeError
        If input provided is not a supported DataFrame.
    KeyError
        If columns provided are not in the input DataFrame.
    ValueError
        If provided list of columns to select is empty and would result in an empty output DataFrame.
    """

    def __init__(self, columns: list):
        # if the columns parameter is not a list, make it into a list
        self.columns = columns

    def fit(self, X, y=None):
        """Fit the transformer by storing the column names to keep during transform.

        Checks:

        1. If input is a supported DataFrame
        2. If column names are in such DataFrame

        Parameters
        ----------
        X : DataFrame
            The data on which we apply the column selection.
        y : Series, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnSelector
            The fitted transformer.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame
        KeyError
            If one or more of the columns provided doesn't exist in the input DataFrame.
        ValueError
            If provided list of columns to select is empty and would result in an empty output DataFrame.
        """
        self.columns_ = as_list(self.columns)
        X = nw.from_native(X)
        self._check_column_names(X)
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a DataFrame with only the specified columns.

        Parameters
        ----------
        X : DataFrame
            The data on which we apply the column selection.

        Returns
        -------
        DataFrame
            The data with the specified columns dropped.

        Raises
        ------
        TypeError
            If `X` is not a supported DataFrame.
        """
        X = nw.from_native(X)
        if self.columns:
            return nw.to_native(X.select(self.columns_))
        return nw.to_native(X)

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
