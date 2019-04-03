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
            
            
            

class ColumnsCapper(TransformerMixin):
    """Returns a dataframe with selected columns capped to a maximum threshold.
    the threshold can be set manually or a quantile can be used.
    
    param:
    X: `pd.DataFrame` to which to apply capping
    quantile_thresholds: `float or list of floats` if float (e.g. 0.9) is the quantile to use as threshold. if 
                            list, there is one value for each column to cap
    absolute_threshold: `float or list of floats` (e.g. 100) absolute value to use to cap values. if list, there is one
                            value for each column to cap
    cols_names: `list of column names or None`. if list, cap these columns. if None, caps all columns in dataframe
    use_quantile: `bool`. if true uses the quantile, otherwise uses the absolute threshold
    
    
    """
    
    def __init__(self, quantile_thresholds, cols_names=None, absolute_thresholds=None, use_quantile=True):
        self.cols_names = cols_names
        self.absolute_thresholds = absolute_thresholds
        self.quantile_thresholds = quantile_thresholds
        self.use_quantile = use_quantile
        
    def transform(self, X):
        if self.cols_names is None:
            self.cols_names = X.columns
        if self.use_quantile:
            if type(self.quantile_thresholds) is float:
                for col in self.cols_names:
                    q = X[col].quantile(self.quantile_thresholds, interpolation='linear')
                    X[col] = X[col].apply(lambda x: x if (x < q) else q)
            if type(self.quantile_thresholds) is list:
                for col, q_t in zip(self.cols_names, self.quantile_thresholds):
                    q = X[col].quantile(q=q_t, axis=1, interpolation='linear')
                    X[col] = X[col].apply(lambda x: x if (x < q) else q)
        else:
            if type(self.absolute_thresholds) is float:
                for col in self.cols_names:
                    X[col] = X[col].apply(lambda x: x if (x < self.absolute_thresholds) else self.absolute_thresholds)
            if type(self.absolute_thresholds) is list:
                for col, threshold in zip(self.cols_names, self.thresholds):
                    X[col] = X[col].apply(lambda x: x if (x < self.threshold) else self.threshold)
        return X
    
    def fit(self):
        return self
