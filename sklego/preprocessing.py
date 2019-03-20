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

        :param X: pandas dataframe to select dtypes for
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
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Provided variable X is not of type pandas.DataFrame")
