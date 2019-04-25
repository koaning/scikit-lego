import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices, PatsyError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_random_state, check_is_fitted

from sklego.common import TrainOnlyTransformerMixin


class RandomAdder(TrainOnlyTransformerMixin, BaseEstimator):
    def __init__(self, noise=1, random_state=None):
        self.noise = noise
        self.random_state = random_state

    def fit(self, X, y):
        super().fit(X, y)
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = X.shape[1]

        return self

    def transform_train(self, X):
        rs = check_random_state(self.random_state)
        check_is_fitted(self, ['dim_'])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        return X + rs.normal(0, self.noise, size=X.shape)


class PatsyTransformer(TransformerMixin, BaseEstimator):
    """
    The patsy transformer offers a method to select the right columns
    from a dataframe as well as a DSL for transformations. It is inspired
    from R formulas.

    This is can be useful as a first step in the pipeline.

    :param formula: a patsy-compatible formula
    """

    def __init__(self, formula):
        self.formula = formula

    def fit(self, X, y=None):
        """Fits the estimator"""
        X_ = dmatrix(self.formula, X)
        assert np.array(X_).shape[0] == np.array(X).shape[0]
        self.design_info_ = X_.design_info
        return self

    def transform(self, X):
        """
        Applies the formula to the matrix/dataframe X.

        Returns an design array that can be used in sklearn pipelines.
        """
        check_is_fitted(self, 'design_info_')
        try:
            return build_design_matrices([self.design_info_], X)[0]
        except PatsyError as e:
            raise RuntimeError from e


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


class ColumnCapper(TransformerMixin, BaseEstimator):
    """
    Caps the values of columns according to the given quantile thresholds.

    :type min_quantile: float, optional, default=0.05
    :param min_quantile: The minimum quantile to compute the lowerbound of the transformed
        columns. Must be in the interval [0; 1].

    :type max_quantile: float, optional, default=0.95
    :param max_quantile: The maximum quantile to compute the upperbound of the transformed
        columns. Must be in the interval [0; 1].

    :type discard_infs: bool, optional, default=False
    :param discard_infs: Whether to discard ``-np.inf`` and ``np.inf`` values or not. If
        ``False``, such values will be capped. If ``True``, they will be replaced by
        ``np.nan``.

        .. note::
            Setting ``discard_infs=True`` is important if the `inf` values are results
            of divisions by 0, which are interpreted by ``pandas`` as ``-np.inf`` or
            ``np.inf`` depending on the signal of the numerator.

    :raises:
        ``TypeError`` if the quantiles are not numbers

        ``ValueError`` if ``min_quantile`` > ``max_quantile`` or if the quantiles are
        not in the interval [0; 1]

    :Example:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklego.preprocessing import ColumnCapper
    >>> df = pd.DataFrame({'a':[1, 2, 3, 4], 'b':[11, 12, np.inf, 14]})
    >>> df
       a          b
    0  1  11.000000
    1  2  12.000000
    2  3        inf
    3  4  14.000000
    >>> capper = ColumnCapper(min_quantile=.05, max_quantile=.9, discard_infs=False)
    >>> df['a_capped'] = capper.fit_transform(df['a'])
    >>> df['b_capped'] = capper.fit_transform(df['b'])
    >>> df
       a          b  a_capped  b_capped
    0  1  11.000000      1.15      11.1
    1  2  12.000000      2.00      12.0
    2  3        inf      3.00      13.6
    3  4  14.000000      3.70      13.6
    >>> capper = ColumnCapper(discard_infs=True) # Discarding infs
    >>> df[['a', 'b']] = capper.fit_transform(df[['a', 'b']]) # Transforming multiple columns
    >>> df
          a     b  a_capped  b_capped
    0  1.15  11.1      1.15      11.1
    1  2.00  12.0      2.00      12.0
    2  3.00   NaN      3.00      13.6
    3  3.85  13.8      3.70      13.6
    """
    def __init__(self, min_quantile=0.05, max_quantile=0.95, discard_infs=False):

        self._check_quantiles(min_quantile, max_quantile)

        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.discard_infs = discard_infs

    def fit(self, X, y=None):
        """
        Computes the quantiles for each column of ``X``.

        :type X: pandas.DataFrame, pandas.Series, numpy.ndarray or list
        :param X: The column(s) from which the capping limit(s) will be computed.

        :param y: Ignored.

        :rtype: sklego.preprocessing.ColumnCapper
        :returns: The fitted object.

        :raises:
            ``TypeError`` if ``X`` is not an object of :class:`pandas.DataFrame`,
            :class:`pandas.Series`, :class:`numpy.ndarray` or :class:`list`

            ``ValueError`` if ``X`` contains non-numeric columns
        """
        X = self._check_X_and_convert_to_pandas(X)

        # Saving the number of columns to ensure coherence between fit and transform inputs
        self._n_columns = X.shape[1]

        # Making sure that the magnitudes of -np.inf and np.inf won't cause any trouble
        X = X[(-np.inf < X) & (X < np.inf)]

        # Computing the quantiles for each column of X
        self._quantiles = X.quantile([self.min_quantile, self.max_quantile])

        return self

    def transform(self, X):
        """
        Performs the capping on the column(s) of ``X``.

        :type X: pandas.DataFrame, pandas.Series, numpy.ndarray or list
        :param X: The column(s) for which the capping limit(s) will be applied.

        :rtype: numpy.ndarray
        :returns: A copy of ``X`` with capped limits.

        :raises:
            ``TypeError`` if ``X`` is not an object of :class:`pandas.DataFrame`,
            :class:`pandas.Series`, :class:`numpy.ndarray` or :class:`list`

            ``ValueError`` if ``X`` contains non-numeric columns or if the number of
            columns from ``X`` differs from the number of columns when fitting
        """
        check_is_fitted(self, '_n_columns')
        X = self._check_X_and_convert_to_pandas(X)

        if X.shape[1] != self._n_columns:
            raise ValueError("Reshape your data. X must have the same number of "
                             + "columns in fit and transform")

        if self.discard_infs:
            X.replace([np.inf, -np.inf], [np.nan, np.nan], inplace=True)

        # Actually capping
        for column in X.columns:
            min_value, max_value = self._quantiles[column]
            X.loc[X[column] < min_value, column] = min_value
            X.loc[X[column] > max_value, column] = max_value

        if X.shape[1] == 1:
            return X.values[:, 0]
        return X.values

    @staticmethod
    def _check_quantiles(min_quantile, max_quantile):
        """
        Checks for the validity of min_quantile and max_quantile:

        * They must be numbers (int of float)
        * They must be in the interval [0; 1]
        * `min_quantile` must be less than or equal to `max_quantile`
        """
        for quantile in min_quantile, max_quantile:
            if not isinstance(quantile, float) and not isinstance(quantile, int):
                raise TypeError("min_quantile and max_quantile must be numbers")
            if quantile < 0 or 1 < quantile:
                raise ValueError("min_quantile and max_quantile must be in [0; 1]")

        if min_quantile > max_quantile:
            raise ValueError("min_quantile must be less than or equal to max_quantile")

    def _check_X_and_convert_to_pandas(self, X):
        """
        Creates a copy of `X` as a pandas.DataFrame object for safety reasons and to gain
        access to `replace`, `quantile` and `loc` attributes.

        The columns names are reset for compatibility purposes between `X` in fit and `X`
        in transform, which can be either pandas.DataFrame, pandas.Series or numpy.ndarray
        objects (not necessarily the same types on both methods).

        This method also checks if `X` is compatible with the requirements of ColumnCapper.
        """
        if isinstance(X, list):
            X = np.array(X)
        X = check_array(X, copy=True, force_all_finite=False, ensure_2d=False, dtype=FLOAT_DTYPES, estimator=self)
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = pd.DataFrame(X.values)
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            raise TypeError("Provided variable X must be of type pandas.DataFrame, "
                            + "pandas.Series, numpy.ndarray or list")
        return X
