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

    :type quantile_range: tuple or list, optional, default=(5.0, 95.0)
    :param quantile_range: The quantile ranges to perform the capping. Their valus must
        be in the interval [0; 100].

    :type interpolation: str, optional, default='linear'
    :param interpolation: The interpolation method to compute the quantiles when the
        desired quantile lies between two data points `i` and `j`. The Available values
        are:

        * ``'linear'``: `i + (j - i) * fraction`, where `fraction` is the fractional part of\
            the index surrounded by `i` and `j`.
        * ``'lower'``: `i`.
        * ``'higher'``: `j`.
        * ``'nearest'``: `i` or `j` whichever is nearest.
        * ``'midpoint'``: (`i` + `j`) / 2.

    :type discard_infs: bool, optional, default=False
    :param discard_infs: Whether to discard ``-np.inf`` and ``np.inf`` values or not. If
        ``False``, such values will be capped. If ``True``, they will be replaced by
        ``np.nan``.

        .. note::
            Setting ``discard_infs=True`` is important if the `inf` values are results
            of divisions by 0, which are interpreted by ``pandas`` as ``-np.inf`` or
            ``np.inf`` depending on the signal of the numerator.

    :type copy: bool, optional, default=True
    :param copy: If False, try to avoid a copy and do inplace capping instead. This is not
        guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse
        CSR matrix, a copy may still be returned.

    :raises:
        ``TypeError``, ``ValueError``

    :Example:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklego.preprocessing import ColumnCapper
    >>> df = pd.DataFrame({'a':[2, 4.5, 7, 9], 'b':[11, 12, np.inf, 14]})
    >>> df
         a     b
    0  2.0  11.0
    1  4.5  12.0
    2  7.0   inf
    3  9.0  14.0
    >>> capper = ColumnCapper()
    >>> capper.fit_transform(df)
    array([[ 2.375, 11.1  ],
           [ 4.5  , 12.   ],
           [ 7.   , 13.8  ],
           [ 8.7  , 13.8  ]])
    >>> capper = ColumnCapper(discard_infs=True) # Discarding infs
    >>> df[['a', 'b']] = capper.fit_transform(df)
    >>> df
           a     b
    0  2.375  11.1
    1  4.500  12.0
    2  7.000   NaN
    3  8.700  13.8
    """
    def __init__(self, quantile_range=(5.0, 95.0), interpolation='linear', discard_infs=False, copy=True):

        self._check_quantile_range(quantile_range)
        self._check_interpolation(interpolation)

        self.quantile_range = quantile_range
        self.interpolation = interpolation
        self.discard_infs = discard_infs
        self.copy = copy

    def fit(self, X, y=None):
        """
        Computes the quantiles for each column of ``X``.

        :type X: pandas.DataFrame or numpy.ndarray
        :param X: The column(s) from which the capping limit(s) will be computed.

        :param y: Ignored.

        :rtype: sklego.preprocessing.ColumnCapper
        :returns: The fitted object.

        :raises:
            ``ValueError`` if ``X`` contains non-numeric columns
        """
        X = check_array(X, copy=True, force_all_finite=False, dtype=FLOAT_DTYPES, estimator=self)

        # If X contains infs, we need to replace them by nans before computing quantiles
        np.putmask(X, (X == np.inf) | (X == -np.inf), np.nan)

        # There should be no column containing only nan cells at this point. If that's not the case,
        # it means that the user asked ColumnCapper to fit some column containing only nan or inf cells.
        nans_mask = np.isnan(X)
        invalid_columns_mask = nans_mask.sum(axis=0) == X.shape[0]  # Contains as many nans as rows
        if invalid_columns_mask.any():
            raise ValueError("ColumnCapper cannot fit columns containing only inf/nan values")

        q = [quantile_limit/100 for quantile_limit in self.quantile_range]
        self.quantiles_ = np.nanquantile(a=X, q=q, axis=0, overwrite_input=True,
                                         interpolation=self.interpolation)

        # Saving the number of columns to ensure coherence between fit and transform inputs
        self.n_columns_ = X.shape[1]

        return self

    def transform(self, X):
        """
        Performs the capping on the column(s) of ``X``.

        :type X: pandas.DataFrame or numpy.ndarray
        :param X: The column(s) for which the capping limit(s) will be applied.

        :rtype: numpy.ndarray
        :returns: ``X`` values with capped limits.

        :raises:
            ``ValueError`` if the number of columns from ``X`` differs from the
            number of columns when fitting
        """
        check_is_fitted(self, 'quantiles_')
        X = check_array(X, copy=self.copy, force_all_finite=False, dtype=FLOAT_DTYPES, estimator=self)

        if X.shape[1] != self.n_columns_:
            raise ValueError("X must have the same number of columns in fit and transform")

        if self.discard_infs:
            np.putmask(X, (X == np.inf) | (X == -np.inf), np.nan)

        # Actually capping
        X = np.minimum(X, self.quantiles_[1, :])
        X = np.maximum(X, self.quantiles_[0, :])

        return X

    @staticmethod
    def _check_quantile_range(quantile_range):
        """
        Checks for the validity of quantile_range.
        """
        if not isinstance(quantile_range, tuple) and not isinstance(quantile_range, list):
            raise TypeError("quantile_range must be a tuple or a list")
        if len(quantile_range) != 2:
            raise ValueError("quantile_range must contain 2 elements: min_quantile and max_quantile")

        min_quantile, max_quantile = quantile_range

        for quantile in min_quantile, max_quantile:
            if not isinstance(quantile, float) and not isinstance(quantile, int):
                raise TypeError("min_quantile and max_quantile must be numbers")
            if quantile < 0 or 100 < quantile:
                raise ValueError("min_quantile and max_quantile must be in [0; 100]")

        if min_quantile > max_quantile:
            raise ValueError("min_quantile must be less than or equal to max_quantile")

    @staticmethod
    def _check_interpolation(interpolation):
        """
        Checks for the validity of interpolation
        """
        allowed_interpolations = ('linear', 'lower', 'higher', 'midpoint', 'nearest')
        if interpolation not in allowed_interpolations:
            raise ValueError("Available interpolation methods: {}".format(', '.join(allowed_interpolations)))
