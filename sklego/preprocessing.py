import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices, PatsyError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_random_state, check_is_fitted

from sklego.common import TrainOnlyTransformerMixin, as_list


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
        self.X_dtypes_ = X.dtypes
        self.feature_names_ = list(X.select_dtypes(include=self.include, exclude=self.exclude).columns)

        if len(self.type_columns_) == 0:
            raise ValueError(f'Provided type(s) results in empty dateframe')

        return self

    def get_feature_names(self, *args, **kwargs):
        return self.feature_names_

    def transform(self, X):
        """
        Transforms pandas dataframe by (de)selecting columns based on their dtype
        :param X: pandas dataframe to select dtypes for
        """
        check_is_fitted(self, ['type_columns_', 'X_dtypes_', 'feature_names_'])
        self._check_X_for_type(X)
        if (self.X_dtypes_ != X.dtypes).any():
            raise ValueError(f'Column dtypes were not equal during fit and transform. Fit types: \n'
                             f'{self.X_dtypes_}\n'
                             f'transform: \n'
                             f'{X.dtypes}')

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
        self.columns = as_list(columns)

    def fit(self, X, y=None):
        """
        Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """
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
            return X[self.columns]
        return X

    def get_feature_names(self):
        return self.columns

    def _check_column_length(self):
        """Check if no column is selected"""
        if len(self.columns) == 0:
            raise ValueError("Expected columns to be at least of length 1, found length of 0 instead")

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


class OrthogonalTransformer(BaseEstimator, TransformerMixin):
    """
    Transform the columns of a dataframe or numpy array to a column orthogonal or orthonormal matrix.
    Q, R such that X = Q*R, with Q orthogonal, from which follows Q = X*inv(R)
    :param normalize: whether orthogonal matrix should be orthonormal as well
    """

    def __init__(self, normalize=False):
        self.normalize = normalize

    def fit(self, X, y=None):
        """
        Store the inverse of R of the QR decomposition of X, which can be used to calculate the orthogonal projection
        of X. If normalization is required, also stores a vector with normalization terms
        """
        X = check_array(X, estimator=self)

        if not X.shape[0] > 1:
            raise ValueError("Orthogonal transformation not valid for one sample")

        # Q, R such that X = Q*R, with Q orthogonal, from which follows Q = X*inv(R)
        Q, R = np.linalg.qr(X)
        self.inv_R_ = np.linalg.inv(R)

        if self.normalize:
            self.normalization_vector_ = np.linalg.norm(Q, ord=2, axis=0)
        else:
            self.normalization_vector_ = np.ones((X.shape[1], ))

        return self

    def transform(self, X):
        """Transforms X using the fitted inverse of R. Normalizes the result if required"""
        if self.normalize:
            check_is_fitted(self, ['inv_R_', 'normalization_vector_'])
        else:
            check_is_fitted(self, ['inv_R_'])

        X = check_array(X, estimator=self)

        return X @ self.inv_R_ / self.normalization_vector_


def scalar_projection(vec, unto):
    return vec.dot(unto)/unto.dot(unto)


def vector_projection(vec, unto):
    return scalar_projection(vec, unto) * unto


class InformationFilter(BaseEstimator, TransformerMixin):
    """
    The `InformationFilter` uses a variant of the gram smidt process
    to filter information out of the dataset. This can be useful if you
    want to filter information out of a dataset because of fairness.
    To explain how it works: given a training matrix :math:`X` that contains
    columns :math:`x_1, ..., x_k`. If we assume columns :math:`x_1` and :math:`x_2`
    to be the sensitive columns then the information-filter will
    remove information by applying these transformations;
    .. math::
       \\begin{split}
       v_1 & = x_1 \\\\
       v_2 & = x_2 - \\frac{x_2 v_1}{v_1 v_1}\\\\
       v_3 & = x_3 - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}\\\\
       ... \\\\
       v_k & = x_k - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}
       \\end{split}
    Concatenating our vectors (but removing the sensitive ones) gives us
    a new training matrix :math:`X_{fair} =  [v_3, ..., v_k]`.
    :param columns: the columns to filter out this can be a sequence of either int
                    (in the case of numpy) or string (in the case of pandas).
    :param alpha: parameter to control how much to filter, for alpha=1 we filter out
                  all information while for alpha=0 we don't apply any.
    """

    def __init__(self, columns, alpha=1):
        self.columns = columns
        # sklearn does not allow `as_list` immediately because of cloning reasons
        self.cols = as_list(columns)
        self.alpha = alpha

    def _check_coltype(self, X):
        for col in self.cols:
            if isinstance(col, str):
                if isinstance(X, np.ndarray):
                    raise ValueError(f"column {col} is a string but datatype receive is numpy.")
                if isinstance(X, pd.DataFrame):
                    if col not in X.columns:
                        raise ValueError(f"column {col} is not in {X.columns}")
            if isinstance(col, int):
                if col not in range(np.atleast_2d(np.array(X)).shape[1]):
                    raise ValueError(f"column {col} is out of bounds for input shape {X.shape}")

    def _col_idx(self, X, name):
        if isinstance(name, str):
            if isinstance(X, np.ndarray):
                raise ValueError("You cannot have a column of type string on a numpy input matrix.")
            return {name: i for i, name in enumerate(X.columns)}[name]
        return name

    def _make_v_vectors(self, X, col_ids):
        vs = np.zeros((X.shape[0], len(col_ids)))
        for i, c in enumerate(col_ids):
            vs[:, i] = X[:, col_ids[i]]
            for j in range(0, i):
                vs[:, i] = vs[:, i] - vector_projection(vs[:, i], vs[:, j])
        return vs

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset orthogonal to sensitive columns."""
        self._check_coltype(X)
        self.col_ids_ = [v if isinstance(v, int) else self._col_idx(X, v) for v in self.cols]
        X = check_array(X, estimator=self)
        X_fair = X.copy()
        v_vectors = self._make_v_vectors(X, self.col_ids_)
        # gram smidt process but only on sensitive attributes
        for i, col in enumerate(X_fair.T):
            for v in v_vectors.T:
                X_fair[:, i] = X_fair[:, i] - vector_projection(X_fair[:, i], v)
        # we want to learn matrix P: X P = X_fair
        # this means we first need to create X_fair in order to learn P
        self.projection_, resid, rank, s = np.linalg.lstsq(X, X_fair, rcond=None)
        return self

    def transform(self, X):
        """Transforms X by applying the information filter."""
        check_is_fitted(self, ['projection_', 'col_ids_'])
        self._check_coltype(X)
        X = check_array(X, estimator=self)
        # apply the projection and remove the column we won't need
        X_fair = X @ self.projection_
        X_removed = np.delete(X_fair, self.col_ids_, axis=1)
        X_orig = np.delete(X, self.col_ids_, axis=1)
        return self.alpha * np.atleast_2d(X_removed) + (1 - self.alpha) * np.atleast_2d(X_orig)


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
        # if the columns parameter is not a list, make it into a list
        self.columns = as_list(columns)

    def fit(self, X, y=None):
        """
        Checks 1) if input is a DataFrame, and 2) if column names are in this DataFrame

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :param y: ``pd.Series`` labels for X. unused for column selection
        :returns: ``ColumnSelector`` object.
        """

        self._check_X_for_type(X)
        self._check_column_names(X)
        self.feature_names_ = list(X.drop(columns=self.columns).columns)
        self._check_column_length()
        return self

    def transform(self, X):
        """Returns a pandas DataFrame with only the specified columns

        :param X: ``pd.DataFrame`` on which we apply the column selection
        :returns: ``pd.DataFrame`` with only the selected columns
        """
        check_is_fitted(self, ['feature_names_'])
        self._check_X_for_type(X)
        if self.columns:
            return X.drop(columns=self.columns)
        return X

    def get_feature_names(self):
        return self.feature_names_

    def _check_column_length(self):
        """Check if all columns are droped"""
        if len(self.feature_names_) == 0:
            raise ValueError(f"Dropping {self.columns} would result in an empty output DataFrame")

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


class RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    """
    This is a transformer for features with some form of circularity.
    E.g. for days of the week you might face the problem that, conceptually, day 7 is as
    close to day 6 as it is to day 1. While numerically their distance is different.
    This transformer remedies that problem.
    The transformer selects a column and transforms it with a given number of repeating
    (radial) basis functions. Radial basis functions are bell-curve shaped functions
    which take the original data as input. The basis functions are equally spaced over
    the input range. The key feature of repeating basis funtions is that they are
    continuous when moving from the max to the min of the input range. As a result these
    repeating basis functions can capture how close each datapoint is to the center of
    each repeating basis function, even when the input data has a circular nature.

    :type column: int or list, default=0
    :param column: Indexes the data on its second axis. Integers are interpreted as
        positional columns, while strings can reference DataFrame columns by name.

    :type remainder: {'drop', 'passthrough'}, default="drop"
    :param remainder: By default, only the specified column is transformed, and the
        non-specified columns are dropped. (default of ``'drop'``). By specifying
        ``remainder='passthrough'``, all remaining columns will be automatically passed
        through. This subset of columns is concatenated with the output of the transformer.

    :type n_periods: int, default=12
    :param n_periods: number of basis functions to create, i.e., the number of columns that
        will exit the transformer.

    :type input_range: tuple or None, default=None
    :param input_range: the values at which the data repeats itself. For example, for days of
        the week this is (1,7). If input_range=None it is inferred from the training data.
    """

    def __init__(
        self, column=0, remainder="drop", n_periods=12, input_range=None
    ):
        self.column = column
        self.remainder = remainder
        self.n_periods = n_periods
        self.input_range = input_range

    def fit(self, X, y=None):
        self.pipeline_ = ColumnTransformer(
            [
                (
                    "repeatingbasis",
                    _RepeatingBasisFunction(
                        n_periods=self.n_periods, input_range=self.input_range
                    ),
                    [self.column],
                )
            ],
            remainder=self.remainder,
        )

        self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        check_is_fitted(self, ["pipeline_"])
        return self.pipeline_.transform(X)


class _RepeatingBasisFunction(TransformerMixin, BaseEstimator):
    def __init__(self, n_periods: int = 12, input_range=None):
        self.n_periods = n_periods
        self.input_range = input_range

    def fit(self, X, y=None):
        X = check_array(X, estimator=self)

        # find min and max for standardization if not given explicitly
        if self.input_range is None:
            self.input_range = (X.min(), X.max())

        # exclude the last value because it's identical to the first for repeating basis functions
        self.bases_ = np.linspace(0, 1, self.n_periods + 1)[:-1]

        # curves should narrower (wider) when we have more (fewer) basis functions
        self.width_ = 1 / self.n_periods

        return self

    def transform(self, X):
        X = check_array(X, estimator=self, ensure_2d=True)
        check_is_fitted(self, ["bases_", "width_"])
        # This transformer only accepts one feature as input
        if len(X.shape) == 1:
            raise ValueError(f"X should have exactly one column, it has: {X.shape[1]}")

        # MinMax Scale to 0-1
        X = (X - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

        base_distances = self._array_bases_distances(X, self.bases_)

        # apply rbf function to series for each basis
        return self._rbf(base_distances)

    def _array_base_distance(self, arr: np.ndarray, base: float) -> np.ndarray:
        """Calculates the distances between all array values and the base,
        where 0 and 1 are assumed to be at the same position"""
        abs_diff_0 = np.abs(arr - base)
        abs_diff_1 = 1 - abs_diff_0
        concat = np.concatenate(
            (abs_diff_0.reshape(-1, 1), abs_diff_1.reshape(-1, 1)), axis=1
        )
        final = concat.min(axis=1)
        return final

    def _array_bases_distances(self, array, bases):
        """Calculates the distances between all combinations of array and bases values"""
        array = array.reshape(-1, 1)
        bases = bases.reshape(1, -1)

        return np.apply_along_axis(
            lambda b: self._array_base_distance(array, base=b), axis=0, arr=bases
        )

    def _rbf(self, arr):
        return np.exp(-(arr / self.width_) ** 2)
