import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted


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

    def __init__(
        self,
        quantile_range=(5.0, 95.0),
        interpolation="linear",
        discard_infs=False,
        copy=True,
    ):

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
        X = check_array(
            X, copy=True, force_all_finite=False, dtype=FLOAT_DTYPES, estimator=self
        )

        # If X contains infs, we need to replace them by nans before computing quantiles
        np.putmask(X, (X == np.inf) | (X == -np.inf), np.nan)

        # There should be no column containing only nan cells at this point. If that's not the case,
        # it means that the user asked ColumnCapper to fit some column containing only nan or inf cells.
        nans_mask = np.isnan(X)
        invalid_columns_mask = (
            nans_mask.sum(axis=0) == X.shape[0]
        )  # Contains as many nans as rows
        if invalid_columns_mask.any():
            raise ValueError(
                "ColumnCapper cannot fit columns containing only inf/nan values"
            )

        q = [quantile_limit / 100 for quantile_limit in self.quantile_range]
        self.quantiles_ = np.nanquantile(
            a=X, q=q, axis=0, overwrite_input=True, interpolation=self.interpolation
        )

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
        check_is_fitted(self, "quantiles_")
        X = check_array(
            X,
            copy=self.copy,
            force_all_finite=False,
            dtype=FLOAT_DTYPES,
            estimator=self,
        )

        if X.shape[1] != self.n_columns_:
            raise ValueError(
                "X must have the same number of columns in fit and transform"
            )

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
        if not isinstance(quantile_range, tuple) and not isinstance(
            quantile_range, list
        ):
            raise TypeError("quantile_range must be a tuple or a list")
        if len(quantile_range) != 2:
            raise ValueError(
                "quantile_range must contain 2 elements: min_quantile and max_quantile"
            )

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
        allowed_interpolations = ("linear", "lower", "higher", "midpoint", "nearest")
        if interpolation not in allowed_interpolations:
            raise ValueError(
                "Available interpolation methods: {}".format(
                    ", ".join(allowed_interpolations)
                )
            )
