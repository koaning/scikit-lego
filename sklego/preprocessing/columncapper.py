from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class ColumnCapper(TransformerMixin, BaseEstimator):
    """The `ColumnCapper` transformer caps the values of columns according to the given quantile thresholds.

    The capping is performed independently for each column of the input data. The quantile thresholds are computed
    during the fitting phase. The capping is performed during the transformation phase.

    Parameters
    ----------
    quantile_range : Tuple[float, float] | List[float], default=(5.0, 95.0)
        The quantile ranges to perform the capping. Their values must be in the interval [0; 100].
    interpolation : Literal["linear", "lower", "higher", "midpoint", "nearest"], default="linear"
        The interpolation method to compute the quantiles when the desired quantile lies between two data points `i`
        and `j`. This value is passed to
        [`numpy.nanquantile`](https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html) function.

        The Available values are:

        - `"linear"`: `i + (j - i) * fraction`, where `fraction` is the fractional part of the index surrounded by `i`
            and `j`.
        * `"lower"`: `i`.
        * `"higher"`: `j`.
        * `"nearest"`: `i` or `j` whichever is nearest.
        * `"midpoint"`: (`i` + `j`) / 2.
    discard_infs : bool, default=False
        Whether to discard `-np.inf` and `np.inf` values or not. If False, such values will be capped. If True,
        they will be replaced by `np.nan`.

        !!! info
            Setting `discard_infs=True` is important if the `inf` values are results of divisions by 0, which are
            interpreted by `pandas` as `-np.inf` or `np.inf` depending on the sign of the numerator.
    copy : bool, default=True
        If False, try to avoid a copy and do inplace capping instead. This is not guaranteed to always work inplace;
        e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.

    Attributes
    ----------
    quantiles_ : np.ndarray of shape (2, n_features)
        The computed quantiles for each column of the input data. The first row contains the lower quantile, the second
        row contains the upper quantile.
    n_features_in_ : int
        Number of features seen during `fit`.
    n_columns_ : int
        Deprecated, please use `n_features_in_` instead.

    Examples
    --------
    ```py
    import pandas as pd
    import numpy as np
    from sklego.preprocessing import ColumnCapper

    df = pd.DataFrame({'a':[2, 4.5, 7, 9], 'b':[11, 12, np.inf, 14]})
    df
    '''
         a     b
    0  2.0  11.0
    1  4.5  12.0
    2  7.0   inf
    3  9.0  14.0
    '''

    capper = ColumnCapper()
    capper.fit_transform(df)
    '''
    array([[ 2.375, 11.1  ],
           [ 4.5  , 12.   ],
           [ 7.   , 13.8  ],
           [ 8.7  , 13.8  ]])
    '''

    capper = ColumnCapper(discard_infs=True) # Discarding infs
    df[['a', 'b']] = capper.fit_transform(df)
    df
    '''
           a     b
    0  2.375  11.1
    1  4.500  12.0
    2  7.000   NaN
    3  8.700  13.8
    '''
    ```
    """

    def __init__(
        self,
        quantile_range=(5.0, 95.0),
        interpolation="linear",
        discard_infs=False,
        copy=True,
    ):
        self.quantile_range = quantile_range
        self.interpolation = interpolation
        self.discard_infs = discard_infs
        self.copy = copy

    def fit(self, X, y=None):
        """Fit the `ColumnCapper` transformer by computing quantiles for each column of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the quantiles for capping.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : ColumnCapper
            The fitted transformer.

        Raises
        ------
        ValueError
            If `X` contains non-numeric columns.
        """
        self._check_quantile_range(self.quantile_range)
        self._check_interpolation(self.interpolation)

        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, copy=True, ensure_all_finite=False, reset=True)

        # If X contains infs, we need to replace them by nans before computing quantiles
        np.putmask(X, (X == np.inf) | (X == -np.inf), np.nan)

        # There should be no column containing only nan cells at this point. If that's not the case,
        # it means that the user asked ColumnCapper to fit some column containing only nan or inf cells.
        nans_mask = np.isnan(X)
        invalid_columns_mask = nans_mask.sum(axis=0) == X.shape[0]  # Contains as many nans as rows
        if invalid_columns_mask.any():
            raise ValueError("ColumnCapper cannot fit columns containing only inf/nan values")

        q = [quantile_limit / 100 for quantile_limit in self.quantile_range]
        self.quantiles_ = np.nanquantile(a=X, q=q, axis=0, overwrite_input=True, method=self.interpolation)

        return self

    def transform(self, X):
        """Performs the capping on the column(s) of `X` according to the quantile thresholds computed during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data for which the capping limit(s) will be applied.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            `X` values with capped limits.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        check_is_fitted(self, ["quantiles_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, copy=self.copy, ensure_all_finite=False, reset=False)

        if self.discard_infs:
            np.putmask(X, (X == np.inf) | (X == -np.inf), np.nan)

        # Actually capping
        X = np.minimum(X, self.quantiles_[1, :])
        X = np.maximum(X, self.quantiles_[0, :])

        return X

    @staticmethod
    def _check_quantile_range(quantile_range):
        """Checks for the validity of quantile_range.

        Parameters
        ----------
        quantile_range : Tuple[float, float] | List[float]
            The quantile ranges to perform the capping. Their values must be in the interval [0; 100].

        Raises
        ------
        TypeError
            If `quantile_range` is not a tuple or a list.
        ValueError
            - If `quantile_range` does not contain exactly 2 elements.
            - If `quantile_range` contains values outside of [0; 100].
            - If `quantile_range` contains values in the wrong order.
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
        """Checks for the validity of interpolation.

        Parameters
        ----------
        interpolation : Literal["linear", "lower", "higher", "midpoint", "nearest"]
            Interpolation method to compute the quantiles

        Raises
        ------
        ValueError
            If `interpolation` is not one of the allowed values.
        """
        allowed_interpolations = ("linear", "lower", "higher", "midpoint", "nearest")
        if interpolation not in allowed_interpolations:
            raise ValueError("Available interpolation methods: {}".format(", ".join(allowed_interpolations)))

    @property
    def n_columns_(self):
        warn(
            "Please use `n_features_in_` instead of `n_columns_`, `n_columns_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_

    def _more_tags(self):
        return {"allow_nan": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
