try:
    import cvxpy as cp
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    cp = NotInstalledPackage("cvxpy")

from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


def _mk_monotonic_average(xs, ys, intervals, method="increasing", **kwargs):
    """Creates smoothed averages of `ys` at the intervals given by `intervals`.

    Parameters
    ----------
    xs : array-like of shape (n_samples,)
        All the datapoints of a feature (represents the x-axis).
    ys : array-like of shape (n_samples,)
        All the datapoints what we'd like to predict (represents the y-axis).
    intervals : array-like of shape (n_intervals,)
        The intervals at which we'd like to get a good average value.
    method : Literal["increasing", "decreasing"]}, default="increasing"
        The method that is used for smoothing, can be either `"increasing"` or `"decreasing"`.

    Returns
    -------
    np.ndarray of shape (n_intervals,)
        An array of the same shape of `intervals` that represents the average `y`-values at those intervals,
        keeping the constraint in mind.
    """
    x_internal = np.array([xs >= i for i in intervals]).T.astype(float)
    betas = cp.Variable(x_internal.shape[1])
    objective = cp.Minimize(cp.sum_squares(x_internal @ betas - ys))
    if method == "increasing":
        constraints = [betas[i + 1] >= 0 for i in range(betas.shape[0] - 1)]
    elif method == "decreasing":
        constraints = [betas[i + 1] <= 0 for i in range(betas.shape[0] - 1)]
    else:
        raise ValueError(f"method must be either `increasing` or `decreasing`, got: {method}")
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return betas.value.cumsum()


def _mk_average(xs, ys, intervals, method="average", span=1, **kwargs):
    """Creates smoothed averages of `ys` at the intervals given by `intervals`.

    Parameters
    ----------
    xs : array-like of shape (n_samples,)
        All the datapoints of a feature (represents the x-axis).
    ys : array-like of shape (n_samples,)
        All the datapoints what we'd like to predict (represents the y-axis).
    intervals : array-like of shape (n_intervals,)
        The intervals at which we'd like to get a good average value
    method : Literal["average", "normal"], default="average"
        The method that is used for smoothing, can be either `"average"` or `"normal"`.
    span : float, default=1.0
        If the method is `"average"` then this is the span around the interval that is used to determine the average
        `y`-value, if the method is `"normal"` the span becomes the value of sigma that is used for weighted averaging.

    Returns
    -------
    np.ndarray of shape (n_intervals,)
        An array of the same shape of `intervals` that represents the average `y`-values at those intervals.
    """
    results = np.zeros(intervals.shape)
    for idx, interval in enumerate(intervals):
        if method == "average":
            distances = 1 / (0.01 + np.abs(xs - interval))
            predicate = (xs < (interval + span)) | (xs < (interval - span))
        elif method == "normal":
            distances = np.exp(-((xs - interval) ** 2) / span)
            predicate = xs == xs
        else:
            raise ValueError("method needs to be either `average` or `normal`")
        subset = ys[predicate]
        dist_subset = distances[predicate]
        results[idx] = np.average(subset, weights=dist_subset)
    return results


class IntervalEncoder(TransformerMixin, BaseEstimator):
    """The `IntervalEncoder` transformer bends features in `X` with regards to`y`.

    We take each column in `X` separately and smooth it towards `y` using the strategy that is defined in `method`.

    Note that this allows us to make certain features strictly monotonic in your machine learning model if you follow
    this with an appropriate model.

    Parameters
    ----------
    n_chunks : int, default=10
        The number of cuts that makes the interval.
    span : float, default=1.0
        A hyperparameter for the interpolation method, if the method is `"normal"` it resembles the width of the radial
        basis function used to weigh the points. It is ignored if the method is `"increasing"` or `"decreasing"`.
    method : Literal["average", "normal", "increasing", "decreasing"], default="normal"
        The interpolation method used, can be either `"average"`, `"normal"`, `"increasing"` or `"decreasing"`.

    Attributes
    ----------
    quantiles_ : np.ndarray of shape (n_features, n_chunks)
        The quantiles that are used to cut the interval.
    heights_ : np.ndarray of shape (n_features, n_chunks)
        The heights of the quantiles that are used to cut the interval.
    n_features_in_ : int
        Number of features seen during `fit`.
    num_cols_ : int
        Deprecated, please use `n_features_in_` instead.
    """

    _ALLOWED_METHODS = ("average", "normal", "increasing", "decreasing")

    def __init__(self, n_chunks=10, span=1, method="normal"):
        self.span = span
        self.method = method
        self.n_chunks = n_chunks

    def fit(self, X, y):
        """Fit the `IntervalEncoder` transformer by computing interpolation quantiles for each column of `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : IntervalEncoder
            The fitted transformer.

        Raises
        ------
        ValueError
            - If `method` is not one of `"average"`, `"normal"`, `"increasing"` or `"decreasing"`.
            - If `n_chunks` is not a positive integer.
            - If `span` is not between 0 and 1.
        """

        if self.method not in self._ALLOWED_METHODS:
            raise ValueError(f"`method` must be in {self._ALLOWED_METHODS}, got `{self.method}`")
        if self.n_chunks <= 0:
            raise ValueError(f"`n_chunks` must be >= 1, received {self.n_chunks}")
        if self.span > 1.0:
            raise ValueError(f"Error, we expect 0 <= span <= 1, received span={self.span}")
        if self.span < 0.0:
            raise ValueError(f"Error, we expect 0 <= span <= 1, received span={self.span}")

        # these two matrices will have shape (columns, quantiles)
        # quantiles indicate where the interval split occurs
        X, y = validate_data(self, X=X, y=y, reset=True)

        self.quantiles_ = np.zeros((X.shape[1], self.n_chunks))
        # heights indicate what heights these intervals will have
        self.heights_ = np.zeros((X.shape[1], self.n_chunks))
        self.n_features_in_ = X.shape[1]

        average_func = _mk_average if self.method in ["average", "normal"] else _mk_monotonic_average

        for col in range(X.shape[1]):
            self.quantiles_[col, :] = np.quantile(X[:, col], q=np.linspace(0, 1, self.n_chunks))
            self.heights_[col, :] = average_func(
                X[:, col],
                y,
                self.quantiles_[col, :],
                span=self.span,
                method=self.method,
            )
        return self

    def transform(self, X):
        """Performs smoothing on the column(s) of `X` according to the quantile values computed during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data for which the smoothing will be applied.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            `X` values with smoothed values.

        Raises
        ------
        ValueError
            If the number of columns from `X` differs from the number of columns when fitting.
        """
        check_is_fitted(self, ["quantiles_", "heights_", "n_features_in_"])
        X = validate_data(self, X=X, reset=False)

        transformed = np.zeros(X.shape)
        for col in range(transformed.shape[1]):
            transformed[:, col] = np.interp(X[:, col], self.quantiles_[col, :], self.heights_[col, :])
        return transformed

    @property
    def num_cols_(self):
        warn(
            "Please use `n_features_in_` instead of `num_cols_`, `num_cols_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_

    @property
    def allowed_methods(self):
        warn(
            "Please use `_ALLOWED_METHODS` instead of `allowed_methods`,"
            "`allowed_methods` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self._ALLOWED_METHODS
