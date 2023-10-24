from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    check_random_state,
    FLOAT_DTYPES,
)


class RandomRegressor(BaseEstimator, RegressorMixin):
    """
    A RandomRegressor makes random predictions only based on the "y"
    value that is seen. The goal is that such a regressor can be used
    for benchmarking. It should be easily beatable.

    :param str strategy: how we want to select random values, can be "uniform" or "normal"
    :param int seed: the seed value, default: 42
    """

    _ALLOWED_STRATEGIES = ("uniform", "normal")

    def __init__(self, strategy="uniform", random_state=None):
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X: np.array, y: np.array) -> "RandomRegressor":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        if self.strategy not in self._ALLOWED_STRATEGIES:
            raise ValueError(
                f"strategy {self.strategy} is not in {self._ALLOWED_STRATEGIES}"
            )
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.n_features_in_ = X.shape[1]

        self.min_ = np.min(y)
        self.max_ = np.max(y)
        self.mu_ = np.mean(y)
        self.sigma_ = np.std(y)

        return self

    def predict(self, X):
        """
        Predict new data by making random guesses.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        rs = check_random_state(self.random_state)
        check_is_fitted(self, ["n_features_in_", "min_", "max_", "mu_", "sigma_"])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Unexpected input dimension {X.shape[1]}, expected {self.dim_}"
            )

        if self.strategy == "normal":
            return rs.normal(self.mu_, self.sigma_, X.shape[0])
        if self.strategy == "uniform":
            return rs.uniform(self.min_, self.max_, X.shape[0])

    @property
    def dim_(self):
        warn(
            "Please use `n_features_in_` instead of `dim_`, `dim_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_
