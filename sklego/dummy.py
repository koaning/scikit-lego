from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_random_state
from sklearn_compat.utils.validation import validate_data


class RandomRegressor(RegressorMixin, BaseEstimator):
    """A `RandomRegressor` makes random predictions only based on the `y` value that is seen.

    The goal is that such a regressor can be used for benchmarking. It _should be_ easily beatable.

    Parameters
    ----------
    strategy : Literal["uniform", "normal"], default="uniform"
        How we want to select random values, either "uniform" or "normal"
    random_state : int | None, default=None
        The seed value used for the random number generator.

    Attributes
    ----------
    min_ : float
        The minimum value of `y` seen during `fit`.
    max_ : float
        The maximum value of `y` seen during `fit`.
    mu_ : float
        The mean value of `y` seen during `fit`.
    sigma_ : float
        The standard deviation of `y` seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    dim_ : int
        Deprecated, please use `n_features_in_` instead.

    Examples
    --------
    ```py
    from sklego.dummy import RandomRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=10, n_features=2, random_state=42)

    RandomRegressor(strategy="uniform", random_state=123).fit(X, y).predict(X).round(2)
    # array([ 57.63, -66.05, -83.92,  13.88,  64.56, -24.77, 143.33,  54.12,
    #     -7.34, -34.11])

    RandomRegressor(strategy="normal", random_state=123).fit(X, y).predict(X).round(2)
    # array([-128.45,   78.05,    7.23, -170.15,  -78.18,  142.9 , -261.39,
    #     -63.34,  104.68, -106.75])
    ```
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
            raise ValueError(f"strategy {self.strategy} is not in {self._ALLOWED_STRATEGIES}")
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)

        self.min_ = np.min(y)
        self.max_ = np.max(y)
        self.mu_ = np.mean(y)
        self.sigma_ = np.std(y)

        return self

    def predict(self, X):
        """Predict new data by generating random guesses following the given `strategy` based on the `y` statistics seen
        during `fit`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        rs = check_random_state(self.random_state)
        check_is_fitted(self, ["n_features_in_", "min_", "max_", "mu_", "sigma_"])

        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

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

    @property
    def allowed_strategies(self):
        warn(
            "Please use `_ALLOWED_STRATEGIES` instead of `allowed_strategies`,"
            "`allowed_strategies` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self._ALLOWED_STRATEGIES

    def _more_tags(self):
        return {"poor_score": True, "non_deterministic": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        tags.regressor_tags.poor_score = True
        return tags
