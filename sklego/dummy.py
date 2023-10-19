import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted, check_random_state


class RandomRegressor(BaseEstimator, RegressorMixin):
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
    dim_ : int
        The number of features seen during `fit`.

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

    def __init__(self, strategy="uniform", random_state=None):
        self.strategy = strategy
        self.random_state = random_state
        self.allowed_strategies = ("uniform", "normal")

    def fit(self, X, y):
        """Fit the estimator on training data `X` and `y` by calculating and storing the min, max, mean and std of `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : RandomRegressor
            The fitted estimator.
        """

        if self.strategy not in self.allowed_strategies:
            raise ValueError(f"strategy '{self.strategy}' is not in {self.allowed_strategies}")
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = X.shape[1]

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
        check_is_fitted(self, ["dim_", "min_", "max_", "mu_", "sigma_"])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if X.shape[1] != self.dim_:
            raise ValueError(f"Unexpected input dimension {X.shape[1]}, expected {self.dim_}")

        if self.strategy == "normal":
            return rs.normal(self.mu_, self.sigma_, X.shape[0])
        if self.strategy == "uniform":
            return rs.uniform(self.min_, self.max_, X.shape[0])
