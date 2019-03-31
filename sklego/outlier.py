import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state, FLOAT_DTYPES


class GMMDetector(BaseEstimator, ClassifierMixin):
    """
    A GMMDetector tries to estimate a Gaussian Mixture Model on a dataset
    such and will attempt a outlier prediction by setting a threshold. Can
    be used to make a shortlist of outlier candidates.
    """
    def __init__(self, **gmm_kwargs):
        """
        :param strategy: One of 'uniform', 'normal' or 'gmm'.
        """
        self.gmm = GaussianMixture(**gmm_kwargs)

    def fit(self, X: np.array, y: np.array) -> "RandomRegressor":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.dim_ = X.shape[1]

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
        check_is_fitted(self, ['dim_', 'min_', 'max_', 'mu_', 'sigma_'])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if X.shape[1] != self.dim_:
            raise ValueError(f'Unexpected input dimension {X.shape[1]}, expected {self.dim_}')
        if self.strategy not in self.allowed_strategies:
            raise ValueError(f"strategy {self.strategy} is not in {self.allowed_strategies}")

        if self.strategy == 'normal':
            return rs.normal(self.mu_, self.sigma_, X.shape[0])
        if self.strategy == 'uniform':
            return rs.uniform(self.min_, self.max_, X.shape[0])