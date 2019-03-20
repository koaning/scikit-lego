import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RandomRegressor(BaseEstimator, RegressorMixin):
    """
    A RandomRegressor makes random predictions only based on the "y"
    value that is seen. The goal is that such a regressor can be used
    for benchmarking. It should be easily beatable.

    :param str strategy: how we want to select random values, can be "uniform" or "normal"
    :param int seed: the seed value, default: 42
    """
    def __init__(self, strategy="uniform", seed=42):
        self.allowed_strategies = ("uniform", "normal")
        if strategy not in self.allowed_strategies:
            raise ValueError(f"strategy {strategy} is not in {self.allowed_strategies}")
        self.seed = seed
        self.strategy = strategy
        self.min_ = None
        self.max_ = None
        self.mu_ = None
        self.sigma_ = None
        self.gmm = None

    def fit(self, X: np.array, y: np.array) -> "RandomRegressor":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
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
        np.random.seed(self.seed)
        if self.strategy not in self.allowed_strategies:
            raise ValueError(f"strategy {self.strategy} is not in {self.allowed_strategies}")
        if self.strategy == 'normal':
            return np.random.normal(self.mu_, self.sigma_, X.shape[0])
        if self.strategy == 'uniform':
            return np.random.uniform(self.min_, self.max_, X.shape[0])
