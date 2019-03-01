import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RandomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, strategy="uniform"):
        """
        :param strategy: One of 'uniform', 'normal' or 'gmm'.
        """
        self.allowed_strategies = ("uniform", "normal")
        if strategy not in self.allowed_strategies:
            raise ValueError(f"strategy {strategy} is not in {self.allowed_strategies}")
        self.strategy = strategy
        self.min_ = None
        self.max_ = None
        self.mu_ = None
        self.sigma_ = None
        self.gmm = None

    def fit(self, X, y):
        self.min_ = np.min(y)
        self.max_ = np.max(y)
        self.mu_ = np.mean(y)
        self.sigma_ = np.std(y)
        return self

    def predict(self, X):
        if self.strategy not in self.allowed_strategies:
            raise ValueError(f"strategy {strategy} is not in {self.allowed_strategies}")
        if self.strategy == 'normal':
            return np.random.normal(self.mu_, self.sigma_, X.shape[0])
        if self.strategy == 'uniform':
            return np.random.uniform(self.min_, self.max_, X.shape[0])