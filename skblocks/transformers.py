import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RandomAdder(TransformerMixin, BaseEstimator):
    def __init__(self, noise=1):
        self.n_bins = noise

    def transform(self, X):
        return X + np.random.normal(0, self.n_bins, size=X.shape)

    def fit(self, X, y):
        return self
