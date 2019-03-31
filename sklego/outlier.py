import numpy as np
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GMMDetector(BaseEstimator):
    def __init__(self, threshold=0.99, **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs
        self.threshold = threshold
        self.likelihood_threshold = None
        self.gmm = None

    def fit(self, X: np.array, y=None) -> "GMMDetector":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: ignored but kept in for pipeline support
        :return: Returns an instance of self.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        self.gmm = GaussianMixture(**self.gmm_kwargs).fit(X)
        self.likelihood_threshold = np.quantile(self.gmm.score_samples(X), self.threshold)
        return self

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmm', 'likelihood_threshold'])
        return self.gmm.score_samples(X) > self.likelihood_threshold
