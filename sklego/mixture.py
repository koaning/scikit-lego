import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs
        self.classes = None
        self.gmms = None

    def fit(self, X: np.array, y: np.array) -> "GMMClassifier":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.gmms = {}
        self.classes = np.unique(y)
        for cls in self.classes:
            subset_x, subset_y = X[y == cls], y[y == cls]
            self.gmms[cls] = GaussianMixture(**self.gmm_kwargs).fit(subset_x, subset_y)
        return self

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmms', 'classes'])
        res = np.zeros((X.shape[0], self.classes.shape[0]))
        for idx, cls in enumerate(self.classes):
            res[:, idx] = self.gmms[cls].score_samples(X)
        return np.exp(res)/np.exp(res).sum(axis=1).reshape((X.shape[0], 1))
