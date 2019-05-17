import inspect

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


def _check_gmm_keywords(kwargs):
    for key in kwargs.keys():
        if key not in inspect.signature(GaussianMixture).parameters.keys():
            raise ValueError(f"Keyword argument {key} is not in `sklearn.mixture.GaussianMixture`")


class GaussianMixtureNB(BaseEstimator, ClassifierMixin):
    """
    The GaussianMixtureNB trains a Naive Bayes Classifier that uses a mixture
    of gaussians instead of merely training a single one.

    You can pass any keyword parameter that scikit-learn's Gaussian Mixture
    Model uses and it will be passed along.
    """
    def __init__(self, **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs

    def fit(self, X: np.array, y: np.array) -> "GaussianMixtureNB":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        _check_gmm_keywords(self.gmm_kwargs)
        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        self.num_fit_cols_ = X.shape[1]
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            self.gmms_[c] = [GaussianMixture(**self.gmm_kwargs).fit(subset_x[:, i].reshape(-1, 1), subset_y)
                             for i in range(X.shape[1])]
        return self

    def predict(self, X):
        check_is_fitted(self, ['gmms_', 'classes_'])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.array):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.num_fit_cols_ != X.shape[1]:
            raise ValueError(f"number of columns {X.shape[1]} does not match fit size {self.num_fit_cols_}")
        check_is_fitted(self, ['gmms_', 'classes_'])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for k, v in self.gmms_.items():
            class_idx = int(np.argwhere(self.classes_ == k))
            probs[:, class_idx] = np.array([m.score_samples(np.expand_dims(X[:, idx], 1)) for
                                            idx, m in enumerate(v)]).sum(axis=0)
        likelihood = np.exp(probs)
        return likelihood / likelihood.sum(axis=1).reshape(-1, 1)
