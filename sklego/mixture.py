import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES

from scipy.stats import gaussian_kde


def _check_gmm_keywords(kwargs):
    for key in kwargs.keys():
        if key not in inspect.signature(GaussianMixture).parameters.keys():
            raise ValueError(f"Keyword argument {key} is not in `sklearn.mixture.GaussianMixture`")


class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs

    def fit(self, X: np.array, y: np.array) -> "GMMClassifier":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        _check_gmm_keywords(self.gmm_kwargs)
        self.gmms_ = {}
        self.classes_ = np.unique(y)
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            self.gmms_[c] = GaussianMixture(**self.gmm_kwargs).fit(subset_x, subset_y)
        return self

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmms_', 'classes_'])
        res = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx, c in enumerate(self.classes_):
            res[:, idx] = self.gmms_[c].score_samples(X)
        return np.exp(res)/np.exp(res).sum(axis=1)[:, np.newaxis]


class GMMOutlierDetector(BaseEstimator):
    """
    The GMMDetector trains a Gaussian Mixture Model on a dataset X. Once
    a density is trained we can evaluate the likelihood scores to see if
    it is deemed likely. By giving a threshold this model might then label
    outliers if their likelihood score is too low.

    :param threshold: the limit at which the model thinks an outlier appears, must be between (0, 1)
    :param gmm_kwargs: features that are passed to the `GaussianMixture` from sklearn
    """
    def __init__(self, threshold=0.99, method='quantile', **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs
        self.threshold = threshold
        self.method = method

    def fit(self, X: np.array, y=None) -> "GMMOutlierDetector":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: ignored but kept in for pipeline support
        :return: Returns an instance of self.
        """

        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        if (self.method == "quantile") and ((self.threshold > 1) or (self.threshold < 0)):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold < 1")

        elif (self.method == "stddev") and (self.threshold < 0):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold ")

        elif (self.method != "quantile") and (self.method != "stdev"):
            raise ValueError("Method not recognised. Method must be either quantile or std")

        _check_gmm_keywords(self.gmm_kwargs)
        self.gmm_ = GaussianMixture(**self.gmm_kwargs).fit(X)

        if self.method == "quantile":
            self.likelihood_threshold_ = np.quantile(self.gmm_.score_samples(X), 1 - self.threshold)

        elif self.method == "stdev":
            likelihoods = self.gmm_.score_samples(X)
            density = gaussian_kde(likelihoods)
            my_range = np.arange(likelihoods.min(), likelihoods.max(),
                                 (np.abs((likelihoods.max()) - (likelihoods.min())) / 10000))

            fitted_dist = density(my_range)
            index_max_y = np.argmax(fitted_dist)
            index_max_x = my_range[index_max_y]
            mean_likelihood = likelihoods.mean()
            new_likelihoods = likelihoods[likelihoods < index_max_x]
            new_likelihoods_std = np.sqrt(np.sum((new_likelihoods - mean_likelihood) ** 2)/(len(new_likelihoods) - 1))
            self.likelihood_threshold_ = mean_likelihood - (self.threshold * new_likelihoods_std)

        return self

    def score_samples(self, X):

        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        return self.gmm_.score_samples(X)

    def predict(self, X):
        """
        Predict if a point is an outlier. If the output is 0 then
        the model does not think it is an outlier.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmm_', 'likelihood_threshold_'])
        return (self.score_samples(X) < self.likelihood_threshold_).astype(np.int)
