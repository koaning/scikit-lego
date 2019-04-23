
import inspect

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, OutlierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
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
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        _check_gmm_keywords(self.gmm_kwargs)
        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            self.gmms_[c] = GaussianMixture(**self.gmm_kwargs).fit(subset_x, subset_y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['gmms_', 'classes_'])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmms_', 'classes_'])
        res = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx, c in enumerate(self.classes_):
            res[:, idx] = self.gmms_[c].score_samples(X)
        return np.exp(res)/np.exp(res).sum(axis=1)[:, np.newaxis]


class GMMOutlierDetector(OutlierMixin, BaseEstimator):
    """
    The GMMDetector trains a Gaussian Mixture Model on a dataset X. Once
    a density is trained we can evaluate the likelihood scores to see if
    it is deemed likely. By giving a threshold this model might then label
    outliers if their likelihood score is too low.

    :param threshold: the limit at which the model thinks an outlier appears, must be between (0, 1)
    :param gmm_kwargs: features that are passed to the `GaussianMixture` from sklearn
    :param method: the method that the threshold will be applied to, possible values = [stddev, default=quantile]

    If you select method="quantile" then the threshold value represents the
    quantile value to start calling something an outlier.

    If you select method="stddev" then the threshold value represents the
    numbers of standard deviations before calling something an outlier.
    """
    def __init__(self, threshold=0.99, method='quantile', random_state=42, **gmm_kwargs):
        self.gmm_kwargs = gmm_kwargs
        self.threshold = threshold
        self.method = method
        self.random_state = random_state
        self.allowed_methods = ["quantile", "stddev"]

    def fit(self, X: np.array, y=None) -> "GMMOutlierDetector":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: ignored but kept in for pipeline support
        :return: Returns an instance of self.
        """

        # GMM sometimes throws an error if you don't do this
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        if (self.method == "quantile") and ((self.threshold > 1) or (self.threshold < 0)):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold < 1")
        if (self.method == "stddev") and (self.threshold < 0):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold ")
        if self.method not in self.allowed_methods:
            raise ValueError(f"Method not recognised. Method must be in {self.allowed_methods}")

        _check_gmm_keywords(self.gmm_kwargs)
        self.gmm_ = GaussianMixture(**self.gmm_kwargs, random_state=self.random_state).fit(X)
        score_samples = self.gmm_.score_samples(X)

        if self.method == "quantile":
            self.likelihood_threshold_ = np.quantile(score_samples, 1 - self.threshold)

        if self.method == "stddev":
            density = gaussian_kde(score_samples)
            max_x_value = minimize_scalar(lambda x: -density(x)).x
            mean_likelihood = score_samples.mean()
            new_likelihoods = score_samples[score_samples < max_x_value]
            new_likelihoods_std = np.std(new_likelihoods - mean_likelihood)
            self.likelihood_threshold_ = mean_likelihood - (self.threshold * new_likelihoods_std)

        return self

    def score_samples(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['gmm_', 'likelihood_threshold_'])
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        return self.gmm_.score_samples(X) * -1

    def decision_function(self, X):

        # We subtract self.offset_ to make 0 be the threshold value for being
        # an outlier:

        return self.score_samples(X) + self.likelihood_threshold_

    def predict(self, X):
        """
        Predict if a point is an outlier. If the output is 0 then
        the model does not think it is an outlier.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: array, shape=(n_samples,) the predicted data. 1 for inliers, -1 for outliers.
        """
        predictions = (self.decision_function(X) >= 0).astype(np.int)
        predictions[predictions == 0] = -1
        return predictions
