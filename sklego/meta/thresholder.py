import numpy as np
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.utils.validation import (
    check_is_fitted
)
from sklearn.exceptions import NotFittedError

from sklego.base import ProbabilisticClassifier


class Thresholder(BaseEstimator, ClassifierMixin):
    """
    Takes a two class estimator and moves the threshold. This way you might
    design the algorithm to only accept a certain class if the probability
    for it is larger than, say, 90% instead of 50%.

    :param model: the model to threshold
    :param threshold: the actual threshold to use
    :param refit: if True, we will always retrain the model even if it is already fitted.
    If False we only refit if the original model isn't fitted.
    """

    def __init__(self, model, threshold: float, refit=False):
        self.model = model
        self.threshold = threshold
        self.refit = refit

    def _handle_refit(self, X, y, sample_weight=None):
        """Only refit when we need to, unless refit=True is present."""
        if self.refit:
            self.estimator_ = clone(self.model)
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        else:
            try:
                _ = self.estimator_.predict(X[:1])
            except NotFittedError:
                self.estimator_.fit(X, y, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :param sample_weight: array-like, shape=(n_samples) Individual weights for each sample.
        :return: Returns an instance of self.
        """
        self.estimator_ = self.model
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The Thresholder meta model only works on classification models with .predict_proba."
            )
        self._handle_refit(X, y, sample_weight)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "The Thresholder meta model only works on models with two classes."
            )
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        predicate = self.estimator_.predict_proba(X)[:, 1] > self.threshold
        return np.where(predicate, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "estimator_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)
