import numpy as np
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    FLOAT_DTYPES,
)

from sklego.base import ProbabilisticClassifier


class Thresholder(BaseEstimator, ClassifierMixin):
    """
    Takes a two class estimator and moves the threshold. This way you might
    design the algorithm to only accept a certain class if the probability
    for it is larger than, say, 90% instead of 50%.
    """

    def __init__(self, model, threshold: float):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.estimator_ = clone(self.model)
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The Thresholder meta model only works on classifcation models with .predict_proba."
            )
        self.estimator_.fit(X, y)
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
