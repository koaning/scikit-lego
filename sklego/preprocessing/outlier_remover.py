from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
)

from sklego.common import TrainOnlyTransformerMixin

class OutlierRemover(TrainOnlyTransformerMixin, BaseEstimator):
    """
    Removes outliers (train-time only) using the supplied removal model.

    :param outlier_detector: must implement `fit` and `predict` methods
    :param refit: If True, fits the estimator during pipeline.fit().

    """

    def __init__(self, outlier_detector, refit=True):
        self.outlier_detector = outlier_detector
        self.refit = refit
        self.estimator_ = None

    def fit(self, X, y=None):
        self.estimator_ = clone(self.outlier_detector)
        if self.refit:
            super().fit(X, y)
            self.estimator_.fit(X, y)
        return self

    def transform_train(self, X):
        check_is_fitted(self, "estimator_")
        predictions = self.estimator_.predict(X)
        check_array(predictions, estimator=self.outlier_detector, ensure_2d=False)
        return X[predictions != -1]
