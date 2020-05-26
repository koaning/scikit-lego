import numpy as np
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    FLOAT_DTYPES,
)


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Allows using an estimator such as a model as a transformer in an earlier step of a pipeline

    :param estimator: An instance of the estimator that should be used for the transformation
    :param predict_func: The function called on the estimator when transforming e.g. (`predict`, `predict_proba`)
    """

    def __init__(self, estimator, predict_func="predict"):
        self.estimator = estimator
        self.predict_func = predict_func

    def fit(self, X, y):
        """Fits the estimator"""
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        """
        Applies the `predict_func` on the fitted estimator.

        Returns an array of shape `(X.shape[0], )`.
        """
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, self.predict_func)(X).reshape(-1, 1)
