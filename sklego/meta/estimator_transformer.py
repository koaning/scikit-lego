from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MetaEstimatorMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
)


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Allows using an estimator such as a model as a transformer in an earlier step of a pipeline

    :param estimator: An instance of the estimator that should be used for the transformation
    :param predict_func: The function called on the estimator when transforming e.g. (`predict`, `predict_proba`)
    :param check_X: Whether to check the input data for NaNs, Infs and non-numeric values
    """

    def __init__(self, estimator, predict_func="predict", check_X=True):
        self.estimator = estimator
        self.predict_func = predict_func
        self.check_X = check_X

    def fit(self, X, y, **kwargs):
        """Fits the estimator"""
        X, y = check_X_y(X, y, estimator=self, force_all_finite=self.check_X, dtype=None, multi_output=True)

        self.multi_output_ = len(y.shape) > 1
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)
        return self

    def transform(self, X):
        """
        Applies the `predict_func` on the fitted estimator.

        Returns array of shape `(X.shape[0], )` if estimator is not multi output.
        For multi output estimators an array of shape `(X.shape[0], y.shape[1])` is returned.
        """
        check_is_fitted(self, "estimator_")
        output = getattr(self.estimator_, self.predict_func)(X)
        return output if self.multi_output_ else output.reshape(-1, 1)
