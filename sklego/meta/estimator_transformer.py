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
from sklearn.exceptions import NotFittedError


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """
    Allows using an estimator such as a model as a transformer in an earlier step of a pipeline

    :param estimator: An instance of the estimator that should be used for the transformation
    :param predict_func: The function called on the estimator when transforming e.g. (`predict`, `predict_proba`)
    """

    def __init__(self, estimator, predict_func="predict"):
        self.estimator = estimator
        self.predict_func = predict_func

    def fit(self, X, y, **kwargs):
        """Fits the estimator"""
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True)
        self.multi_output_ = len(y.shape) > 1
        self.output_len_ = y.shape[1] if self.multi_output_ else 1
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

    def get_feature_names_out(self, feature_names_out=None) -> list:
        """
        Defines descriptive names for each output of the (fitted) estimator.
        :param feature_names_out: Redundant parameter for which the contents are ignored in this function.
        feature_names_out is defined here because EstimatorTransformer can be part of a larger complex pipeline.
        Some components may depend on defined feature_names_out and some not, but it is passed to all components
        in the pipeline if `Pipeline.get_feature_names_out` is called. feature_names_out is therefore necessary
        to define here to avoid `TypeError`s when used within a scikit-learn `Pipeline` object.
        :return: List of descriptive names for each output variable from the fitted estimator.
        """
        check_is_fitted(self, "estimator_")

        estimator_name_lower = self.estimator_.__class__.__name__.lower()
        if self.multi_output_:
            feature_names = [f"{estimator_name_lower}_{i}" for i in range(self.output_len_)]
        else:
            feature_names = [estimator_name_lower]
        return feature_names
