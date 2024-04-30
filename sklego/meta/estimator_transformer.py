from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_X_y


class EstimatorTransformer(TransformerMixin, MetaEstimatorMixin, BaseEstimator):
    """Allow using an estimator as a transformer in an earlier step of a pipeline.

    !!! warning
        By default all the checks on the inputs `X` and `y` are delegated to the wrapped estimator.

        To change such behaviour, set `check_input` to `True`.

    Parameters
    ----------
    estimator : scikit-learn compatible estimator
        The estimator to be applied to the data, used as transformer.
    predict_func : str, default="predict"
        The method called on the estimator when transforming e.g. (`"predict"`, `"predict_proba"`).
    check_input : bool, default=False
        Whether or not to check the input data. If False, the checks are delegated to the wrapped estimator.

    Attributes
    ----------
    estimator_ : scikit-learn compatible estimator
        The fitted underlying estimator.
    multi_output_ : bool
        Whether or not the estimator is multi output.
    """

    def __init__(self, estimator, predict_func="predict", check_input=False):
        self.estimator = estimator
        self.predict_func = predict_func
        self.check_input = check_input

    def fit(self, X, y, **kwargs):
        """Fit the underlying estimator on training data `X` and `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional keyword arguments passed to the `fit` method of the underlying estimator.

        Returns
        -------
        self : EstimatorTransformer
            The fitted transformer.
        """

        if self.check_input:
            X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, multi_output=True)

        self.multi_output_ = len(y.shape) > 1
        self.output_len_ = y.shape[1] if self.multi_output_ else 1
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)
        return self

    def transform(self, X):
        """Transform the data by applying the `predict_func` of the fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        output : array-like of shape (n_samples,) | (n_samples, n_outputs)
            The transformed data. Array will be of shape `(X.shape[0], )` if estimator is not multi output.
            For multi output estimators an array of shape `(X.shape[0], y.shape[1])` is returned.
        """
        # The check below will also check if the underlying estimator_ is fitted.
        # This is checked through the __sklearn_is_fitted method.
        check_is_fitted(self)
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
        check_is_fitted(self)
        estimator_name_lower = self.estimator_.__class__.__name__.lower()
        if self.multi_output_:
            feature_names = [f"{estimator_name_lower}_{i}" for i in range(self.output_len_)]
        else:
            feature_names = [estimator_name_lower]
        return feature_names

    def __sklearn_is_fitted(self) -> bool:
        """
        Custom additional requirements that need to be satisfied to pass check_is_fitted.
        :return: Boolean indicating if the additional requirements
        for determining check_is_fitted are satisfied.
        """
        # check_is_fitted(self) will call this method.
        # Source code of where this method is called in check_is_fitted:
        # https://github.com/scikit-learn/scikit-learn/blob/626b4608d4f840af7c37bff2ccb38fcfd2ef594f/sklearn/utils/validation.py#L1338
        has_fit_attr = all(hasattr(self, attr) for attr in ["multi_output_", "output_len_", "estimator_"])
        check_is_fitted(self.estimator_)
        return has_fit_attr
