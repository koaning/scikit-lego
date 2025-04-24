from sklearn import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import _check_n_features, validate_data


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

    Example
    -------
    ```py
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklego.meta import EstimatorTransformer

    np.random.seed(0)
    n1, n2 = 50, 100
    X = np.concatenate([np.random.normal(0, 1, (n1, 2)), np.random.normal(2, 1, (n2, 2))], axis=0)
    y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0).reshape(-1)

    estimator_transformer = EstimatorTransformer(
        estimator=LogisticRegression(max_iter=1000, random_state=0),
        predict_func="predict_proba",
        check_input= False
        )

    # Fit the data
    estimator_transformer.fit(X, y)
    print(f'Is the output multimodal? {estimator_transformer.multi_output_}')
    ### Is the output multimodal? False

    # Transform the data using "predict_func" as specified
    transformed_X = estimator_transformer.transform(X)
    print(f'Shape of transformed data: {transformed_X.shape}')
    ### Shape of transformed data: (300, 1)
    ```
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
            X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, multi_output=True, reset=True)
        else:
            _check_n_features(self, X, reset=True)

        self.multi_output_ = len(y.shape) > 1
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **kwargs)
        self.n_features_in_ = X.shape[1]
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

        check_is_fitted(self, "estimator_")
        if self.check_input:
            X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        else:
            _check_n_features(self, X, reset=False)

        output = getattr(self.estimator_, self.predict_func)(X)
        return output if self.multi_output_ else output.reshape(-1, 1)
