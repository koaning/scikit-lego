import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, check_X_y


class DecayEstimator(BaseEstimator):
    """Morphs an estimator such that the training weights can be adapted to ensure that points that are far away have
    less weight.

    This meta estimator will only work for estimators that allow a `sample_weights` argument in their `.fit()` method.
    The meta estimator `.fit()` method computes the weights to pass to the estimator's `.fit()` method.

    !!! warning
        It is up to the user to sort the dataset appropriately.

    !!! warning
        By default all the checks on the inputs `X` and `y` are delegated to the wrapped estimator.

        To change such behaviour, set `check_input` to `True`.

        Remark that if the check is skipped, then `y` should have a `shape` attribute, which is
        used to extract the number of samples in training data, and compute the weights.

    !!! info
        The DecayEstimator will use exponential decay to weight the parameters.

        $$w_{t-1} = decay * w_{t}$$

    Parameters
    ----------
    model : scikit-learn compatible estimator
        The estimator to be wrapped.
    decay : float, default=0.999
        The decay factor to use.
    decay_func : str, default="exponential"
        The decay function to use. Currently only exponential decay is supported.
    check_input : bool, default=False
        Whether or not to check the input data. If False, the checks are delegated to the wrapped estimator.

    Attributes
    ----------
    estimator_ : scikit-learn compatible estimator
        The fitted estimator.
    weights_ : array-like of shape (n_samples,)
        The weights used to train the estimator.
    classes_ : array-like of shape (n_classes,)
        The classes labels. Only present if the wrapped estimator is a classifier.
    """

    def __init__(self, model, decay: float = 0.999, decay_func="exponential", check_input=False):
        self.model = model
        self.decay = decay
        self.decay_func = decay_func
        self.check_input = check_input

    def _is_classifier(self):
        """Checks if the wrapped estimator is a classifier."""
        return any(["ClassifierMixin" in p.__name__ for p in type(self.model).__bases__])

    @property
    def _estimator_type(self):
        """Computes `_estimator_type` dynamically from the wrapped model."""
        return self.model._estimator_type

    def fit(self, X, y):
        """Fit the underlying estimator on the training data `X` and `y` using the calculated sample weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : DecayEstimator
            The fitted estimator.
        """

        if self.check_input:
            X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES, ensure_min_features=0)

        self.weights_ = np.cumprod(np.ones(y.shape[0]) * self.decay)[::-1]
        self.estimator_ = clone(self.model)
        try:
            self.estimator_.fit(X, y, sample_weight=self.weights_)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(f"Model {type(self.model).__name__}.fit() does not have 'sample_weight'")
        if self._is_classifier():
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """Predict target values for `X` using trained underlying estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        if self._is_classifier():
            check_is_fitted(self, ["classes_"])
        check_is_fitted(self, ["weights_", "estimator_"])
        return self.estimator_.predict(X)

    def score(self, X, y):
        """Alias for `.score()` method of the underlying estimator."""
        return self.estimator_.score(X, y)
