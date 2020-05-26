import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    FLOAT_DTYPES,
)


class DecayEstimator(BaseEstimator):
    """
    Morphs an estimator suchs that the training weights can be
    adapted to ensure that points that are far away have less weight.
    Note that it is up to the user to sort the dataset appropriately.
    This meta estimator will only work for estimators that have a
    "sample_weights" argument in their `.fit()` method.

    The DecayEstimator will use exponential decay to weight the parameters.

    w_{t-1} = decay * w_{t}
    """

    def __init__(self, model, decay: float = 0.999, decay_func="exponential"):
        self.model = model
        self.decay = decay
        self.func = decay_func

    def _is_classifier(self):
        return any(
            ["ClassifierMixin" in p.__name__ for p in type(self.model).__bases__]
        )

    def fit(self, X, y):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        self.weights_ = np.cumprod(np.ones(X.shape[0]) * self.decay)[::-1]
        self.estimator_ = clone(self.model)
        try:
            self.estimator_.fit(X, y, sample_weight=self.weights_)
        except TypeError as e:
            if "sample_weight" in str(e):
                raise TypeError(
                    f"Model {type(self.model).__name__}.fit() does not have 'sample_weight'"
                )
        if self._is_classifier():
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        if self._is_classifier():
            check_is_fitted(self, ["classes_"])
        check_is_fitted(self, ["weights_", "estimator_"])
        return self.estimator_.predict(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)