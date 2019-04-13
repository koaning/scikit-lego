import autograd.numpy as np
from autograd import grad

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state, FLOAT_DTYPES


class DeadZoneRegression(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.3, relative=False, effect="linear", n_iter=100, stepsize=0.001):
        self.allowed_effects = ("constant", "linear", "quadratic")
        self.threshold = threshold
        self.relative = relative
        self.effect = effect
        self.n_iter = n_iter
        self.stepsize = stepsize

    def fit(self, X: np.array, y: np.array) -> "DeadZoneRegression":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        if self.effect not in self.allowed_effects:
            raise ValueError(f"effect {self.effect} is not in {self.allowed_effects}")
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        def deadzone(err):
            abs_err = np.abs(err)
            if self.effect == "constant":
                return np.where(abs_err > self.threshold, 1, 0)
            if self.effect == "linear":
                return np.where(abs_err > self.threshold, abs_err - self.threshold, 0)
            if self.effect == "quadratic":
                return np.where(abs_err > self.threshold, (abs_err - self.threshold)**2, 0)

        def training_loss(weights):
            preds = X @ weights
            error = y - preds
            if self.relative:
                error = (y - preds)/y
            return np.sum(deadzone(error))

        weights = np.zeros((X.shape[1], 1))
        print(weights)
        training_gradient = grad(training_loss)
        print(weights)
        for i in range(self.n_iter):
            weights -= training_gradient(weights) * self.stepsize
        self.coefs_ = weights
        return self

    def predict(self, X):
        """
        Predict new data by making random guesses.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['coefs_'])

        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return X @ self.coefs_
