import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class DeadZoneRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, threshold=0.3, relative=False, effect="linear", n_iter=2000, stepsize=0.01, check_grad=False):
        self.threshold = threshold
        self.relative = relative
        self.effect = effect
        self.n_iter = n_iter
        self.stepsize = stepsize
        self.check_grad = check_grad
        self.allowed_effects = ("linear", "quadratic", "constant")
        self.loss_log_ = None
        self.wts_log_ = None
        self.deriv_log_ = None
        self.coefs_ = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if self.effect not in self.allowed_effects:
            raise ValueError(f"effect {self.effect} must be in {self.allowed_effects}")

        def deadzone(errors):
            if self.effect == "linear":
                return np.where(errors > self.threshold, errors, np.zeros(errors.shape))
            if self.effect == "quadratic":
                return np.where(errors > self.threshold, errors**2, np.zeros(errors.shape))

        def training_loss(weights):
            diff = np.abs(np.dot(X, weights) - y)
            if self.relative:
                diff = diff / y
            return np.mean(deadzone(diff))

        n, k = X.shape

        # Build a function that returns gradients of training loss using autograd.
        training_gradient_fun = grad(training_loss)

        # Check the gradients numerically, just to be safe.
        weights = np.random.normal(0, 1, k)
        if self.check_grad:
            check_grads(training_loss, modes=['rev'])(weights)

        # Optimize weights using gradient descent.
        self.loss_log_ = np.zeros(self.n_iter)
        self.wts_log_ = np.zeros((self.n_iter, k))
        self.deriv_log_ = np.zeros((self.n_iter, k))
        for i in range(self.n_iter):
            weights -= training_gradient_fun(weights) * self.stepsize
            self.wts_log_[i, :] = weights.ravel()
            self.loss_log_[i] = training_loss(weights)
            self.deriv_log_[i, :] = training_gradient_fun(weights).ravel()
        self.coefs_ = weights
        return self

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ['coefs_'])
        return np.dot(X, self.coefs_)
