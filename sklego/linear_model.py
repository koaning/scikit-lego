import autograd.numpy as np
import cvxpy as cp
from autograd import grad
from autograd.test_util import check_grads

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES, column_or_1d


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


class FairClassifier(BaseEstimator, ClassifierMixin):
    r"""
    A fair logistic regression classifier.

    Minimizes the Log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier.

    Only works for binary classification problems

    .. math::
        \begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right) d
        \boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}

    Source:
    - M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification

    :param covariance_threshold: The maximum allowed covariance between the sensitive attributes and the distance to the
    decision boundary
    :param sensitive_cols: List of sensitive column names(when X is a dataframe)
    or a list of column indices when X is a numpy array.
    :param C: Inverse of regularization strength; must be a positive float.
    Like in support vector machines, smaller values specify stronger regularization.
    :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    :param max_iter: Maximum number of iterations taken for the solvers to converge.

    """
    def __init__(self, covariance_threshold, sensitive_cols, C=1.0, fit_intercept=True, max_iter=100):
        self.sensitive_cols = sensitive_cols
        self.fit_intercept = fit_intercept
        self.covariance_threshold = covariance_threshold
        self.max_iter = max_iter
        self.C = C

    def add_intercept(self, X):
        if self.fit_intercept:
            return np.c_[np.ones(len(X)), X]

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_large_sparse=False)
        column_or_1d(y)
        sensitive = X[:, self.sensitive_cols] if isinstance(X, np.ndarray) else X[self.sensitive_cols]
        X = self.add_intercept(X)
        n_obs, n_features = X.shape

        theta = cp.Variable(n_features)
        y_hat = X @ theta
        log_likelihood = cp.sum(
            cp.multiply(y, y_hat) -
            cp.log_sum_exp(cp.hstack([np.zeros((n_obs, 1)), cp.reshape(y_hat, (n_obs, 1))]), axis=1) -
            1 / self.C * cp.norm(theta[1:])
        )

        dec_boundary_cov = y_hat @ (sensitive - np.mean(sensitive, axis=0)) / n_obs
        constraints = [cp.abs(dec_boundary_cov) <= self.covariance_threshold]

        problem = cp.Problem(cp.Maximize(log_likelihood), constraints)
        problem.solve(max_iters=self.max_iter)

        if problem.status in ['infeasible', 'unbounded']:
            raise ValueError(f'problem was found to be {problem.status}')

        self.n_iter_ = problem.solver_stats.num_iters
        self.coef_ = theta.value
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['coef_', 'n_iter_'])
        X = check_array(X)
        X = self.add_intercept(X)
        return X @ self.coef_

    def predict(self, X):
        return self.predict_proba(X) > 0.5
