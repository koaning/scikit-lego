try:
    import cvxpy as cp
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    cp = NotInstalledPackage("cvxpy")

import autograd.numpy as np
import pandas as pd
from autograd import grad
from autograd.test_util import check_grads
from deprecated.sphinx import deprecated
from scipy.special._ufuncs import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.validation import (
    check_is_fitted,
    check_array,
    FLOAT_DTYPES,
    column_or_1d,
)


class LowessRegression(BaseEstimator, RegressorMixin):
    """
    Does LowessRegression. Note that this *can* get expensive to predict.

    :param sigma: float, how wide we will smooth the data
    :param span: float, what percentage of the data is to be used. Defaults to using all data.
    """

    def __init__(self, sigma=1, span=None):
        self.sigma = sigma
        self.span = span

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if self.span is not None:
            if not 0 <= self.span <= 1:
                raise ValueError(
                    f"Param `span` must be 0 <= span <= 1, got: {self.span}"
                )
        if self.sigma < 0:
            raise ValueError(f"Param `sigma` must be >= 0, got: {self.sigma}")
        self.X_ = X
        self.y_ = y
        return self

    def _calc_wts(self, x_i):
        distances = np.array(
            [np.linalg.norm(self.X_[i, :] - x_i) for i in range(self.X_.shape[0])]
        )
        weights = np.exp(-(distances ** 2) / self.sigma)
        if self.span:
            weights = weights * (distances <= np.quantile(distances, q=self.span))
        return weights

    def predict(self, X):
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: Returns an array of predictions shape=(n_samples,)
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["X_", "y_"])
        results = np.zeros(X.shape[0])
        for idx in range(X.shape[0]):
            results[idx] = np.average(self.y_, weights=self._calc_wts(x_i=X[idx, :]))
        return results


class ProbWeightRegression(BaseEstimator, RegressorMixin):
    """
    This regressor assumes that all input signals in `X` need to be reweighted
    with weights that sum up to one in order to predict `y`. This can be very useful
    in combination with `sklego.meta.EstimatorTransformer` because it allows you
    to construct an ensemble.

    :param non_negative: boolean, default=True, setting that forces all weights to be >= 0
    """

    def __init__(self, non_negative=True):
        self.non_negative = non_negative

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        # Construct the problem.
        betas = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X * betas - y))
        constraints = [sum(betas) == 1]
        if self.non_negative:
            constraints.append(0 <= betas)

        # Solve the problem.
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.coefs_ = betas.value
        return self

    def predict(self, X):
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: Returns an array of predictions shape=(n_samples,)
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["coefs_"])
        return np.dot(X, self.coefs_)


class DeadZoneRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        threshold=0.3,
        relative=False,
        effect="linear",
        n_iter=2000,
        stepsize=0.01,
        check_grad=False,
    ):
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
                return np.where(
                    errors > self.threshold, errors ** 2, np.zeros(errors.shape)
                )

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
            check_grads(training_loss, modes=["rev"])(weights)

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
        check_is_fitted(self, ["coefs_"])
        return np.dot(X, self.coefs_)


class _FairClassifier(BaseEstimator, LinearClassifierMixin):
    def __init__(
        self,
        sensitive_cols=None,
        C=1.0,
        penalty="l1",
        fit_intercept=True,
        max_iter=100,
        train_sensitive_cols=False,
    ):
        self.sensitive_cols = sensitive_cols
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.max_iter = max_iter
        self.train_sensitive_cols = train_sensitive_cols
        self.C = C

    def fit(self, X, y):
        if self.penalty not in ["l1", "none"]:
            raise ValueError(
                f"penalty should be either 'l1' or 'none', got {self.penalty}"
            )

        self.sensitive_col_idx_ = self.sensitive_cols
        if isinstance(X, pd.DataFrame):
            self.sensitive_col_idx_ = [
                i for i, name in enumerate(X.columns) if name in self.sensitive_cols
            ]
        X, y = check_X_y(X, y, accept_large_sparse=False)

        sensitive = X[:, self.sensitive_col_idx_]
        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)
        X = self._add_intercept(X)

        column_or_1d(y)
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)
        self.classes_ = label_encoder.classes_

        if len(self.classes_) > 2:
            raise ValueError(
                f"This solver needs samples of exactly 2 classes"
                f" in the data, but the data contains {len(self.classes_)}: {self.classes_}"
            )

        self._solve(sensitive, X, y)
        return self

    def constraints(self, y_hat, y_true, sensitive, n_obs):
        raise NotImplementedError(
            "subclasses of _FairClassifier should implement constraints"
        )

    def _solve(self, sensitive, X, y):
        n_obs, n_features = X.shape
        theta = cp.Variable(n_features)
        y_hat = X @ theta

        log_likelihood = cp.sum(
            cp.multiply(y, y_hat)
            - cp.log_sum_exp(
                cp.hstack([np.zeros((n_obs, 1)), cp.reshape(y_hat, (n_obs, 1))]), axis=1
            )
        )
        if self.penalty == "l1":
            log_likelihood -= cp.sum((1 / self.C) * cp.norm(theta[1:]))

        constraints = self.constraints(y_hat, y, sensitive, n_obs)

        problem = cp.Problem(cp.Maximize(log_likelihood), constraints)
        problem.solve(max_iters=self.max_iter)

        if problem.status in ["infeasible", "unbounded"]:
            raise ValueError(f"problem was found to be {problem.status}")

        self.n_iter_ = problem.solver_stats.num_iters

        if self.fit_intercept:
            self.coef_ = theta.value[np.newaxis, 1:]
            self.intercept_ = theta.value[0:1]
        else:
            self.coef_ = theta.value[np.newaxis, :]
            self.intercept_ = np.array([0.0])

    def predict_proba(self, X):
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        return expit(decision_2d)

    def decision_function(self, X):
        X = check_array(X)

        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)
        return super().decision_function(X)

    def _add_intercept(self, X):
        if self.fit_intercept:
            return np.c_[np.ones(len(X)), X]


class DemographicParityClassifier(BaseEstimator, LinearClassifierMixin):
    r"""
    A logistic regression classifier which can be constrained on demographic parity (p% score).

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

    :param covariance_threshold:
        The maximum allowed covariance between the sensitive attributes and the distance to the
        decision boundary. If set to None, no fairness constraint is enforced
    :param sensitive_cols:
        List of sensitive column names(when X is a dataframe)
        or a list of column indices when X is a numpy array.
    :param C:
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    :param penalty: Used to specify the norm used in the penalization. Expects 'none' or 'l1'
    :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    :param max_iter: Maximum number of iterations taken for the solvers to converge.
    :param train_sensitive_cols: Indicates whether the model should use the sensitive columns in the fit step.
    :param multi_class: The method to use for multiclass predictions
    :param n_jobs: The amount of parallel jobs thata should be used to fit multiclass models

    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):

        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[
            multi_class
        ]
        return multiclass_meta(
            _DemographicParityClassifier(*args, **kwargs), n_jobs=n_jobs
        )


@deprecated(
    version="0.4.0",
    reason="Please use `sklego.linear_model.DemographicParityClassifier instead`",
)
class FairClassifier(DemographicParityClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _DemographicParityClassifier(_FairClassifier):
    def __init__(
        self,
        covariance_threshold,
        sensitive_cols=None,
        C=1.0,
        penalty="l1",
        fit_intercept=True,
        max_iter=100,
        train_sensitive_cols=False,
    ):
        super().__init__(
            sensitive_cols=sensitive_cols,
            C=C,
            penalty=penalty,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            train_sensitive_cols=train_sensitive_cols,
        )

        self.covariance_threshold = covariance_threshold

    def constraints(self, y_hat, y_true, sensitive, n_obs):
        if self.covariance_threshold is not None:
            dec_boundary_cov = y_hat @ (sensitive - np.mean(sensitive, axis=0)) / n_obs
            return [cp.abs(dec_boundary_cov) <= self.covariance_threshold]
        else:
            return []


class EqualOpportunityClassifier(BaseEstimator, LinearClassifierMixin):
    r"""
    A logistic regression classifier which can be constrained on equal opportunity score.

    Minimizes the Log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier for those examples that have a y_true of 1.

    Only works for binary classification problems

    .. math::
       \begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right) d
        \boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}

    where POS is the subset of the population where y_true = 1


    :param covariance_threshold:
        The maximum allowed covariance between the sensitive attributes and the distance to the
        decision boundary. If set to None, no fairness constraint is enforced
    :param positive_target: The name of the class which is associated with a positive outcome
    :param sensitive_cols:
        List of sensitive column names(when X is a dataframe)
        or a list of column indices when X is a numpy array.
    :param C:
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger regularization.
    :param penalty: Used to specify the norm used in the penalization. Expects 'none' or 'l1'
    :param fit_intercept: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
    :param max_iter: Maximum number of iterations taken for the solvers to converge.
    :param train_sensitive_cols: Indicates whether the model should use the sensitive columns in the fit step.
    :param multi_class: The method to use for multiclass predictions
    :param n_jobs: The amount of parallel jobs thata should be used to fit multiclass models

    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):

        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[
            multi_class
        ]
        return multiclass_meta(
            _EqualOpportunityClassifier(*args, **kwargs), n_jobs=n_jobs
        )


class _EqualOpportunityClassifier(_FairClassifier):
    def __init__(
        self,
        covariance_threshold,
        positive_target,
        sensitive_cols=None,
        C=1.0,
        penalty="l1",
        fit_intercept=True,
        max_iter=100,
        train_sensitive_cols=False,
    ):
        super().__init__(
            sensitive_cols=sensitive_cols,
            C=C,
            penalty=penalty,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            train_sensitive_cols=train_sensitive_cols,
        )
        self.positive_target = positive_target
        self.covariance_threshold = covariance_threshold

    def constraints(self, y_hat, y_true, sensitive, n_obs):
        if self.covariance_threshold is not None:
            n_obs = len(y_true[y_true == self.positive_target])
            dec_boundary_cov = (
                y_hat[y_true == self.positive_target]
                @ (
                    sensitive[y_true == self.positive_target]
                    - np.mean(sensitive, axis=0)
                )
                / n_obs
            )
            return [cp.abs(dec_boundary_cov) <= self.covariance_threshold]
        else:
            return []
