try:
    import cvxpy as cp
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    cp = NotInstalledPackage("cvxpy")

from abc import ABC, abstractmethod

import autograd.numpy as np
import pandas as pd
from autograd import grad
from autograd.test_util import check_grads
from deprecated.sphinx import deprecated
from scipy.optimize import minimize
from scipy.special._ufuncs import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.validation import (
    _check_sample_weight,
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


class BaseScipyMinimizeRegressor(BaseEstimator, RegressorMixin, ABC):
    """
    Base class for regressors relying on scipy's minimze method. Derive a class from this one and give it the function to be minimized.

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Defaults to 1.0.

    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : np.array of shape (n_features,)
        Estimated coefficients of the model.

    intercept_ : float
        Independent term in the linear model. Set to 0.0 if fit_intercept = False.

    Notes
    -----
    This implementation uses scipy.optimize.minimize, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
    """

    def __init__(self, alpha=0.0, l1_ratio=0.0, fit_intercept=True, copy_X=True, positive=False):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive

    @abstractmethod
    def _get_objective(self, X, y, sample_weight):
        """
        Produce the loss function to be minimized, and its gradient to speed up computations.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        loss : Callable[[np.array], float]
            The loss function to be minimized.

        grad_loss : Callable[[np.array], np.array]
            The gradient of the loss function. Speeds up finding the minimum.
        """

    def _loss_regularize(self, loss):
        def regularized_loss(params):
            return (
                loss(params)
                + self.alpha * self.l1_ratio * np.sum(np.abs(params))
                + 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(params ** 2)
            )

        return regularized_loss

    def _grad_loss_regularize(self, grad_loss):
        def regularized_grad_loss(params):
            return (
                grad_loss(params)
                + self.alpha * self.l1_ratio * np.sign(params)
                + self.alpha * (1 - self.l1_ratio) * params
            )

        return regularized_grad_loss

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model using the SLSQP algorithm.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features)
            The training data.

        y : np.array, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        Fitted regressor.
        """
        X_, grad_loss, loss = self._prepare_inputs(X, sample_weight, y)

        d = X_.shape[1] - self.n_features_in_  # This is either zero or one.
        bounds = (
            self.n_features_in_ * [(0, np.inf)] + d * [(-np.inf, np.inf)]
            if self.positive
            else None
        )
        minimize_result = minimize(
            loss,
            x0=np.zeros(self.n_features_in_ + d),
            bounds=bounds,
            method="SLSQP",
            jac=grad_loss,
            tol=1e-20,
        )
        self.convergence_status_ = minimize_result.message

        if self.fit_intercept:
            *self.coef_, self.intercept_ = minimize_result.x
        else:
            self.coef_ = minimize_result.x
            self.intercept_ = 0.0

        self.coef_ = np.array(self.coef_)

        return self

    def _prepare_inputs(self, X, sample_weight, y):
        X, y = check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X)
        self.n_features_in_ = X.shape[1]

        n = X.shape[0]
        if self.copy_X:
            X_ = X.copy()
        else:
            X_ = X
        if self.fit_intercept:
            X_ = np.hstack([X_, np.ones(shape=(n, 1))])

        loss, grad_loss = self._get_objective(X_, y, sample_weight)

        return X_, grad_loss, loss

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : np.array, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.array, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.coef_ + self.intercept_


class LADRegression(BaseScipyMinimizeRegressor):
    """
    Least absolute deviation Regression.

    LADRegression fits a linear model to minimize the residual sum of absolute deviations between
    the observed targets in the dataset, and the targets predicted by the linear approximation, i.e.

        1 / (2 * n_samples) * ||y - Xw||_1
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||_2 ** 2

    Compared to linear regression, this approach is robust to outliers. You can even
    optimize for the lowest MAPE (Mean Average Percentage Error), if you pass in np.abs(1/y_train) for the
    sample_weight keyword when fitting the regressor.

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Defaults to 1.0.

    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.

    Attributes
    ----------
    coef_ : np.array of shape (n_features,)
        Estimated coefficients of the model.

    intercept_ : float
        Independent term in the linear model. Set to 0.0 if fit_intercept = False.

    Notes
    -----
    This implementation uses scipy.optimize.minimize, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([1, 2, 3, 4])
    >>> l = LADRegression().fit(X, y)
    >>> l.coef_
    array([1., 2., 3., 4.])

    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([-1, 2, -3, 4])
    >>> l = LADRegression(positive=True).fit(X, y)
    >>> l.coef_
    array([7.39575926e-18, 1.42423304e+00, 2.80467827e-17, 4.29789588e+00])

    """

    def _get_objective(self, X, y, sample_weight):
        @self._loss_regularize
        def mae_loss(params):
            return np.mean(sample_weight * np.abs(y - X @ params))

        @self._grad_loss_regularize
        def grad_mae_loss(params):
            return -(sample_weight * np.sign(y - X @ params)) @ X / X.shape[0]

        return mae_loss, grad_mae_loss


class ImbalancedLinearRegression(BaseScipyMinimizeRegressor):
    """
    Linear regression where overestimating is `overestimation_punishment_factor` times worse than underestimating.

    A value of `overestimation_punishment_factor=5` implies that overestimations by the model are penalized with a factor of 5
    while underestimations have a default factor of 1. The formula optimized for is

        1 / (2 * n_samples) * switch^T * ||y - Xw||_2 ** 2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||_2 ** 2

    where switch is a vector with value overestimation_punishment_factor if y - Xw < 0, else 1.

    ImbalancedLinearRegression fits a linear model to minimize the residual sum of squares between
    the observed targets in the dataset, and the targets predicted by the linear approximation.
    Compared to normal linear regression, this approach allows for a different treatment of over or under estimations.

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms. Defaults to 1.0.

    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.

    overestimation_punishment_factor : float, default=1
        Factor to punish overestimations more (if the value is larger than 1) or less (if the value is between 0 and 1).

    Attributes
    ----------
    coef_ : np.array of shape (n_features,)
        Estimated coefficients of the model.

    intercept_ : float
        Independent term in the linear model. Set to 0.0 if fit_intercept = False.

    Notes
    -----
    This implementation uses scipy.optimize.minimize, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.randn(100, 4)
    >>> y = X @ np.array([1, 2, 3, 4]) + 2*np.random.randn(100)
    >>> over_bad = ImbalancedLinearRegression(overestimation_punishment_factor=50).fit(X, y)
    >>> over_bad.coef_
    array([0.36267036, 1.39526844, 3.4247146 , 3.93679175])

    >>> under_bad = ImbalancedLinearRegression(overestimation_punishment_factor=0.01).fit(X, y)
    >>> under_bad.coef_
    array([0.73519586, 1.28698197, 2.61362614, 4.35989806])

    """

    def __init__(self, alpha=0.0, l1_ratio=0.0, fit_intercept=True, copy_X=True, positive=False,
                 overestimation_punishment_factor=1.0):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive)
        self.overestimation_punishment_factor = overestimation_punishment_factor

    def _get_objective(self, X, y, sample_weight):
        @self._loss_regularize
        def imbalanced_loss(params):
            return 0.5 * np.mean(
                sample_weight
                * np.where(X @ params > y, self.overestimation_punishment_factor, 1)
                * np.square(y - X @ params)
            )

        @self._grad_loss_regularize
        def grad_imbalanced_loss(params):
            return (
                -(
                    sample_weight
                    * np.where(X @ params > y, self.overestimation_punishment_factor, 1)
                    * (y - X @ params)
                )
                @ X
                / X.shape[0]
            )

        return imbalanced_loss, grad_imbalanced_loss
