try:
    import cvxpy as cp
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    cp = NotInstalledPackage("cvxpy")
import logging
from abc import ABC, abstractmethod
from inspect import signature
from warnings import warn

import narwhals.stable.v1 as nw
import numpy as np
from scipy.optimize import minimize
from scipy.special._ufuncs import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    _check_sample_weight,
    check_is_fitted,
    column_or_1d,
)
from sklearn_compat.utils.validation import check_array, validate_data


class LowessRegression(RegressorMixin, BaseEstimator):
    """`LowessRegression` estimator: LOWESS (Locally Weighted Scatterplot Smoothing) is a type of
    [local regression](https://en.wikipedia.org/wiki/Local_regression).

    !!! warning
        This model *can* get expensive to predict.
        In fact the prediction step needs to compute the distance between each sample to predict `x_i` with all the
        training samples.

    Parameters
    ----------
    sigma : float, default=1.0
        The bandwidth parameter that determines the width of the smoothing.
    span : float | None, default=None
        The fraction of data points to consider during smoothing.

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, n_features)
        The training data.
    y_ : np.ndarray of shape (n_samples,)
        The target (training) values.


    Examples
    --------
    ```python
    from sklego.linear_model import LowessRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=100, n_features=2, noise=10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lowess = LowessRegression(sigma=1, span=0.5)
    lowess.fit(X_train, y_train)

    y_pred = lowess.predict(X_test)
    print(y_pred)
    ```
    """

    def __init__(self, sigma=1, span=None):
        self.sigma = sigma
        self.span = span

    def fit(self, X, y):
        """Fit the estimator on training data `X` and `y` by storing them in `self.X_` and `self.y_`, and
        validating the parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : LowessRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `span` is not between 0 and 1.
            - If `sigma` is negative.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)
        if self.span is not None:
            if not 0 <= self.span <= 1:
                raise ValueError(f"Param `span` must be 0 <= span <= 1, got: {self.span}")
        if self.sigma < 0:
            raise ValueError(f"Param `sigma` must be >= 0, got: {self.sigma}")
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def _calc_wts(self, x_i):
        """Calculate the weights for a single point `x_i` using the training data `self.X_` and the parameters
        `self.sigma` and `self.span`. The weights are calculated as `np.exp(-(distances**2) / self.sigma)`,
        where distances are the distances between `x_i` and all the training samples.

        If `self.span` is not None, then the weights are multiplied by
        `(distances <= np.quantile(distances, q=self.span))`.
        """
        distances = np.linalg.norm(self.X_ - x_i, axis=1)
        weights = np.exp(-(distances**2) / self.sigma)
        if self.span:
            weights = weights * (distances <= np.quantile(distances, q=self.span))
        return weights

    def predict(self, X):
        """Predict target values for `X` using fitted estimator. This process is expensive because it needs to compute
        the distance between each sample `x_i` with all the training samples.

        Then it calculates the weights for **each sample** `x_i` as `np.exp(-(distances**2) / self.sigma)` and finally
        it computes the weighted average of the `y` values weighted by these weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["X_", "y_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        try:
            results = np.stack([np.average(self.y_, weights=self._calc_wts(x_i=x_i)) for x_i in X])
        except ZeroDivisionError:
            msg = (
                "Weights, resulting from `np.exp(-(distances**2) / self.sigma)`, are all zero. "
                "Try to increase the value of `sigma` or to normalize the input data.\n\n"
                "`distances` refer to the distance between each sample `x_i` with all the"
                "training samples."
            )
            raise ValueError(msg)

        return results


class ProbWeightRegression(RegressorMixin, BaseEstimator):
    """`ProbWeightRegression` assumes that all input signals in `X` need to be reweighted with weights that sum up to
    one in order to predict `y`.

    This can be very useful in combination with `sklego.meta.EstimatorTransformer` because it allows to construct
    an ensemble.

    Parameters
    ----------
    non_negative : bool, default=True
        If True, forces all weights to be non-negative.

    Attributes
    ----------
    n_features_in_ : int
        The number of features seen during `fit`.
    coef_ : np.ndarray, shape (n_columns,)
        The learned coefficients after fitting the model.
    coefs_ : np.ndarray, shape (n_columns,)
        Deprecated, please use `coef_` instead.

    Examples
    --------
    ```python
    import numpy as np
    from sklego.linear_model import ProbWeightRegression

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])

    pwr = ProbWeightRegression().fit(X, y)

    # The weights sum up to 1
    assert np.isclose(pwr.coef_.sum(), 1)

    X_test = np.array([[5, 6], [6, 7]])

    # The prediction is positive (all weights are positive, and features are positive)
    assert all(pwr.predict(X_test) > 0)

    # The weights are positive
    assert all(pwr.coef_ > -1e-8)
    ```

    !!! info

        This model requires [`cvxpy`](https://www.cvxpy.org/) to be installed. If you don't have it installed, you can
        install it with:

        ```bash
        pip install cvxpy
        # or pip install scikit-lego"[cvxpy]"
        ```
    """

    def __init__(self, non_negative=True):
        self.non_negative = non_negative

    def fit(self, X, y):
        r"""Fit the estimator on training data `X` and `y` by solving the following convex optimization problem:

        $$\begin{array}{ll}{\operatorname{minimize}} & {\sum_{i=1}^{N}\left(\mathbf{x}_{i}
        \boldsymbol{\beta}-y_{i}\right)^{2}} \\
        {\text { subject to }} & {\sum_{j=1}^{p} \beta_{j}=1} \\
        {(\text{If non_negative=True})} & {\beta_{j} \geq 0, \quad j=1, \ldots, p} \end{array}$$

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : ProbWeightRegression
            The fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)

        # Construct the problem.
        betas = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ betas - y))
        constraints = [sum(betas) == 1]
        if self.non_negative:
            constraints.append(0 <= betas)

        # Solve the problem.
        prob = cp.Problem(objective, constraints)
        prob.solve()
        self.coef_ = betas.value
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by multiplying `X` with the learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["coef_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        return np.dot(X, self.coef_)

    @property
    def coefs_(self):
        warn(
            "Please use `coef_` instead of `coefs_`, `coefs_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.coef_


class DeadZoneRegressor(RegressorMixin, BaseEstimator):
    r"""The `DeadZoneRegressor` estimator implements a regression model that incorporates a _dead zone effect_ for
    improving the robustness of regression predictions.

    The dead zone effect allows the model to reduce the impact of small errors in the training data on the regression
    results, which can be particularly useful when dealing with noisy or unreliable data.

    The estimator minimizes the following loss function using gradient descent:

    $$\frac{1}{n} \sum_{i=1}^{n} \text{deadzone}\left(\left|X_i \cdot w - y_i\right|\right)$$

    where:

    $$\text{deadzone}(e) =
    \begin{cases}
    1 & \text{if } e > \text{threshold} \text{ & effect="constant"} \\
    e & \text{if } e > \text{threshold} \text{ & effect="linear"} \\
    e^2 & \text{if } e > \text{threshold} \text{ & effect="quadratic"} \\
    0 & \text{otherwise}
    \end{cases}
    $$

    Parameters
    ----------
    threshold : float, default=0.3
        The threshold value for the dead zone effect.
    relative : bool, default=False
        If True, the threshold is relative to the target value. Namely the _dead zone effect_ is applied to the
        relative error between the predicted and target values.
    effect : Literal["linear", "quadratic", "constant"], default="linear"
        The type of dead zone effect to apply. It can be one of the following:

        - "linear": the errors within the threshold have no impact (their contribution is effectively zero), and errors
            outside the threshold are penalized linearly.
        - "quadratic": the errors within the threshold have no impact (their contribution is effectively zero), and
            errors outside the threshold are penalized quadratically (squared).
        - "constant": the errors within the threshold have no impact, and errors outside the threshold are penalized
            with a constant value.
    n_iter : int, default=2000
        The number of iterations to run the gradient descent algorithm.
    stepsize : float, default=0.01
        The step size for the gradient descent algorithm.
    check_grad : bool, default=False
        If True, check the gradients numerically, _just to be safe_.

    Attributes
    ----------
    coef_ : np.ndarray, shape (n_columns,)
        The learned coefficients after fitting the model.
    coefs_ : np.ndarray, shape (n_columns,)
        Deprecated, please use `coef_` instead.

    Examples
    --------

    ```python
    import numpy as np
    from sklego.linear_model import DeadZoneRegressor

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([1, 2, 3, 4])

    dzr = DeadZoneRegressor(threshold=0.5, relative=False, effect="quadratic").fit(X, y)

    X_test = np.array([[5, 6], [6, 7]])
    y_pred = dzr.predict(X_test)

    print(y_pred)
    ```
    """

    _ALLOWED_EFFECTS = ("linear", "quadratic", "constant")

    def __init__(
        self,
        threshold=0.3,
        relative=False,
        effect="linear",
    ):
        self.threshold = threshold
        self.relative = relative
        self.effect = effect

    def fit(self, X, y):
        """Fit the estimator on training data `X` and `y` by optimizing the loss function using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : DeadZoneRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If `effect` is not one of "linear", "quadratic" or "constant".
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)

        if self.effect not in self._ALLOWED_EFFECTS:
            raise ValueError(f"effect {self.effect} must be in {self._ALLOWED_EFFECTS}")

        def deadzone(errors):
            if self.effect == "constant":
                error_weight = errors.shape[0]
            elif self.effect == "linear":
                error_weight = errors
            elif self.effect == "quadratic":
                error_weight = errors**2

            return np.where(errors > self.threshold, error_weight, 0.0)

        def training_loss(weights):
            prediction = np.dot(X, weights)
            errors = np.abs(prediction - y)

            if self.relative:
                errors /= np.abs(y)

            loss = np.mean(deadzone(errors))
            return loss

        def deadzone_derivative(errors):
            if self.effect == "constant":
                error_weight = 0.0
            elif self.effect == "linear":
                error_weight = 1.0
            elif self.effect == "quadratic":
                error_weight = 2 * errors

            return np.where(errors > self.threshold, error_weight, 0.0)

        def training_loss_derivative(weights):
            prediction = np.dot(X, weights)
            errors = np.abs(prediction - y)

            if self.relative:
                errors /= np.abs(y)

            loss_derivative = deadzone_derivative(errors)
            errors_derivative = np.sign(prediction - y)

            if self.relative:
                errors_derivative /= np.abs(y)

            derivative = np.dot(errors_derivative * loss_derivative, X) / X.shape[0]

            return derivative

        self.n_features_in_ = X.shape[1]

        minimize_result = minimize(
            training_loss,
            x0=np.zeros(self.n_features_in_),  # np.random.normal(0, 1, n_features_)
            tol=1e-20,
            jac=training_loss_derivative,
        )

        self.convergence_status_ = minimize_result.message
        self.coef_ = minimize_result.x
        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by multiplying `X` with the learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["coef_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        return np.dot(X, self.coef_)

    @property
    def coefs_(self):
        warn(
            "Please use `coef_` instead of `coefs_`, `coefs_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.coef_

    @property
    def allowed_effects(self):
        warn(
            "Please use `_ALLOWED_EFFECTS` instead of `allowed_effects`,"
            "`allowed_effects` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self._ALLOWED_EFFECTS


class _FairClassifier(LinearClassifierMixin, BaseEstimator):
    """Base class for fair classifiers that address sensitive attribute fairness.

    This base class provides a foundation for fair classifiers that aim to mitigate bias and discrimination by taking
    into account sensitive attributes during the classification process.

    This estimator works by solving a constrained optimization problem that maximizes the log-likelihood of the
    training data while satisfying fairness constraints.

    !!! warning
        The classification problem should be a binary one.

    Parameters
    ----------
    sensitive_cols : List[str] | List[int] | None, default=None
        A list of column names or column indexes in the input data that represent sensitive attributes.
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Smaller values specify stronger regularization.
    penalty : Literal["l1", "l2", "none", None], default="l1"
        The type of penalty to apply to the model. "l1" applies L1 regularization, "l2" applies L2 regularization,
        while None (or "none") disables regularization.
    fit_intercept : bool, default=True
        Whether or not to fit an intercept term. If True, an intercept term is added to the model.
    max_iter : int, default=100
        Maximum number of iterations for the solver.
    train_sensitive_cols : bool, default=False
        Whether or not to include sensitive columns during training. If False, sensitive columns are removed from the
        input data during training.

    Attributes
    ----------
    sensitive_col_idx_ : array
        Indices of columns in the input data that correspond to sensitive attributes.
    classes_ : array-like of shape (n_classes,)
        The classes seen at `fit`.

    Note
    ----
    Subclasses of `_FairClassifier` should implement the `constraints` method to specify fairness constraints.
    This class primarily handles the logistic regression optimization and preprocessing steps.

    `_FairClassifier` should not be used directly; it serves as a base class for fair classification models.
    """

    _ALLOWED_PENALTIES = ("l1", "l2", "none", None)

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
        """Fit the estimator on training data `X` and `y`.

        It handles preprocessing and optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : _FairClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If `penalty` is not one of "l1", "l2", "none" or None.
        """
        if self.penalty not in self._ALLOWED_PENALTIES:
            raise ValueError(f"penalty should be one of {self._ALLOWED_PENALTIES}, got {self.penalty}")

        if self.penalty == "none":
            warn(
                "Please use `penalty=None` instead of `penalty='none'`, 'none' will be deprecated in future versions",
                DeprecationWarning,
            )

        self.sensitive_col_idx_ = self.sensitive_cols
        X = nw.from_native(X, eager_only=True, strict=False)

        if isinstance(X, nw.DataFrame):
            self.sensitive_col_idx_ = [i for i, name in enumerate(X.columns) if name in self.sensitive_cols]

        X, y = check_X_y(X, y, accept_large_sparse=False)
        sensitive = X[:, self.sensitive_col_idx_]

        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)

        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        if X.shape[1] == 0:
            msg = "Cannot fit the model, at least 1 feature(s) is required."
            raise ValueError(msg)

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
        raise NotImplementedError("subclasses of `_FairClassifier` should implement constraints")

    def _solve(self, sensitive, X, y):
        """Solve the optimization problem for the fair classifier."""
        n_obs, n_features = X.shape
        theta = cp.Variable(n_features)
        y_hat = X @ theta

        log_likelihood = cp.sum(
            cp.multiply(y, y_hat)
            - cp.log_sum_exp(cp.hstack([np.zeros((n_obs, 1)), cp.reshape(y_hat, (n_obs, 1))]), axis=1)
        )

        if self.penalty == "l1":
            log_likelihood -= cp.norm(theta[int(self.fit_intercept) :], 1) / self.C

        elif self.penalty == "l2":
            log_likelihood -= cp.norm(theta[int(self.fit_intercept) :], 2) / self.C

        constraints = self.constraints(y_hat, y, sensitive, n_obs)

        problem = cp.Problem(cp.Maximize(log_likelihood), constraints)

        if "max_iters" in signature(problem.solve).parameters:
            kwargs = {"max_iters": self.max_iter}
        else:
            if self.max_iter:
                logging.warning("solver does not support `max_iters` and the argument will be ignored")
            kwargs = {}

        problem.solve(**kwargs)

        if problem.status in ["infeasible", "unbounded"]:
            raise ValueError(f"problem was found to be {problem.status}")

        self.n_iter_ = getattr(problem.solver_stats, "num_iters", 0)

        if self.fit_intercept:
            self.coef_ = theta.value[np.newaxis, 1:]
            self.intercept_ = theta.value[0:1]
        else:
            self.coef_ = theta.value[np.newaxis, :]
            self.intercept_ = np.array([0.0])

    def predict_proba(self, X):
        """Predict class probabilities for `X`.

        This method predicts class probabilities for input data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, 2)
            The predicted class probabilities for each data point.
        """
        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        return expit(decision_2d)

    def decision_function(self, X):
        X = check_array(X)

        if not self.train_sensitive_cols:
            X = np.delete(X, self.sensitive_col_idx_, axis=1)
        return super().decision_function(X)

    def _more_tags(self):
        return {"poor_score": True}


class DemographicParityClassifier(LinearClassifierMixin, BaseEstimator):
    r"""`DemographicParityClassifier` is a logistic regression classifier which can be constrained on demographic
    parity (p% score).

    It minimizes the log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier.

    !!! warning
        This classifier only works for binary classification problems.

    $$\begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_\boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}$$

    Parameters
    ----------
    covariance_threshold : float | None
        The maximum allowed covariance between the sensitive attributes and the distance to the decision boundary.
        If set to None, no fairness constraint is enforced.
    sensitive_cols : List[str] | List[int] | None, default=None
        List of sensitive column names (if X is a dataframe) or a list of column indices (if X is a numpy array).
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values
        specify stronger regularization.
    penalty : Literal["l1", "l2", "none", None], default="l1"
        The type of penalty to apply to the model. "l1" applies L1 regularization, "l2" applies L2 regularization,
        while None (or "none") disables regularization.
    fit_intercept : bool, default=True
        Whether or not a constant term (a.k.a. bias or intercept) should be added to the decision function.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    train_sensitive_cols : bool, default=False
        Indicates whether the model should use the sensitive columns in the fit step.
    multi_class : Literal["ovr", "ovo"], default="ovr"
        The method to use for multiclass predictions.
    n_jobs : int | None, default=1
        The amount of parallel jobs that should be used to fit the model.

    Source
    ------
    M. Zafar et al. (2017), Fairness Constraints: Mechanisms for Fair Classification


    Examples
    --------
    ```python
    from sklego.linear_model import DemographicParityClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    dp = DemographicParityClassifier(
        covariance_threshold=0.1, sensitive_cols=[0]
    ).fit(X_train, y_train)

    y_pred = dp.predict_proba(X_test)

    print(y_pred)
    ```
    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):
        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[multi_class]
        return multiclass_meta(_DemographicParityClassifier(*args, **kwargs), n_jobs=n_jobs)


class _DemographicParityClassifier(_FairClassifier):
    """Classifier for Demographic Parity fairness constraint.

    This classifier extends the `_FairClassifier` and adds a Demographic Parity fairness constraint.
    Demographic Parity ensures that the probability of a positive outcome is the same for all groups defined by
    sensitive attributes.

    The class implements Demographic Parity fairness constraint to the `_FairClassifier` optimization problem.
    """

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
        """Implement the Demographic Parity fairness constraint."""
        if self.covariance_threshold is not None:
            dec_boundary_cov = y_hat @ (sensitive - np.mean(sensitive, axis=0)) / n_obs
            return [cp.abs(dec_boundary_cov) <= self.covariance_threshold]
        else:
            return []


class EqualOpportunityClassifier(LinearClassifierMixin, BaseEstimator):
    r"""`EqualOpportunityClassifier` is a logistic regression classifier which can be constrained on equal opportunity
    score.

    It minimizes the log loss while constraining the correlation between the specified `sensitive_cols` and the
    distance to the decision boundary of the classifier for those examples that have a y_true of 1.

    !!! warning
        This classifier only works for binary classification problems.

    $$\begin{array}{cl}{\operatorname{minimize}} & -\sum_{i=1}^{N} \log p\left(y_{i} | \mathbf{x}_{i},
        \boldsymbol{\theta}\right) \\
        {\text { subject to }} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_\boldsymbol{\theta}\left(\mathbf{x}_{i}\right) \leq \mathbf{c}} \\
        {} & {\frac{1}{POS} \sum_{i=1}^{POS}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
        d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right) \geq-\mathbf{c}}\end{array}$$

    where POS is the subset of the population where $\text{y_true} = 1$

    Parameters
    ----------
    covariance_threshold : float | None
        The maximum allowed covariance between the sensitive attributes and the distance to the decision boundary.
        If set to None, no fairness constraint is enforced.
    positive_target : int
        The name of the class which is associated with a positive outcome
    sensitive_cols : List[str] | List[int] | None, default=None
        List of sensitive column names (if X is a dataframe) or a list of column indices (if X is a numpy array).
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values
        specify stronger regularization.
    penalty : Literal["l1", "l2", "none", None], default="l1"
        The type of penalty to apply to the model. "l1" applies L1 regularization, "l2" applies L2 regularization,
        while None (or "none") disables regularization.
    fit_intercept : bool, default=True
        Whether or not a constant term (a.k.a. bias or intercept) should be added to the decision function.
    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.
    train_sensitive_cols : bool, default=False
        Indicates whether the model should use the sensitive columns in the fit step.
    multi_class : Literal["ovr", "ovo"], default="ovr"
        The method to use for multiclass predictions.
    n_jobs : int | None, default=1
        The amount of parallel jobs that should be used to fit the model.

    Examples
    --------

    ```python
    from sklego.linear_model import EqualOpportunityClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    eo = EqualOpportunityClassifier(
        covariance_threshold=0.1, positive_target=1, sensitive_cols=[0]
    ).fit(X_train, y_train)

    y_pred = eo.predict_proba(X_test)

    print(y_pred)
    ```
    """

    def __new__(cls, *args, multi_class="ovr", n_jobs=1, **kwargs):
        multiclass_meta = {"ovr": OneVsRestClassifier, "ovo": OneVsOneClassifier}[multi_class]
        return multiclass_meta(_EqualOpportunityClassifier(*args, **kwargs), n_jobs=n_jobs)


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
                @ (sensitive[y_true == self.positive_target] - np.mean(sensitive, axis=0))
                / n_obs
            )
            return [cp.abs(dec_boundary_cov) <= self.covariance_threshold]
        else:
            return []


class BaseScipyMinimizeRegressor(RegressorMixin, BaseEstimator, ABC):
    """Abstract base class for regressors relying on Scipy's
    [minimize method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) to minimize a
    (custom) loss function.

    Derive a class from this one and give it the function to be minimized. The derived class should implement the
    `_get_objective` method, which should return the loss function and its gradient.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.positive = positive
        self.method = method

    @abstractmethod
    def _get_objective(self, X, y, sample_weight):
        """Produce the loss function to be minimized, and its gradient to speed up computations.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.
        y : np.ndarray of shape (n_samples,)
            The target values.
        sample_weight : np.ndarray of shape (n_samples,) | None, default=None
            Individual weights for each sample.

        Returns
        -------
        loss : Callable[[np.ndarray], float]
            The loss function to be minimized.
        grad_loss : Callable[[np.ndarray], np.ndarray]
            The gradient of the loss function. Speeds up finding the minimum.
        """
        ...

    def _regularized_loss(self, params):
        return +self.alpha * self.l1_ratio * np.sum(np.abs(params)) + 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(
            params**2
        )

    def _regularized_grad_loss(self, params):
        return +self.alpha * self.l1_ratio * np.sign(params) + self.alpha * (1 - self.l1_ratio) * params

    def fit(self, X, y, sample_weight=None):
        """Fit the linear model on training data `X` and `y` by optimizing the loss function using gradient descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,) | None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : BaseScipyMinimizeRegressor
            Fitted linear model.
        """
        if self.method not in {"SLSQP", "TNC", "L-BFGS-B"}:
            msg = f"method should be one of 'SLSQP', 'TNC', 'L-BFGS-B', got {self.method} instead"
            raise ValueError(msg)

        X_, grad_loss, loss = self._prepare_inputs(X, sample_weight, y)

        d = X_.shape[1] - self.n_features_in_  # This is either zero or one.
        bounds = self.n_features_in_ * [(0, np.inf)] + d * [(-np.inf, np.inf)] if self.positive else None
        minimize_result = minimize(
            loss,
            x0=np.zeros(self.n_features_in_ + d),
            bounds=bounds,
            method=self.method,
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
        """Prepare the inputs for the optimization problem.

        This method is called by `fit` to prepare the inputs for the optimization problem. It adds an intercept column
        to `X` if `fit_intercept=True`, and returns the loss function and its gradient.
        """
        X, y = validate_data(self, X=X, y=y, y_numeric=True, reset=True)

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
        """Predict target values for `X` using fitted linear model by multiplying `X` with the learned coefficients.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        return X @ self.coef_ + self.intercept_


class ImbalancedLinearRegression(BaseScipyMinimizeRegressor):
    r"""Linear regression where overestimating is `overestimation_punishment_factor` times worse than underestimating.

    A value of `overestimation_punishment_factor=5` implies that overestimations by the model are penalized with a
    factor of 5 while underestimations have a default factor of 1. The formula optimized for is

    $$\frac{1}{2 N} \|s \circ (y - Xw) \|_2^2 + \alpha \cdot l_1 \cdot\|w\|_1 + \frac{\alpha}{2} \cdot (1-l_1)\cdot
    \|w\|_2^2$$

    where $\circ$ is component-wise multiplication and

    $$ s = \begin{cases}
    \text{overestimation_punishment_factor} & \text{if } y - Xw < 0 \\
    1 & \text{otherwise}
    \end{cases}
    $$

    `ImbalancedLinearRegression` fits a linear model to minimize the residual sum of squares between the observed
    targets in the dataset, and the targets predicted by the linear approximation.
    Compared to normal linear regression, this approach allows for a different treatment of over or under estimations.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.
    overestimation_punishment_factor : float, default=1.0
        Factor to punish overestimations more (if the value is larger than 1) or less (if the value is between 0 and 1).

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.linear_model import ImbalancedLinearRegression

    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4]) + 2*np.random.randn(100)

    over_bad = ImbalancedLinearRegression(overestimation_punishment_factor=50).fit(X, y)
    over_bad.coef_
    # array([0.36267036, 1.39526844, 3.4247146 , 3.93679175])

    under_bad = ImbalancedLinearRegression(overestimation_punishment_factor=0.01).fit(X, y)
    under_bad.coef_
    # array([0.73519586, 1.28698197, 2.61362614, 4.35989806])
    ```
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
        overestimation_punishment_factor=1.0,
    ):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive, method)
        self.overestimation_punishment_factor = overestimation_punishment_factor

    def _get_objective(self, X, y, sample_weight):
        def imbalanced_loss(params):
            return 0.5 * np.average(
                np.where(X @ params > y, self.overestimation_punishment_factor, 1) * np.square(y - X @ params),
                weights=sample_weight,
            ) + self._regularized_loss(params)

        def grad_imbalanced_loss(params):
            return (
                -(sample_weight * np.where(X @ params > y, self.overestimation_punishment_factor, 1) * (y - X @ params))
                @ X
                / sample_weight.sum()
            ) + self._regularized_grad_loss(params)

        return imbalanced_loss, grad_imbalanced_loss


class QuantileRegression(BaseScipyMinimizeRegressor):
    r"""Compute quantile regression. This can be used for computing confidence intervals of linear regressions.

    `QuantileRegression` fits a linear model to minimize a weighted residual sum of absolute deviations between
    the observed targets in the dataset and the targets predicted by the linear approximation, i.e.

    $$\frac{\text{switch} \cdot ||y - Xw||_1}{2 N} + \alpha \cdot l_1 \cdot ||w||_1
        + \frac{\alpha}{2} \cdot (1 - l_1) \cdot ||w||^2_2$$

    where

    $$\text{switch} = \begin{cases}
    \text{quantile} & \text{if } y - Xw < 0 \\
    1-\text{quantile} & \text{otherwise}
    \end{cases}$$

    The regressor defaults to `LADRegression` for its default value of `quantile=0.5`.

    Compared to linear regression, this approach is robust to outliers.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    !!! warning
        If, while fitting the model, `sample_weight` contains any zero values, some solvers may not converge properly.
        We would expect that a sample weight of zero is equivalent to removing the sample, however unittests tell us
        that this is always the case only for `method='SLSQP'` (our default)

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.
    quantile : float, default=0.5
        The line output by the model will have a share of approximately `quantile` data points under it. It  should
        be a value between 0 and 1.

        A value of `quantile=1` outputs a line that is above each data point, for example.
        `quantile=0.5` corresponds to LADRegression.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.linear_model import QuantileRegression

    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4])

    model = QuantileRegression().fit(X, y)
    model.coef_
    # array([1., 2., 3., 4.])

    y = X @ np.array([-1, 2, -3, 4])
    model = QuantileRegression(quantile=0.8).fit(X, y)
    model.coef_
    # array([-1.,  2., -3.,  4.])
    ```
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
        quantile=0.5,
    ):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive, method)
        self.quantile = quantile

    def _get_objective(self, X, y, sample_weight):
        def quantile_loss(params):
            return np.average(
                np.where(X @ params < y, self.quantile, 1 - self.quantile) * np.abs(y - X @ params),
                weights=sample_weight,
            ) + self._regularized_loss(params)

        def grad_quantile_loss(params):
            return (
                -(sample_weight * np.where(X @ params < y, self.quantile, 1 - self.quantile) * np.sign(y - X @ params))
                @ X
                / sample_weight.sum()
            ) + self._regularized_grad_loss(params)

        return quantile_loss, grad_quantile_loss

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator on training data `X` and `y` by minimizing the quantile loss function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,) | None, default=None
            Individual weights for each sample.

        Returns
        -------
        self : QuantileRegression
            The fitted estimator.

        Raises
        ------
        ValueError
            If `quantile` is not between 0 and 1.
        """
        if 0 <= self.quantile <= 1:
            super().fit(X, y, sample_weight)
        else:
            raise ValueError("Parameter `quantile` should be between zero and one.")

        return self


class LADRegression(QuantileRegression):
    r"""Least absolute deviation Regression.

    `LADRegression` fits a linear model to minimize the residual sum of absolute deviations between the observed targets
    in the dataset, and the targets predicted by the linear approximation, i.e.

    $$\frac{1}{N}\|y - Xw \|_1 + \alpha \cdot l_1 \cdot\|w\|_1 + \frac{\alpha}{2} \cdot (1-l_1)\cdot \|w\|^2_2$$

    Compared to linear regression, this approach is robust to outliers. You can even optimize for the lowest MAPE
    (Mean Average Percentage Error), by providing `sample_weight=np.abs(1/y_train)` when fitting the regressor.

    !!! info
        This implementation uses
        [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

    !!! warning
        If, while fitting the model, `sample_weight` contains any zero values, some solvers may not converge properly.
        We would expect that a sample weight of zero is equivalent to removing the sample, however unittests tell us
        that this is always the case only for `method='SLSQP'` (our default)

    Parameters
    ----------
    alpha : float, default=0.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with `0 <= l1_ratio <= 1`:

        - `l1_ratio = 0` is equivalent to an L2 penalty.
        - `l1_ratio = 1` is equivalent to an L1 penalty.
        - `0 < l1_ratio < 1` is the combination of L1 and L2.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
    copy_X : bool, default=True
        If True, `X` will be copied; else, it may be overwritten.
    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    method : Literal["SLSQP", "TNC", "L-BFGS-B"], default="SLSQP"
        Type of solver to use for optimization.
    quantile : float, default=0.5
        The line output by the model will have a share of approximately `quantile` data points under it. It  should
        be a value between 0 and 1.

        A value of `quantile=1` outputs a line that is above each data point, for example.
        `quantile=0.5` corresponds to LADRegression.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients of the model.
    intercept_ : float
        Independent term in the linear model. Set to 0.0 if `fit_intercept = False`.
    n_features_in_ : int
        Number of features seen during `fit`.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.linear_model import LADRegression

    np.random.seed(0)
    X = np.random.randn(100, 4)
    y = X @ np.array([1, 2, 3, 4])

    model = LADRegression().fit(X, y)
    model.coef_
    # array([1., 2., 3., 4.])

    y = X @ np.array([-1, 2, -3, 4])
    model = LADRegression(positive=True).fit(X, y)
    model.coef_
    # array([7.39575926e-18, 1.42423304e+00, 2.80467827e-17, 4.29789588e+00])
    ```
    """

    def __init__(
        self,
        alpha=0.0,
        l1_ratio=0.0,
        fit_intercept=True,
        copy_X=True,
        positive=False,
        method="SLSQP",
    ):
        super().__init__(alpha, l1_ratio, fit_intercept, copy_X, positive, method, quantile=0.5)
