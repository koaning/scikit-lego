import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GaussianMixtureNB(BaseEstimator, ClassifierMixin):
    """
    The GaussianMixtureNB trains a Naive Bayes Classifier that uses a mixture
    of gaussians instead of merely training a single one.

    You can pass any keyword parameter that scikit-learn's Gaussian Mixture
    Model uses and it will be passed along.
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X: np.array, y: np.array) -> "GaussianMixtureNB":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        self.num_fit_cols_ = X.shape[1]
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            self.gmms_[c] = [
                GaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    tol=self.tol,
                    reg_covar=self.reg_covar,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    init_params=self.init_params,
                    weights_init=self.weights_init,
                    means_init=self.means_init,
                    precisions_init=self.precisions_init,
                    random_state=self.random_state,
                    warm_start=self.warm_start,
                ).fit(subset_x[:, i].reshape(-1, 1), subset_y)
                for i in range(X.shape[1])
            ]
        return self

    def predict(self, X):
        check_is_fitted(self, ["gmms_", "classes_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.array):
        check_is_fitted(self, ["gmms_", "classes_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.num_fit_cols_ != X.shape[1]:
            raise ValueError(
                f"number of columns {X.shape[1]} does not match fit size {self.num_fit_cols_}"
            )
        check_is_fitted(self, ["gmms_", "classes_"])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for k, v in self.gmms_.items():
            class_idx = int(np.argwhere(self.classes_ == k))
            probs[:, class_idx] = np.array(
                [
                    m.score_samples(np.expand_dims(X[:, idx], 1))
                    for idx, m in enumerate(v)
                ]
            ).sum(axis=0)
        likelihood = np.exp(probs)
        return likelihood / likelihood.sum(axis=1).reshape(-1, 1)


class BayesianGaussianMixtureNB(BaseEstimator, ClassifierMixin):
    """
    The BayesianGaussianMixtureNB trains a Naive Bayes Classifier that uses a bayesian
    mixture of gaussians instead of merely training a single one.

    You can pass any keyword parameter that scikit-learn's Bayesian Gaussian Mixture
    Model uses and it will be passed along.
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=0.001,
        reg_covar=1e-06,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.array, y: np.array) -> "BayesianGaussianMixtureNB":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :param y: array-like, shape=(n_samples, ) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        self.num_fit_cols_ = X.shape[1]
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            self.gmms_[c] = [
                BayesianGaussianMixture(
                    n_components=self.n_components,
                    covariance_type=self.covariance_type,
                    tol=self.tol,
                    reg_covar=self.reg_covar,
                    max_iter=self.max_iter,
                    n_init=self.n_init,
                    init_params=self.init_params,
                    weight_concentration_prior_type=self.weight_concentration_prior_type,
                    weight_concentration_prior=self.weight_concentration_prior,
                    mean_precision_prior=self.mean_precision_prior,
                    mean_prior=self.mean_prior,
                    degrees_of_freedom_prior=self.degrees_of_freedom_prior,
                    covariance_prior=self.covariance_prior,
                    random_state=self.random_state,
                    warm_start=self.warm_start,
                    verbose=self.verbose,
                    verbose_interval=self.verbose_interval,
                ).fit(subset_x[:, i].reshape(-1, 1), subset_y)
                for i in range(X.shape[1])
            ]
        return self

    def predict(self, X):
        check_is_fitted(self, ["gmms_", "classes_", "num_fit_cols_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.array):
        check_is_fitted(self, ["gmms_", "classes_", "num_fit_cols_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.num_fit_cols_ != X.shape[1]:
            raise ValueError(
                f"number of columns {X.shape[1]} does not match fit size {self.num_fit_cols_}"
            )
        check_is_fitted(self, ["gmms_", "classes_"])
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for k, v in self.gmms_.items():
            class_idx = int(np.argwhere(self.classes_ == k))
            probs[:, class_idx] = np.array(
                [
                    m.score_samples(np.expand_dims(X[:, idx], 1))
                    for idx, m in enumerate(v)
                ]
            ).sum(axis=0)
        likelihood = np.exp(probs)
        return likelihood / likelihood.sum(axis=1).reshape(-1, 1)
