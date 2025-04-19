from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class GaussianMixtureNB(ClassifierMixin, BaseEstimator):
    """The `GaussianMixtureNB` estimator is a naive bayes classifier that uses a mixture of gaussians instead of
    merely a single one. In particular it trains a `GaussianMixture` model for each class in the target and for each
    feature in the data, on the subset of `X` where `y == class`.

    You can pass any keyword parameter that scikit-learn's
    [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    model uses and it will be passed along to each of the models.

    Attributes
    ----------
    gmms_ : dict[int, List[GaussianMixture]]
        A dictionary of Gaussian Mixture Models, one for each class.
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    num_fit_cols_ : int
        Deprecated, please use `n_features_in_` instead.
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

    def fit(self, X, y) -> "GaussianMixtureNB":
        """Fit the `GaussianMixtureNB` estimator using `X` and `y` as training data by fitting a `GaussianMixture` model
        for each class in the target and for each feature in the data, on the subset of `X` where `y == class`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : GaussianMixtureNB
            The fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
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
        self.n_iter_ = sum(sum(gmm.n_iter_ for gmm in gmm_c) for gmm_c in self.gmms_.values())
        return self

    def predict(self, X):
        """Predict labels for `X` using fitted estimator and `predict_proba` method, by picking the class with the
        highest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["gmms_", "classes_", "n_features_in_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        # if self.n_features_in_ != X.shape[1]:
        #     raise ValueError(f"number of columns {X.shape[1]} does not match fit size {self.n_features_in_}")

        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.ndarray):
        """Predict probabilities for `X` using fitted estimator by summing the probabilities of each gaussian for each
        class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, ["gmms_", "classes_", "n_features_in_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        probs = np.zeros((X.shape[0], len(self.classes_)))
        for k, v in self.gmms_.items():
            class_idx = np.argmax(self.classes_ == k)
            probs[:, class_idx] = np.array(
                [m.score_samples(np.expand_dims(X[:, idx], 1)) for idx, m in enumerate(v)]
            ).sum(axis=0)
        likelihood = np.exp(probs)
        return likelihood / likelihood.sum(axis=1).reshape(-1, 1)

    @property
    def num_fit_cols_(self):
        warn(
            "Please use `n_features_in_` instead of `num_fit_cols_`,"
            "`num_fit_cols_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_


class BayesianGaussianMixtureNB(ClassifierMixin, BaseEstimator):
    """The `BayesianGaussianMixtureNB` estimator is a naive bayes classifier that uses a bayesian mixture of gaussians
    instead of merely a single one. In particular it trains a `BayesianGaussianMixture` model for each class in the
    target and for each feature in the data, on the subset of `X` where `y == class`.

    You can pass any keyword parameter that scikit-learn's
    [`BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)
    model uses and it will be passed along to each of the models.

    Attributes
    ----------
    gmms_ : dict[int, List[BayesianGaussianMixture]]
        A dictionary of Gaussian Mixture Models, one for each class.
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.
    num_fit_cols_ : int
        Deprecated, please use `n_features_in_` instead.
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

    def fit(self, X, y) -> "BayesianGaussianMixtureNB":
        """Fit the `BayesianGaussianMixtureNB` estimator using `X` and `y` as training data by fitting a
        `BayesianGaussianMixture` model for each class in the target and for each feature in the data, on the subset of
        `X` where `y == class`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : BayesianGaussianMixtureNB
            The fitted estimator.
        """
        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        self.gmms_ = {}
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
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
        self.n_iter_ = sum(sum(gmm.n_iter_ for gmm in gmm_c) for gmm_c in self.gmms_.values())
        return self

    def predict(self, X):
        """Predict labels for `X` using fitted estimator and `predict_proba` method, by picking the class with the
        highest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data.
        """
        check_is_fitted(self, ["gmms_", "classes_", "n_features_in_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X: np.ndarray):
        """Predict probabilities for `X` using fitted estimator by summing the probabilities of each gaussian for each
        class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, ["gmms_", "classes_", "n_features_in_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        probs = np.zeros((X.shape[0], len(self.classes_)))
        for k, v in self.gmms_.items():
            class_idx = np.argmax(self.classes_ == k)
            probs[:, class_idx] = np.array(
                [m.score_samples(np.expand_dims(X[:, idx], 1)) for idx, m in enumerate(v)]
            ).sum(axis=0)
        likelihood = np.exp(probs)
        return likelihood / likelihood.sum(axis=1).reshape(-1, 1)

    @property
    def num_fit_cols_(self):
        warn(
            "Please use `n_features_in_` instead of `num_fit_cols_`,"
            "`num_fit_cols_` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self.n_features_in_
