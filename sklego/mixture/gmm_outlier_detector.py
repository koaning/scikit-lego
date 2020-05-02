import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES

from scipy.stats import gaussian_kde


class GMMOutlierDetector(OutlierMixin, BaseEstimator):
    """
    The GMMDetector trains a Gaussian Mixture Model on a dataset X. Once
    a density is trained we can evaluate the likelihood scores to see if
    it is deemed likely. By giving a threshold this model might then label
    outliers if their likelihood score is too low.

    :param threshold: the limit at which the model thinks an outlier appears, must be between (0, 1)
    :param method: the method that the threshold will be applied to, possible values = [stddev, default=quantile]

    If you select method="quantile" then the threshold value represents the
    quantile value to start calling something an outlier.

    If you select method="stddev" then the threshold value represents the
    numbers of standard deviations before calling something an outlier.
    """

    def __init__(
        self,
        threshold=0.99,
        method="quantile",
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
        verbose=0,
        verbose_interval=10,
    ):
        self.threshold = threshold
        self.method = method
        self.random_state = random_state
        self.allowed_methods = ["quantile", "stddev"]
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
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def fit(self, X: np.array, y=None) -> "GMMOutlierDetector":
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: ignored but kept in for pipeline support
        :return: Returns an instance of self.
        """

        # GMM sometimes throws an error if you don't do this
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        if (self.method == "quantile") and (
            (self.threshold > 1) or (self.threshold < 0)
        ):
            raise ValueError(
                f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold < 1"
            )
        if (self.method == "stddev") and (self.threshold < 0):
            raise ValueError(
                f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold "
            )
        if self.method not in self.allowed_methods:
            raise ValueError(
                f"Method not recognised. Method must be in {self.allowed_methods}"
            )

        self.gmm_ = GaussianMixture(
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
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )
        self.gmm_.fit(X)
        score_samples = self.gmm_.score_samples(X)

        if self.method == "quantile":
            self.likelihood_threshold_ = np.quantile(score_samples, 1 - self.threshold)

        if self.method == "stddev":
            density = gaussian_kde(score_samples)
            max_x_value = minimize_scalar(lambda x: -density(x)).x
            mean_likelihood = score_samples.mean()
            new_likelihoods = score_samples[score_samples < max_x_value]
            new_likelihoods_std = np.std(new_likelihoods - mean_likelihood)
            self.likelihood_threshold_ = mean_likelihood - (
                self.threshold * new_likelihoods_std
            )

        return self

    def score_samples(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["gmm_", "likelihood_threshold_"])
        if len(X.shape) == 1:
            X = np.expand_dims(X, 1)

        return -self.gmm_.score_samples(X)

    def decision_function(self, X):
        # We subtract self.offset_ to make 0 be the threshold value for being an outlier:
        return self.score_samples(X) + self.likelihood_threshold_

    def predict(self, X):
        """
        Predict if a point is an outlier.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: array, shape=(n_samples,) the predicted data. 1 for inliers, -1 for outliers.
        """
        predictions = (self.decision_function(X) >= 0).astype(np.int)
        predictions[predictions == 1] = -1
        predictions[predictions == 0] = 1
        return predictions
