from warnings import warn

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class GMMOutlierDetector(OutlierMixin, BaseEstimator):
    """The `GMMDetector` trains a Gaussian Mixture model on a dataset `X`. Once a density is trained we can evaluate the
    likelihood scores to see if it is deemed likely.

    By providing a `threshold` this model might then label outliers if their likelihood score is too low.

    !!! note
        The parameters other than `threshold` and `method` are an exact copy of the parameters in
        [sklearn.mixture.GaussianMixture]( https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

    Parameters
    ----------
    threshold : float, default=0.99
        The limit at which the model thinks an outlier appears, must be between (0, 1).
    method : Literal["quantile", "stddev"], default="quantile"
        The method to use to apply the `threshold`.

        !!! info
            If you select `method="quantile"` then the threshold value represents the quantile value to start calling
            something an outlier.

            If you select `method="stddev"` then the threshold value represents the
            numbers of standard deviations before calling something an outlier.

    Attributes
    ----------
    gmm_ : GaussianMixture
        The trained Gaussian Mixture model.
    likelihood_threshold_ : float
        The threshold value used to determine if something is an outlier.

    Examples
    --------
    ```python
    import numpy as np
    from sklego.mixture import GMMOutlierDetector

    # Generate datset, it consists of two clusters
    np.random.seed(1)
    group0 = np.random.normal(0, 3, (10, 2))
    group1 = np.random.normal(2.5, 2, (5, 2))
    data = np.vstack([group0, group1])

    y = np.hstack([np.zeros((group0.shape[0],), dtype=int), np.ones((group1.shape[0],), dtype=int)])

    # Create and fit the GMMOutlierDetector model
    gmm = GMMOutlierDetector(threshold=0.9, n_components=2, random_state=1)
    gmm.fit(data, y)

    # Classify a new point as outlier or not
    p = np.array([[4.5, 0.5]])
    p_pred = gmm.predict(p) # predict the probabilities p belongs to each cluster
    print('The point is an outlier if the score is -1, inlier if the score is 1')
    ### The point is an outlier if the score is -1, inlier if the score is 1

    print(f'The score for this point is {p_pred}.')
    ### The score for this point is [-1].
    ```
    """

    _ALLOWED_METHODS = ("quantile", "stddev")

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

    def fit(self, X: np.ndarray, y=None) -> "GMMOutlierDetector":
        """Fit the `GMMOutlierDetector` model using `X`, `y` as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            Ignored, present for compatibility.

        Returns
        -------
        self : GMMOutlierDetector
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `method="quantile"` and `threshold` is not between (0, 1).
            - If `method="stddev"` and `threshold` is negative.
            - If `method` is not in `["quantile", "stddev"]`.
        """
        # GMM sometimes throws an error if you don't do this
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=True)
        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        if (self.method == "quantile") and ((self.threshold > 1) or (self.threshold < 0)):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold < 1")
        if (self.method == "stddev") and (self.threshold < 0):
            raise ValueError(f"Threshold {self.threshold} with method {self.method} needs to be 0 < threshold ")
        if self.method not in self._ALLOWED_METHODS:
            raise ValueError(f"Method not recognised. Method must be in {self._ALLOWED_METHODS}")

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
            self.likelihood_threshold_ = mean_likelihood - (self.threshold * new_likelihoods_std)

        self.n_iter_ = self.gmm_.n_iter_
        self.n_features_in_ = X.shape[1]
        self.offset_ = self.likelihood_threshold_
        return self

    def score_samples(self, X):
        """Compute the log likelihood for each sample and return the negative value."""
        check_is_fitted(self, ["gmm_", "likelihood_threshold_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        if X.ndim == 1:
            X = np.expand_dims(X, 1)

        return self.gmm_.score_samples(X)

    def decision_function(self, X):
        # We subtract self.offset_ to make 0 be the threshold value for being an outlier:
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        """Predict if a point is an outlier or not using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data. 1 for inliers, -1 for outliers.
        """
        preds = (self.decision_function(X) >= 0).astype(int)
        preds[preds == 0] = -1
        return preds

    @property
    def allowed_methods(self):
        warn(
            "Please use `_ALLOWED_METHODS` instead of `allowed_methods`,"
            "`allowed_methods` will be deprecated in future versions",
            DeprecationWarning,
        )
        return self._ALLOWED_METHODS
