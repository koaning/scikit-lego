import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class BayesianKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    """The `BayesianKernelDensityClassifier` estimator trains using Kernel Density estimations to generate the joint
    distribution.

    You can pass any keyword parameter that scikit-learn's
    [KernelDensity](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html)
    model uses and it will be passed along.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.
    models_ : dict[int, KernelDensity]
        The models for each class seen during `fit`.
    priors_logp_ : dict
        The log priors for each class seen during `fit` (estimated as `np.log(len(x_subset) / len(X))`)
    """

    def __init__(
        self,
        bandwidth=0.2,
        kernel="gaussian",
        algorithm="auto",
        metric="euclidean",
        atol=0,
        rtol=0,
        breath_first=True,
        leaf_size=40,
        metric_params=None,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.algorithm = algorithm
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breath_first = breath_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the `BayesianKernelDensityClassifier` estimator using `X` and `y` as training data by fitting a
        `KernelDensity` model for each class on the subset of X where y == class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : BayesianKernelDensityClassifier
            The fitted estimator.
        """
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)

        self.classes_ = unique_labels(y)
        self.models_, self.priors_logp_ = {}, {}
        for target_label in self.classes_:
            x_subset = X[y == target_label]

            # Computing joint distribution
            self.models_[target_label] = KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
                algorithm=self.algorithm,
                metric=self.metric,
                atol=self.atol,
                rtol=self.rtol,
                breadth_first=self.breath_first,
                leaf_size=self.leaf_size,
                metric_params=self.metric_params,
            ).fit(x_subset)

            # Computing target class prior
            self.priors_logp_[target_label] = np.log(len(x_subset) / len(X))

        return self

    def predict_proba(self, X):
        """Predict probabilities for `X` using fitted estimator and the joint distribution.

        The returned estimates for all classes are in the same order found in the `.classes_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities for each class, ordered as in `self.classes_`.
        """
        check_is_fitted(self)
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        log_prior = np.array([self.priors_logp_[target_label] for target_label in self.classes_])

        log_likelihood = np.array([self.models_[target_label].score_samples(X) for target_label in self.classes_]).T

        log_likelihood_and_prior = np.exp(log_likelihood + log_prior)
        evidence = log_likelihood_and_prior.sum(axis=1, keepdims=True)
        posterior = log_likelihood_and_prior / evidence
        return posterior

    def predict(self, X):
        """Predict labels for `X` using fitted estimator and `predict_proba()` method, by taking the class with the
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
        check_is_fitted(self)
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        return self.classes_[np.argmax(self.predict_proba(X), 1)]
