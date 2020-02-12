import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class BayesianKernelDensityClassifier(BaseEstimator, ClassifierMixin):
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
        """
        Bayesian Classifier that uses Kernel Density Estimations to generate the joint distribution

        All parameters of the model are an exact copy of the parameters in scikit-learn.
        """
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
        """
        Fit the model using X, y as training data.

        :param X: array-like, shape=(n_features, n_samples)
        :param y: array-like, shape=(n_samples)
        :return: Returns an instance of self
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
        """
        Probability estimates.

        The returned estimates for all classes are in the same order found in the `.classes_` attribute.

        :param X: array-like of shape (n_samples, n_features)
        :return: array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in self.classes_.
        """
        check_is_fitted(self)
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        log_prior = np.array(
            [self.priors_logp_[target_label] for target_label in self.classes_]
        )

        log_likelihood = np.array(
            [
                self.models_[target_label].score_samples(X)
                for target_label in self.classes_
            ]
        ).T

        log_likelihood_and_prior = np.exp(log_likelihood + log_prior)
        evidence = log_likelihood_and_prior.sum(axis=1, keepdims=True)
        posterior = log_likelihood_and_prior / evidence
        return posterior

    def predict(self, X):
        """
        Predict class labels for samples in X.

        :param X: array_like, shape (n_samples, n_features)
        :return: array, shape (n_samples)
        """
        check_is_fitted(self)
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        return self.classes_[np.argmax(self.predict_proba(X), 1)]
