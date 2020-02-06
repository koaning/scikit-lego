import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


# noinspection PyPep8Naming,PyAttributeOutsideInit
class BayesianKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian Classifier that uses Kernel Density Estimations to generate the joint distribution
    Parameters:
        - bandwidth: float
        - kernel: for scikit learn KernelDensity
    """
    def __init__(self, bandwidth=0.2, kernel='gaussian'):
        # TODO (2/6/2020) add all other sklearn kwargs
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Checks
        X, y = check_X_y(X, y, estimator=self, dtype=FLOAT_DTYPES)
        X = np.expand_dims(X, 1) if X.ndim == 1 else X  # type: np.ndarray

        # Training
        self.classes_ = unique_labels(y)
        self.models_, self.priors_logp_ = {}, {}
        for target_label in self.classes_:
            selector = y == target_label
            x_subset, y_subset = X[selector], y[selector]

            # Joint distribution
            self.models_[target_label] = KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
            ).fit(x_subset)

            # Target class prior
            self.priors_logp_[target_label] = np.log(len(x_subset) / len(X))

        return self

    def predict_proba(self, X):
        # Checks
        self._check_is_fitted()
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        # Inference
        log_prior = np.array(
            [self.priors_logp_[target_label] for target_label in self.classes_]
        )
        log_likelihood = np.array(
            [self.models_[target_label].score_samples(X) for target_label in self.classes_]
        ).T

        log_likelihood_and_prior = np.exp(log_likelihood + log_prior)
        posterior = log_likelihood_and_prior / log_likelihood_and_prior.sum(axis=1, keepdims=True)
        return posterior

    def predict(self, X):
        # Checks
        self._check_is_fitted()
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)

        # Inference
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

    def _check_is_fitted(self):
        check_is_fitted(self, ['classes_', 'models_', 'priors_logp_'])


if __name__ == '__main__':
    from sklearn.datasets import fetch_species_distributions

    payload = dict(fetch_species_distributions())

    # Create training data
    is_bradypus_label = np.array([int(row[0].decode('ascii').startswith('brady')) for row in payload['train']])
    long_lats = np.vstack((payload['train']['dd long'], payload['train']['dd lat'])).T

    species_names = ['Microryzomys Minutus', 'Bradypus Variegatus']

    is_bradypus_label_test = np.array([int(row[0].decode('ascii').startswith('brady')) for row in payload['test']])
    long_lats_test = np.vstack((payload['test']['dd long'], payload['test']['dd lat'])).T

    # Modeling both species distributions
    spp_model = BayesianKernelDensityClassifier().fit(long_lats, is_bradypus_label)

    print(f'Training Error: {spp_model.score(long_lats, is_bradypus_label):2%}')
    print(f'Testing Error : {spp_model.score(long_lats_test, is_bradypus_label_test):.2%}')