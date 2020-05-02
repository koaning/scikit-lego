import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class GMMClassifier(BaseEstimator, ClassifierMixin):
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
        verbose=0,
        verbose_interval=10,
    ):
        """
        The GMMClassifier trains a Gaussian Mixture Model for each class in y on a dataset X. Once
        a density is trained for each class we can evaluate the likelihood scores to see which class
        is more likely. All parameters of the model are an exact copy of the parameters in scikit-learn.
        """
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

    def fit(self, X: np.array, y: np.array) -> "GMMClassifier":
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
        for c in self.classes_:
            subset_x, subset_y = X[y == c], y[y == c]
            mixture = GaussianMixture(
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
            self.gmms_[c] = mixture.fit(subset_x, subset_y)
        return self

    def predict(self, X):
        check_is_fitted(self, ["gmms_", "classes_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["gmms_", "classes_"])
        res = np.zeros((X.shape[0], self.classes_.shape[0]))
        for idx, c in enumerate(self.classes_):
            res[:, idx] = self.gmms_[c].score_samples(X)
        return np.exp(res) / np.exp(res).sum(axis=1)[:, np.newaxis]
