import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class PCAOutlierDetection(BaseEstimator, OutlierMixin):
    def __init__(
        self,
        n_components=None,
        threshold=None,
        variant="relative",
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.whiten = whiten
        self.variant = variant
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if not self.threshold:
            raise ValueError(f"The `threshold` value cannot be `None`.")

        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )
        self.pca_.fit(X, y)
        self.offset_ = -self.threshold
        return self

    def difference(self, X):
        check_is_fitted(self, ["pca_", "offset_"])
        reduced = self.pca_.transform(X)
        diff = np.sum(np.abs(self.pca_.inverse_transform(reduced) - X), axis=1)
        if self.variant == "relative":
            diff = diff / X.sum(axis=1)
        return diff

    def decision_function(self, X):
        return self.threshold - self.difference(X)

    def score_samples(self, X):
        return -self.difference(X)

    def predict(self, X):
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["pca_", "offset_"])
        result = np.ones(X.shape[0])
        result[self.difference(X) > self.threshold] = -1
        return result.astype(np.int)
