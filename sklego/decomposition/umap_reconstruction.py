import numpy as np
import umap
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES


class UMAPOutlierDetection(BaseEstimator, OutlierMixin):
    """
    Does outlier detection based on the reconstruction error from UMAP.
    """
    def __init__(
        self,
        n_components=2,
        threshold=None,
        variant="relative",
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=None
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the model using X as training data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: ignored but kept in for pipeline support
        :return: Returns an instance of self.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.n_components < 2:
            raise ValueError("Number of components must be at least two.")
        if not self.threshold:
            raise ValueError(f"The `threshold` value cannot be `None`.")

        self.umap_ = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        self.umap_.fit(X, y)
        self.offset_ = -self.threshold
        return self

    def transform(self, X):
        """
        Uses the underlying UMAP method to transform the data.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["umap_", "offset_"])
        return self.umap_.transform(X)

    def difference(self, X):
        """
        Shows the calculated difference between original and reconstructed data. Row by row.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: array, shape=(n_samples,) the difference
        """
        check_is_fitted(self, ["umap_", "offset_"])
        reduced = self.umap_.transform(X)
        diff = np.sum(np.abs(self.umap_.inverse_transform(reduced) - X), axis=1)
        if self.variant == "relative":
            diff = diff / X.sum(axis=1)
        return diff

    def predict(self, X):
        """
        Predict if a point is an outlier.

        :param X: array-like, shape=(n_columns, n_samples, ) training data.
        :return: array, shape=(n_samples,) the predicted data. 1 for inliers, -1 for outliers.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["umap_", "offset_"])
        result = np.ones(X.shape[0])
        result[self.difference(X) > self.threshold] = -1
        return result.astype(np.int)
