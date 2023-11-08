try:
    import umap
except ImportError:
    from sklego.notinstalled import NotInstalledPackage

    umap = NotInstalledPackage("umap-learn")

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted


class UMAPOutlierDetection(BaseEstimator, OutlierMixin):
    """`UMAPOutlierDetection` is an outlier detector based on the reconstruction error from UMAP.

    If the difference between original and reconstructed data is larger than the `threshold`, the point is
        considered an outlier.

    Parameters
    ----------
    n_components : int, default=2
        Number of components of the UMAP model.
    threshold : float | None, default=None
        The threshold used for the decision function.
    variant : Literal["relative", "absolute"], default="relative"
        The variant used for the difference calculation. "relative" means that the difference between original and
        reconstructed data is divided by the sum of the original data.
    n_neighbors : int, default=15
        `n_neighbors` parameter of UMAP model.
    min_dist : float, default=0.1
        `min_dist` parameter of UMAP model.
    metric : str, default="euclidean"
        `metric` parameter of UMAP model
        (see [UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parameters.html#metric) for the full list
        of available metrics and more information).
    random_state : int | None, default=None
        `random_state` parameter of UMAP model.

    Attributes
    ----------
    umap_ : UMAP
        The underlying UMAP model.
    offset_ : float
        The offset used for the decision function.
    """

    def __init__(
        self,
        n_components=2,
        threshold=None,
        variant="relative",
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=None,
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.variant = variant
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the `UMAPOutlierDetection` model using `X` as training data by fitting the underlying UMAP model, and
        checking the `threshold` value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,) or None, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : UMAPOutlierDetection
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `n_components` is less than 2.
            - If `threshold` is `None`.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        if self.n_components < 2:
            raise ValueError("Number of components must be at least two.")
        if not self.threshold:
            raise ValueError("The `threshold` value cannot be `None`.")

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
        """Transform the data using the underlying UMAP method."""
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["umap_", "offset_"])
        return self.umap_.transform(X)

    def difference(self, X):
        """Return the calculated difference between original and reconstructed data. Row by row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            Data to calculate the difference for.

        Returns
        -------
        array-like of shape (n_samples,)
            The calculated difference.
        """
        check_is_fitted(self, ["umap_", "offset_"])
        reduced = self.umap_.transform(X)
        diff = np.sum(np.abs(self.umap_.inverse_transform(reduced) - X), axis=1)
        if self.variant == "relative":
            diff = diff / X.sum(axis=1)
        return diff

    def predict(self, X):
        """Predict if a point is an outlier using fitted estimator.

        If the difference between original and reconstructed data is larger than the `threshold`, the point is
        considered an outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted data. 1 for inliers, -1 for outliers.
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        check_is_fitted(self, ["umap_", "offset_"])
        result = np.ones(X.shape[0])
        result[self.difference(X) > self.threshold] = -1
        return result.astype(int)
