import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class PCAOutlierDetection(OutlierMixin, BaseEstimator):
    """`PCAOutlierDetection` is an outlier detector based on the reconstruction error from PCA.

    If the difference between original and reconstructed data is larger than the `threshold`, the point is
    considered an outlier.

    Parameters
    ----------
    n_components : int | None, default=None
        Number of components of the PCA model.
    threshold : float | None, default=None
        The threshold used for the decision function.
    variant : Literal["relative", "absolute"], default="relative"
        The variant used for the difference calculation. "relative" means that the difference between original and
        reconstructed data is divided by the sum of the original data.
    whiten : bool, default=False
        `whiten` parameter of PCA model.
    svd_solver : Literal["auto", "full", "arpack", "randomized"], default="auto"
        `svd_solver` parameter of PCA model.
    tol : float, default=0.0
        `tol` parameter of PCA model.
    iterated_power : int | Literal["auto"], default="auto"
        `iterated_power` parameter of PCA model.
    random_state : int | None, default=None
        `random_state` parameter of PCA model.

    Attributes
    ----------
    pca_ : PCA
        The underlying PCA model.
    offset_ : float
        The offset used for the decision function.

    Examples
    --------
    ```py
    import numpy as np
    from sklego.decomposition import PCAOutlierDetection

    X = np.array([[-1, -1, -1], [-2, -1, -2], [5, -1, 0], [1, 1, 1], [2, 1, 1], [3, 2, 3]])

    pca_model = PCAOutlierDetection(n_components=2, threshold=0.05)
    pca_model.fit(X)
    pca_pred = pca_model.predict(X)
    pca_pred
    # [ 1  1  1 -1 -1  1]
    ```
    """

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
        """Fit the `PCAOutlierDetection` model using `X` as training data by fitting the underlying PCA model, and
        checking the `threshold` value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,) or None, default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : PCAOutlierDetection
            The fitted estimator.

        Raises
        ------
        ValueError
            If `threshold` is `None`.
        """
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=True)
        if not self.threshold:
            raise ValueError("The `threshold` value cannot be `None`.")

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
        check_is_fitted(self, ["pca_", "offset_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)

        reduced = self.pca_.transform(X)
        diff = np.sum(np.abs(self.pca_.inverse_transform(reduced) - X), axis=1)
        if self.variant == "relative":
            diff = diff / X.sum(axis=1)
        return diff

    def decision_function(self, X):
        """Calculate the decision function for the data as the difference between `threshold` and the `.difference(X)`
        (which is the difference between original data and reconstructed data)."""
        return self.threshold - self.difference(X)

    def score_samples(self, X):
        """Calculate the score for the samples"""
        return -self.difference(X)

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
        check_is_fitted(self, ["pca_", "offset_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        result = np.ones(X.shape[0])
        result[self.difference(X) > self.threshold] = -1
        return result.astype(int)
