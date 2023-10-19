from deprecated import deprecated
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from sklego.common import TrainOnlyTransformerMixin


@deprecated(
    version="0.4.2",
    reason="Please use `sklego.preprocessing.OutlierRemover` instead. "
    "This object will be removed from the meta submodule in version 0.6.0.",
)
class OutlierRemover(TrainOnlyTransformerMixin, BaseEstimator):
    """Removes outliers (train-time only) using the supplied removal model.

    Parameters
    ----------
    outlier_detector : object
        An outlier detector that implements `fit` and `predict` methods.
    refit : bool, default=True
        If True, fits the estimator during pipeline.fit(). If False, the estimator is not fitted during pipeline.fit().

    Attributes
    ----------
    estimator_ : object
        The fitted outlier detector.
    """

    def __init__(self, outlier_detector, refit=True):
        self.outlier_detector = outlier_detector
        self.refit = refit
        self.estimator_ = None

    def fit(self, X, y=None):
        """Fits the estimator, and the outlier detector if `refit` is True.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self : OutlierRemover
            The fitted transformer.
        """
        self.estimator_ = clone(self.outlier_detector)
        if self.refit:
            super().fit(X, y)
            self.estimator_.fit(X, y)
        return self

    def transform_train(self, X):
        """Removes outliers from `X` using the fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data for which the outliers will be removed.

        Returns
        -------
        np.ndarray of shape (n_not_outliers, n_features)
            The data with the outliers removed, where `n_not_outliers = n_samples - n_outliers`.
        """
        check_is_fitted(self, "estimator_")
        predictions = self.estimator_.predict(X)
        check_array(predictions, estimator=self.outlier_detector, ensure_2d=False)
        return X[predictions != -1]
