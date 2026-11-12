from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import _check_n_features, check_array

from sklego.common import TrainOnlyTransformerMixin


class OutlierRemover(TrainOnlyTransformerMixin, BaseEstimator):
    """The `OutlierRemover` transformer removes outliers (train-time only) using the supplied removal model. The
    removal model should implement `.fit()` and `.predict()` methods.

    Parameters
    ----------
    outlier_detector : scikit-learn compatible estimator
        An outlier detector that implements `.fit()` and `.predict()` methods.
    refit : bool, default=True
        Whether or not to fit the underlying estimator during `OutlierRemover(...).fit()`.
        If True, a clone of `outlier_detector` is fitted. If False, the supplied detector must already be fitted
        and is used directly without refitting.

    Attributes
    ----------
    estimator_ : object
        The fitted outlier detector.

    Examples
    --------
    ```py
    import numpy as np

    from sklearn.ensemble import IsolationForest
    from sklego.preprocessing import OutlierRemover

    np.random.seed(0)
    X = np.random.randn(10000, 2)

    isolation_forest = IsolationForest()
    isolation_forest.fit(X)
    detector_preds = isolation_forest.predict(X)

    outlier_remover = OutlierRemover(isolation_forest, refit=True)
    outlier_remover.fit(X)

    X_trans = outlier_remover.transform_train(X)
    ```
    """

    _required_parameters = ["outlier_detector"]

    def __init__(self, outlier_detector, refit=True):
        self.outlier_detector = outlier_detector
        self.refit = refit

    def fit(self, X, y=None):
        """Fit the estimator on training data `X` and `y` by fitting the underlying outlier detector if `refit` is True.

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
        super().fit(X, y)
        if self.refit:
            self.estimator_ = clone(self.outlier_detector)
            self.estimator_.fit(X, y)
        else:
            self.estimator_ = self.outlier_detector
        _check_n_features(self, X, reset=True)
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
        _check_n_features(self, X, reset=False)

        predictions = self.estimator_.predict(X)
        check_array(predictions, estimator=self.outlier_detector, ensure_2d=False)

        return X[predictions != -1]
