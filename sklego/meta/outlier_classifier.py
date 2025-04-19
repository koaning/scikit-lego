import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.base import OutlierModel


class OutlierClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Morphs an outlier detection model into a classifier.

    When an outlier is detected it will output 1 and 0 otherwise. This way you can use familiar metrics again and this
    allows you to consider outlier models as a fraud detector.

    Parameters
    ----------
    model : scikit-learn compatible outlier detection model
        An outlier detection model that will be used for prediction.

    Attributes
    ----------
    estimator_ : scikit-learn compatible outlier detection model
        The fitted underlying outlier detection model.
    classes_ : array-like of shape (2,)
        Classes used for prediction (0 or 1)

    Example
    -------
    ```py
    from sklearn.ensemble import IsolationForest
    from sklego.meta.outlier_classifier import OutlierClassifier

    X = [[0], [0.5], [-1], [99]]
    y = [0, 0, 0, 1]

    isolation_forest = IsolationForest()

    outlier_clf = OutlierClassifier(isolation_forest)
    _ = outlier_clf.fit(X, y)

    preds = outlier_clf.predict([[100], [-0.5], [0.5], [1]])
    # array[1. 0. 0. 0.]

    proba_preds = outlier_clf.predict_proba([[100], [-0.5], [0.5], [1]])
    # [[0.34946567 0.65053433]
    #  [0.79707913 0.20292087]
    #  [0.80275406 0.19724594]
    #  [0.80275406 0.19724594]]
    ```
    """

    _required_parameters = ["model"]

    def __init__(self, model):
        self.model = model

    def _is_outlier_model(self):
        """Check if the underlying model is an outlier detection model."""
        return isinstance(self.model, OutlierModel)

    def fit(self, X, y=None):
        """Fit the underlying estimator to the training data `X` and `y`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Target values.

        Returns
        -------
        self : OutlierClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            - If the underlying model is not an outlier detection model.
            - If the underlying model does not have a `decision_function` method.
        """
        if not self._is_outlier_model():
            raise ValueError("Passed model does not detect outliers!")
        if not hasattr(self.model, "decision_function"):
            raise ValueError(
                f"Passed model {self.model} does not have a `decision_function` "
                f"method. This is required for `predict_proba` estimation."
            )
        if y is not None:
            X, y = validate_data(self, X=X, y=y, reset=True)
        else:
            X = validate_data(self, X=X, reset=True)

        self.estimator_ = clone(self.model).fit(X, y)
        self.n_features_in_ = self.estimator_.n_features_in_
        self.classes_ = np.array([0, 1])

        # fit sigmoid function for `predict_proba`
        decision_function_scores = self.estimator_.decision_function(X)
        self._predict_proba_sigmoid = _SigmoidCalibration().fit(decision_function_scores, y)

        return self

    def predict(self, X):
        """Predict new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values. 0 for inliers, 1 for outliers.
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        X = validate_data(self, X=X, reset=False)
        preds = self.estimator_.predict(X)
        result = (preds == -1).astype(int)
        return result

    def predict_proba(self, X):
        """Predict probability estimates for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted probabilities.
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        X = validate_data(self, X=X, reset=False)
        decision_function_scores = self.estimator_.decision_function(X)
        probabilities = self._predict_proba_sigmoid.predict(decision_function_scores).reshape(-1, 1)
        complement = np.ones_like(probabilities) - probabilities
        return np.hstack((complement, probabilities))
