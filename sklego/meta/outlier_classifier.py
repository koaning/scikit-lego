import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import _SigmoidCalibration
from sklego.base import OutlierModel
from sklearn.utils.validation import check_is_fitted, check_X_y


class OutlierClassifier(BaseEstimator, ClassifierMixin):
    """
    Morphs an estimator that performs outlier detection into a classifier.
    When an outlier is detected it will output 1 and 0 otherwise.
    This way you can use familiar metrics again and this allows you
    to consider outlier models as a fraud detector.
    """

    def __init__(self, model):
        self.model = model

    def _is_outlier_model(self):
        return isinstance(self.model, OutlierModel)

    def fit(self, X, y=None):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self)
        if not self._is_outlier_model():
            raise ValueError("Passed model does not detect outliers!")
        if not hasattr(self.model, 'decision_function'):
            raise ValueError(f'Passed model {self.model} does not have a `decision_function` '
                             f'method. This is required for `predict_proba` estimation.')
        self.estimator_ = self.model.fit(X, y)
        self.classes_ = np.array([0, 1])

        # fit sigmoid function for `predict_proba`
        decision_function_scores = self.estimator_.decision_function(X)
        self._predict_proba_sigmoid = _SigmoidCalibration().fit(decision_function_scores, y)

        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        preds = self.estimator_.predict(X)
        result = np.zeros(preds.shape)
        result[preds == -1] = 1
        return result

    def predict_proba(self, X):
        """
        Predict probability estimates for new data.

        :param X: array-like, shape=(n_columns, n_samples,) input data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["estimator_", "classes_"])
        decision_function_scores = self.estimator_.decision_function(X)
        probabilities = self._predict_proba_sigmoid.predict(decision_function_scores).reshape(-1, 1)
        complement = np.ones_like(probabilities) - probabilities
        return np.hstack((complement, probabilities))
