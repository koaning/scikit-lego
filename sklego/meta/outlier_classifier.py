import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklego.base import OutlierModel
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y
)


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

    def fit(self, X, y):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self)
        if not self._is_outlier_model():
            raise ValueError("Passed model does not detect outliers!")
        self.estimator_ = self.model.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['estimator_', 'classes_'])
        preds = self.estimator_.predict(X)
        result = np.zeros(preds.shape)
        result[preds == -1] = 1
        return result
