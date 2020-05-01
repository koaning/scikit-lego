import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_is_fitted,
)


class OutlierClassifier(BaseEstimator, ClassifierMixin):
    """
    Morphs an estimator that performs outlier detection into a classifier.
    This way you can use familiar metrics again and this allows you
    to consider outlier models as a fraud detector.
    """
    def __init__(self, model):
        self.model = model

    def _is_outlier_model(self):
        return any(
            ["OutlierMixin" in p.__name__ for p in type(self.model).__bases__]
        )

    def fit(self, X, y=None):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        if not self._is_outlier_model():
            raise ValueError("Passed model does not detect outliers!")
        self.estimator_ = self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['estimator_'])
        preds = self.estimator_.predict(X)
        result = np.zeros(preds.shape)
        result[preds == -1] = 1
        return result
