import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import (
    check_is_fitted,
    check_array
)


class RegressionOutlierDetector(BaseEstimator, OutlierMixin):
    """
    Morphs a regression model into one that can detect outliers. We will try
    to predict `column` in X.
    """
    def __init__(self, model, column, lower=2, upper=2, method='sd'):
        self.model = model
        self.column = column
        self.lower = lower
        self.upper = upper
        self.method = method

    def _is_regression_model(self):
        return any(
            ["RegressorMixin" in p.__name__ for p in type(self.model).__bases__]
        )

    def _handle_thresholds(self, y_true, y_pred):
        difference = y_true - y_pred
        results = np.ones(difference.shape, dtype=int)
        allowed_methods = ["sd", "relative", "absolute"]
        if self.method not in allowed_methods:
            ValueError(f"`method` must be in {allowed_methods} got: {self.method}")
        if self.method == "sd":
            lower_limit_hit = -self.lower * self.sd_ > difference
            upper_limit_hit = self.upper * self.sd_ < difference
        if self.method == "relative":
            lower_limit_hit = -self.lower > difference/y_true
            upper_limit_hit = self.upper < difference/y_true
        if self.method == "absolute":
            lower_limit_hit = -self.lower > difference
            upper_limit_hit = self.upper < difference
        results[lower_limit_hit] = -1
        results[upper_limit_hit] = -1
        return results

    def to_x_y(self, X):
        y = X[:, self.idx_]
        cols_to_use = [i for i in range(X.shape[1]) if i != self.column]
        X_to_use = X[:, cols_to_use]
        if len(X_to_use.shape) == 1:
            X_to_use = X_to_use.reshape(-1, 1)
        return X_to_use, y

    def fit(self, X, y=None):
        """
        Fit the data after adapting the same weight.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        self.idx_ = np.argmax([i == self.column for i in X.columns]) if isinstance(X, pd.DataFrame) else self.column
        X = check_array(X, estimator=self)
        if not self._is_regression_model():
            raise ValueError("Passed model must be regression!")
        X, y = self.to_x_y(X)
        self.estimator_ = self.model.fit(X, y)
        self.sd_ = np.std(self.estimator_.predict(X) - y)
        return self

    def predict(self, X, y=None):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ['estimator_', 'sd_', 'idx_'])
        X = check_array(X, estimator=self)
        X, y = self.to_x_y(X)
        preds = self.estimator_.predict(X)
        return self._handle_thresholds(y, preds)

    def score_samples(self, X, y=None):
        check_is_fitted(self, ['estimator_', 'sd_', 'idx_'])
        X = check_array(X, estimator=self)
        X, y_true = self.to_x_y(X)
        y_pred = self.estimator_.predict(X)
        difference = y_true - y_pred
        allowed_methods = ["sd", "relative", "absolute"]
        if self.method not in allowed_methods:
            ValueError(f"`method` must be in {allowed_methods} got: {self.method}")
        if self.method == "sd":
            return difference
        if self.method == "relative":
            return difference/y_true
        if self.method == "absolute":
            return difference
