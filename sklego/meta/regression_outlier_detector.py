import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted


class RegressionOutlierDetector(BaseEstimator, OutlierMixin):
    """Morphs a regression estimator into one that can detect outliers. We will try to predict `column` in X.

    Parameters
    ----------
    model : scikit-learn compatible regression model
        A regression model that will be used for prediction.
    column : int
        The index of the target column to predict in the input data.
    lower : float, default=2.0
        Lower threshold for outlier detection. The method used for detection depends on the `method` parameter.
    upper : float, default=2.0
        Upper threshold for outlier detection. The method used for detection depends on the `method` parameter.
    method : Literal["sd", "relative", "absolute"], default="sd"
        The method to use for outlier detection.

        - `"sd"` uses standard deviation
        - `"relative"` uses relative difference
        - `"absolute"` uses absolute difference

    Attributes
    ----------
    estimator_ : scikit-learn compatible regression model
        The fitted underlying regression model.
    sd_ : float
        The standard deviation of the differences between true and predicted values.
    idx_ : int
        The index of the target column in the input data.
    """

    def __init__(self, model, column, lower=2, upper=2, method="sd"):
        self.model = model
        self.column = column
        self.lower = lower
        self.upper = upper
        self.method = method

    def _is_regression_model(self):
        """Check if the underlying model is a regression model."""
        return any(["RegressorMixin" in p.__name__ for p in type(self.model).__bases__])

    def _handle_thresholds(self, y_true, y_pred):
        """Compute if a sample is an outlier based on the `method` parameter."""
        difference = y_true - y_pred
        results = np.ones(difference.shape, dtype=int)
        allowed_methods = ["sd", "relative", "absolute"]
        if self.method not in allowed_methods:
            ValueError(f"`method` must be in {allowed_methods} got: {self.method}")
        if self.method == "sd":
            lower_limit_hit = -self.lower * self.sd_ > difference
            upper_limit_hit = self.upper * self.sd_ < difference
        if self.method == "relative":
            lower_limit_hit = -self.lower > difference / y_true
            upper_limit_hit = self.upper < difference / y_true
        if self.method == "absolute":
            lower_limit_hit = -self.lower > difference
            upper_limit_hit = self.upper < difference
        results[lower_limit_hit] = -1
        results[upper_limit_hit] = -1
        return results

    def to_x_y(self, X):
        """Split `X` into two arrays `X_to_use` and `y`.
        `y` is the column we want to predict (specified in the `column` parameter) and `X_to_use` is the rest of the
        data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to split.

        Returns
        -------
        X_to_use : array-like of shape (n_samples, n_features-1)
            Data to use for prediction.
        y : array-like of shape (n_samples,)
            The target column.
        """
        y = X[:, self.idx_]
        cols_to_use = [i for i in range(X.shape[1]) if i != self.column]
        X_to_use = X[:, cols_to_use]
        if len(X_to_use.shape) == 1:
            X_to_use = X_to_use.reshape(-1, 1)
        return X_to_use, y

    def fit(self, X, y=None):
        """Fit the underlying model on `X_to_use` and `y` where:

        - `y` is the column we want to predict (`X[:, self.column]`)
        - `X_to_use` is the rest of the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        self : RegressionOutlierDetector
            The fitted estimator.

        Raises
        ------
        ValueError
            If the `model` is not a regression estimator.
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
        """Predict which samples of `X` are outliers using the underlying estimator and given `method`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.
        y : array-like of shape (n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values. 1 for inliers, -1 for outliers.
        """
        check_is_fitted(self, ["estimator_", "sd_", "idx_"])
        X = check_array(X, estimator=self)
        X, y = self.to_x_y(X)
        preds = self.estimator_.predict(X)
        return self._handle_thresholds(y, preds)

    def score_samples(self, X, y=None):
        """Calculate the outlier scores for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for which outlier scores are calculated.
        y : array-like of shape shape=(n_samples,), default=None
            Ignored, present for compatibility.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The outlier scores for the input data.

        Raises
        ------
        ValueError
            If `method` is not one of "sd", "relative", or "absolute".
        """
        check_is_fitted(self, ["estimator_", "sd_", "idx_"])
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
            return difference / y_true
        if self.method == "absolute":
            return difference
