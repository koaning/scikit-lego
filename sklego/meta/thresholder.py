import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from sklego.base import ProbabilisticClassifier


class Thresholder(BaseEstimator, ClassifierMixin):
    """Takes a binary classifier and moves the threshold. This way you might design the algorithm to only accept a
    certain class if the probability for it is larger than, say, 90% instead of 50%.

    !!! info
        Please note that this only works for binary classification problems.

    Parameters
    ----------
    model : scikit-learn compatible classifier
        Classifier that will be wrapped with Thresholder. It should implement `predict_proba` method.
    threshold : float
        The threshold value to use.
    refit : bool, default=False

        - If True, we will always retrain the model even if it is already fitted.
        - If False we only refit if the original model isn't fitted.

    Attributes
    ----------
    estimator_ : scikit-learn compatible classifier
        The fitted classifier.
    classes_ : array-like, shape=(2,)
        The classes labels.
    """

    def __init__(self, model, threshold: float, refit=False):
        self.model = model
        self.threshold = threshold
        self.refit = refit

    def _handle_refit(self, X, y, sample_weight=None):
        """Only refit when we need to, unless `refit=True` is present."""
        if self.refit:
            self.estimator_ = clone(self.model)
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        else:
            try:
                _ = self.estimator_.predict(X[:1])
            except NotFittedError:
                self.estimator_.fit(X, y, sample_weight=sample_weight)

    def fit(self, X, y, sample_weight=None):
        """Fit the underlying estimator using `X` and `y` as training data. If `refit=True` we will always retrain
        (a copy of) the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples, ), default=None
            Individual weights for each sample.

        Returns
        -------
        self : Thresholder
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `model` is not a classifier or it does not implement `predict_proba` method.
            - If `model` does not have two classes.
        """
        self.estimator_ = self.model
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError("The Thresholder meta model only works on classification models with .predict_proba.")
        self._handle_refit(X, y, sample_weight)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError("The `Thresholder` meta model only works on models with two classes.")
        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator and the given `threshold`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        predicate = self.estimator_.predict_proba(X)[:, 1] > self.threshold
        return np.where(predicate, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Alias for `.predict_proba()` method of the underlying estimator."""
        check_is_fitted(self, ["classes_", "estimator_"])
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        """Alias for `.score()` method of the underlying estimator."""
        return self.estimator_.score(X, y)
