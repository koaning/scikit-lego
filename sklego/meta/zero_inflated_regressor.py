import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor, is_classifier
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.exceptions import NotFittedError


class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """
    A meta regressor for zero-inflated datasets, i.e. the targets contain a lot of zeroes.

    `ZeroInflatedRegressor` consists of a classifier and a regressor.

        - The classifier's task is to find of if the target is zero or not.
        - The regressor's task is to output a (usually positive) prediction whenever the classifier indicates that the there should be a non-zero prediction.

    The regressor is only trained on examples where the target is non-zero, which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be zero. If yes, output zero.
    Otherwise, ask the regressor for its prediction and output it.

    Parameters
    ----------
    classifier : Any, scikit-learn classifier
        A classifier that answers the question "Should the output be zero?".

    regressor : Any, scikit-learn regressor
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is non-zero.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    >>> np.random.seed(0)
    >>> X = np.random.randn(10000, 4)
    >>> y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2)
    >>> z = ZeroInflatedRegressor(
    ... classifier=ExtraTreesClassifier(random_state=0),
    ... regressor=ExtraTreesRegressor(random_state=0)
    ... )
    >>> z.fit(X, y)
    ZeroInflatedRegressor(classifier=ExtraTreesClassifier(random_state=0),
                          regressor=ExtraTreesRegressor(random_state=0))
    >>> z.predict(X)[:5]
    array([4.91483294, 0.        , 0.        , 0.04941909, 0.        ])
    """

    def __init__(self, classifier, regressor) -> None:
        """Initialize."""
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training data.

        y : np.ndarray, 1-dimensional
            The target values.

        sample_weight : Optional[np.array], default=None
            Individual weights for each sample.

        Returns
        -------
        ZeroInflatedRegressor
            Fitted regressor.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier or `regressor` is not a regressor.
        """
        X, y = check_X_y(X, y)
        self._check_n_features(X, reset=True)
        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead.")
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")

        try:
            check_is_fitted(self.classifier)
            self.classifier_ = self.classifier
        except NotFittedError:
            self.classifier_ = clone(self.classifier)
            self.classifier_.fit(X, y != 0, sample_weight=sample_weight)

        non_zero_indices = np.where(self.classifier_.predict(X) == 1)[0]

        if non_zero_indices.size > 0:
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                self.regressor_ = clone(self.regressor)
                self.regressor_.fit(
                    X[non_zero_indices],
                    y[non_zero_indices],
                    sample_weight=sample_weight[non_zero_indices] if sample_weight is not None else None
                )
        else:
            raise ValueError(
                "The predicted training labels are all zero, making the regressor obsolete. Change the classifier or use a plain regressor instead.")

        return self

    def predict(self, X):
        """
        Get predictions.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Samples to get predictions of.

        Returns
        -------
        y : np.ndarray, shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)

        output = np.zeros(len(X))
        non_zero_indices = np.where(self.classifier_.predict(X))[0]

        if non_zero_indices.size > 0:
            output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output
