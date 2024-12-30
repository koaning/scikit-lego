import logging
from inspect import signature

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn_compat.utils.validation import validate_data


class ZeroInflatedRegressor(RegressorMixin, MetaEstimatorMixin, BaseEstimator):
    """A meta regressor for zero-inflated datasets, i.e. the targets contain a lot of zeroes.

    `ZeroInflatedRegressor` consists of a classifier and a regressor.

    - The classifier's task is to find if the target is zero or not.
    - The regressor's task is to output a (usually positive) prediction whenever the classifier indicates that
    there should be a non-zero prediction.

    The regressor is only trained on examples where the target is non-zero, which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be zero. If yes, output zero.
    Otherwise, ask the regressor for its prediction and output it.

    Parameters
    ----------
    classifier : scikit-learn compatible classifier
        A classifier that answers the question "Should the output be zero?".
    regressor : scikit-learn compatible regressor
        A regressor for predicting the target. Its prediction is only used if `classifier` says that the output is
        non-zero.
    handle_zero : Literal["error", "ignore"], default="error"
            How to behave in the case that all train set output consists of zero values only.

            - `handle_zero = 'error'`: will raise a `ValueError` (default).
            - `handle_zero = 'ignore'`: will continue to train the regressor on the entire dataset.

    Attributes
    ----------
    classifier_ : scikit-learn compatible classifier
        The fitted classifier.
    regressor_ : scikit-learn compatible regressor
        The fitted regressor.

    Examples
    --------
    ```py
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklego.meta import ZeroInflatedRegressor

    np.random.seed(0)
    X = np.random.randn(10000, 4)
    y = ((X[:, 0]>0) & (X[:, 1]>0)) * np.abs(X[:, 2] * X[:, 3]**2)

    model = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(random_state=0, max_depth=10),
        regressor=ExtraTreesRegressor(random_state=0)
    ).fit(X, y)

    model.predict(X[:5])
    # array([4.91483294, 0.        , 0.        , 0.04941909, 0.        ])

    model.score_samples(X[:5]).round(2)
    # array([3.73, 0.  , 0.11, 0.03, 0.06])
    ```
    """

    _required_parameters = ["classifier", "regressor"]

    def __init__(self, classifier, regressor, handle_zero="error") -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.handle_zero = handle_zero

    def fit(self, X, y, sample_weight=None):
        """Fit the underlying classifier and regressor using `X` and `y` as training data. The regressor is only trained
        on examples where the target is non-zero.

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
        self : ZeroInflatedRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If `classifier` is not a classifier
            If `regressor` is not a regressor
            If all train target entirely consists of zeros and `handle_zero="error"`
        """
        X, y = validate_data(self, X=X, y=y, reset=True)

        if not is_classifier(self.classifier):
            raise ValueError(
                f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead."
            )
        if not is_regressor(self.regressor):
            raise ValueError(f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead.")
        if self.handle_zero not in {"ignore", "error"}:
            raise ValueError(
                f"`handle_zero` has to be one of {'ignore', 'error'}. Received '{self.handle_zero}' instead."
            )

        sample_weight = _check_sample_weight(sample_weight, X)
        try:
            check_is_fitted(self.classifier)
            self.classifier_ = self.classifier
        except NotFittedError:
            self.classifier_ = clone(self.classifier)

            if "sample_weight" in signature(self.classifier_.fit).parameters:
                self.classifier_.fit(X, y != 0, sample_weight=sample_weight)
            else:
                logging.warning("Classifier ignores sample_weight.")
                self.classifier_.fit(X, y != 0)

        indices_for_training = np.where(y != 0)[0]  # these are the non-zero indices
        if (self.handle_zero == "ignore") & (
            indices_for_training.size == 0
        ):  # if we choose to ignore that all train set output is 0
            logging.warning("Regressor will be training on `y` consisting of zero values only.")
            indices_for_training = np.where(y == 0)[0]  # use the whole train set

        if indices_for_training.size > 0:
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                self.regressor_ = clone(self.regressor)

                if "sample_weight" in signature(self.regressor_.fit).parameters:
                    self.regressor_.fit(
                        X[indices_for_training],
                        y[indices_for_training],
                        sample_weight=sample_weight[indices_for_training] if sample_weight is not None else None,
                    )
                else:
                    logging.warning("Regressor ignores sample_weight.")
                    self.regressor_.fit(
                        X[indices_for_training],
                        y[indices_for_training],
                    )
        else:
            raise ValueError(
                """The predicted training labels are all zero, making the regressor obsolete. Change the classifier
                or use a plain regressor instead. Alternatively, you can choose to ignore that predicted labels are
                all zero by setting flag handle_zero = 'ignore'"""
            )

        return self

    def predict(self, X):
        """Predict target values for `X` using fitted estimator by first asking the classifier if the output should be
        zero. If yes, output zero. Otherwise, ask the regressor for its prediction and output it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["n_features_in_", "classifier_", "regressor_"])
        X = validate_data(self, X=X, reset=False)

        output = np.zeros(len(X))
        non_zero_indices = np.where(self.classifier_.predict(X))[0]

        if non_zero_indices.size > 0:
            output[non_zero_indices] = self.regressor_.predict(X[non_zero_indices])

        return output

    @available_if(lambda self: hasattr(self.classifier_, "predict_proba"))
    def score_samples(self, X):
        r"""Predict risk estimate of `X` as the probability of `X` to not be zero times the expected value of `X`:

        $$\text{score_sample(X)} = (1-P(X=0)) \cdot E[X]$$

        where:

        - $P(X=0)$ is calculated using the `.predict_proba()` method of the underlying classifier.
        - $E[X]$ is the regressor prediction on `X`.

        !!! info

            This method requires the underlying classifier to implement `.predict_proba()` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted risk.
        """

        check_is_fitted(self, ["n_features_in_", "classifier_", "regressor_"])
        X = validate_data(self, X=X, reset=False)

        non_zero_proba = self.classifier_.predict_proba(X)[:, 1]
        expected_impact = self.regressor_.predict(X)

        return non_zero_proba * expected_impact
