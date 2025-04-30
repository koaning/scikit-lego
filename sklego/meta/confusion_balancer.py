from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted
from sklearn_compat.utils.validation import validate_data

from sklego.base import ProbabilisticClassifier


class ConfusionBalancer(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    r"""The `ConfusionBalancer` estimator attempts to give its child estimator a more balanced output by learning from
    the confusion matrix during training.

    The idea is that the confusion matrix calculates $P(C_i | M_i)$ where $C_i$ is the actual class and $M_i$ is the
    class that the underlying model gives. We use these probabilities to attempt a more balanced prediction by averaging
    the correction from the confusion matrix with the original probabilities.

    $$P(\text{class}_j) = \alpha P(\text{model}_j) + (1-\alpha) P(\text{class}_j | \text{model}_j) P(\text{model}_j)$$

    Parameters
    ----------
    estimator : scikit-learn compatible classifier
        The estimator to be wrapped, it must implement a `predict_proba` method.
    alpha : float, default=0.5
        Hyperparameter which determines how much smoothing to apply. Must be between 0 and 1.
    cfm_smooth : float, default=0
        Smoothing parameter for the confusion matrices to ensure zeros don't exist.

    Attributes
    ----------
    classes_ : array-like of shape (n_classes,)
        The target class labels.
    cfm_ : array-like of shape (n_classes, n_classes)
        The confusion matrix used for the correction.

     Example
     -------
    ```py
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from sklego.meta import ConfusionBalancer

    np.random.seed(0)
    n1, n2 = 50, 100
    X = np.concatenate([np.random.normal(0, 1, (n1, 2)), np.random.normal(2, 1, (n2, 2))], axis=0)
    y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0).reshape(-1)

    confusion_balancer = ConfusionBalancer(
        estimator=LogisticRegression(),
        alpha=.5,
        cfm_smooth= 1
        )

    # Fit the model
    confusion_balancer.fit(X, y)

    # Predict out of sample datapoint
    dp = np.random.normal(0, 1, (1, 2))
    probs = confusion_balancer.predict_proba(dp) # Get probabilities per class
    preds = confusion_balancer.predict(dp) # Get most likely class

    # print(f'Out of sample datapoint {dp} is predicted to belong in class {int(preds[0])} (Probabilities are {probs} for classes 0 and 1, respectively)')

    ### Out of sample datapoint [[-1.30652685  1.65813068]] is predicted to belong in class 0 (Probabilities are [[0.92173898 0.07826102]] for classes 0 and 1, respectively)
    ```
    """

    _required_parameters = ["estimator"]

    def __init__(self, estimator, alpha: float = 0.5, cfm_smooth=0):
        self.estimator = estimator
        self.alpha = alpha
        self.cfm_smooth = cfm_smooth

    def fit(self, X, y):
        """Fit the underlying estimator on the training data `X` and `y`, it calculates the confusion matrix,
        normalizes it and stores it for later use.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ConfusionBalancer
            The fitted estimator.

        Raises
        ------
        ValueError
            If the underlying estimator does not have a `predict_proba` method.
        """

        X, y = validate_data(self, X=X, y=y, dtype=FLOAT_DTYPES, reset=True)

        if not isinstance(self.estimator, ProbabilisticClassifier):
            raise ValueError(
                "The ConfusionBalancer meta model only works on classification models with .predict_proba."
            )
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = unique_labels(y)
        cfm = confusion_matrix(y, self.estimator_.predict(X)).T + self.cfm_smooth
        self.cfm_ = cfm / cfm.sum(axis=1).reshape(-1, 1)
        return self

    def predict_proba(self, X):
        """Predict probabilities for new data `X` using the underlying estimator and then applying the confusion matrix
        correction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted values.
        """
        check_is_fitted(self, ["cfm_", "classes_", "estimator_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        preds = self.estimator_.predict_proba(X)
        return (1 - self.alpha) * preds + self.alpha * preds @ self.cfm_

    def predict(self, X):
        """Predict most likely class for new data `X` using the underlying estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ["cfm_", "classes_", "estimator_"])
        X = validate_data(self, X=X, dtype=FLOAT_DTYPES, reset=False)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
