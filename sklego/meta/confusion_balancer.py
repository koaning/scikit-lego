from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
)
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    check_is_fitted,
    check_X_y,
    check_array,
    FLOAT_DTYPES,
)

from sklego.base import ProbabilisticClassifier


class ConfusionBalancer(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """
    The ConfusionBalancer attempts to give it's child estimator a more balanced
    output by learning from the confusion matrix during training. The idea is that
    the confusion matrix calculates P(C_i | M_i) where C_i is the actual class and
    M_i is the class that the underlying model gives. We use these probabilities to
    attempt a more balanced prediction by averaging the correction from the confusion
    matrix with the original probabilities.

    .. math::
        p(\text{class_j}) = \alpha p(\text{model}_j) + (1-\alpha) p(\text{class_j} | \text{model}_j) p(\text{model}_j)

    :param model: a scikit learn compatible classification model that has predict_proba
    :param alpha: a hyperparameter between 0 and 1, determines how much to apply smoothing
    :param cfm_smooth: a smoothing parameter for the confusion matrices to ensure zeros don't exist
    """

    def __init__(self, estimator, alpha: float = 0.5, cfm_smooth=0):
        self.estimator = estimator
        self.alpha = alpha
        self.cfm_smooth = cfm_smooth

    def fit(self, X, y):
        """
        Fit the data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :param y: array-like, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self.estimator, dtype=FLOAT_DTYPES)
        if not isinstance(self.estimator, ProbabilisticClassifier):
            raise ValueError(
                "The ConfusionBalancer meta model only works on classifcation models with .predict_proba."
            )
        self.estimator.fit(X, y)
        self.classes_ = unique_labels(y)
        cfm = confusion_matrix(y, self.estimator.predict(X)).T + self.cfm_smooth
        self.cfm_ = cfm / cfm.sum(axis=1).reshape(-1, 1)
        return self

    def predict_proba(self, X):
        """
        Predict new data, with probabilities

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples, n_classes) the predicted data
        """
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        preds = self.estimator.predict_proba(X)
        return (1 - self.alpha) * preds + self.alpha * preds @ self.cfm_

    def predict(self, X):
        """
        Predict new data.

        :param X: array-like, shape=(n_columns, n_samples,) training data.
        :return: array, shape=(n_samples,) the predicted data
        """
        check_is_fitted(self, ["cfm_", "classes_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
