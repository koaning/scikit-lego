import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted, check_X_y


class SubjectiveClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """Corrects predictions of the inner classifier by taking into account a (subjective) prior distribution of the
    classes.

    This can be useful when there is a difference in class distribution between the training data set and the real
    world. Using the confusion matrix of the inner classifier and the prior, the posterior probability for a class,
    given the prediction of the inner classifier, can be computed.

    The background for this posterior estimation is given in
    [this article](https://lucdemortier.github.io/articles/16/PerformanceMetrics).

    Based on the `evidence` attribute, this meta estimator's predictions are based on simple weighing of the inner
    estimator's `predict_proba()` results, the posterior probabilities based on the confusion matrix, or a combination
    of the two approaches.

    Parameters
    ----------
    estimator : scikit-learn compatible classifier
        Classifier that will be wrapped with SubjectiveClassifier. It should implement `predict_proba` method.
    prior : dict[int, float]
        A dictionary mapping `class -> frequency` representing the prior (a.k.a. subjective real-world) class
        distribution. The class frequencies should sum to 1.
    evidence : Literal["predict_proba", "confusion_matrix", "both"], default="both"
        A string indicating which evidence should be used to correct the inner estimator's predictions.

        - If `"both"` the the inner estimator's `predict_proba()` results are multiplied by the posterior probabilities.
        - If `"predict_proba"`, the inner estimator's `predict_proba()` results are multiplied by the prior
            distribution.
        - If `"confusion_matrix"`, the inner estimator's discrete predictions are converted to posterior probabilities
            using the prior and the inner estimator's confusion matrix (obtained from the train data used in `fit()`).

    Attributes
    ----------
    estimator_ : scikit-learn compatible classifier
        The fitted classifier.
    classes_ : array-like, shape=(n_classes,)
        The classes labels.
    posterior_matrix_ : array-like, shape=(n_classes, n_classes)
        The posterior probabilities for each class, given the prediction of the inner classifier.
    """

    def __init__(self, estimator, prior, evidence="both"):
        self.estimator = estimator
        self.prior = prior
        self.evidence = evidence

    def _likelihood(self, predicted_class, given_class, cfm):
        return cfm[given_class, predicted_class] / cfm[given_class, :].sum()

    def _evidence(self, predicted_class, cfm):
        return sum(
            [
                self._likelihood(predicted_class, given_class, cfm) * self.prior[self.classes_[given_class]]
                for given_class in range(cfm.shape[0])
            ]
        )

    def _posterior(self, y, y_hat, cfm):
        y_hat_evidence = self._evidence(y_hat, cfm)
        return (
            (self._likelihood(y_hat, y, cfm) * self.prior[self.classes_[y]] / y_hat_evidence)
            if y_hat_evidence > 0
            else self.prior[y]  # in case confusion matrix has all-zero column for y_hat
        )

    def fit(self, X, y):
        """Fit the inner classfier using `X` and `y` as training data by fitting the underlying estimator and computing
        the posterior probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : SubjectiveClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `estimator` is not a classifier.
            - If `y` contains classes that are not specified in the `prior`
            - If `prior` is not a valid probability distribution (i.e. does not sum to 1).
            - If `evidence` is not one of "predict_proba", "confusion_matrix", or "both".
        """
        if not isinstance(self.estimator, ClassifierMixin):
            raise ValueError(
                "Invalid inner estimator: the SubjectiveClassifier meta model only works on classification models"
            )

        if not np.isclose(sum(self.prior.values()), 1):
            raise ValueError("Invalid prior: the prior probabilities of all classes should sum to 1")

        valid_evidence_types = ["predict_proba", "confusion_matrix", "both"]
        if self.evidence not in valid_evidence_types:
            raise ValueError(f"Invalid evidence: the provided evidence should be one of {valid_evidence_types}")

        X, y = check_X_y(X, y, estimator=self.estimator, dtype=FLOAT_DTYPES)
        if set(y) - set(self.prior.keys()):
            raise ValueError(
                f"Training data is inconsistent with prior: no prior defined for classes "
                f"{set(y) - set(self.prior.keys())}"
            )
        self.estimator.fit(X, y)
        cfm = confusion_matrix(y, self.estimator.predict(X))
        self.posterior_matrix_ = np.array(
            [[self._posterior(y, y_hat, cfm) for y_hat in range(cfm.shape[0])] for y in range(cfm.shape[0])]
        )
        return self

    @staticmethod
    def _weighted_proba(weights, y_hat_probas):
        return normalize(weights * y_hat_probas, norm="l1")

    @staticmethod
    def _to_discrete(y_hat_probas):
        y_hat_discrete = np.zeros(y_hat_probas.shape)
        y_hat_discrete[np.arange(y_hat_probas.shape[0]), y_hat_probas.argmax(axis=1)] = 1
        return y_hat_discrete

    def predict_proba(self, X):
        """Predict probability distribution of the class, based on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        check_is_fitted(self, ["posterior_matrix_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        y_hats = self.estimator.predict_proba(X)  # these are ignorant of the prior

        if self.evidence == "predict_proba":
            prior_weights = np.array([self.prior[klass] for klass in self.classes_])
            return self._weighted_proba(prior_weights, y_hats)
        else:
            posterior_probas = self._to_discrete(y_hats) @ self.posterior_matrix_.T
            return self._weighted_proba(posterior_probas, y_hats) if self.evidence == "both" else posterior_probas

    def predict(self, X):
        """Predict target values for `X` using fitted estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, )
            The predicted class.
        """
        check_is_fitted(self, ["posterior_matrix_"])
        X = check_array(X, estimator=self, dtype=FLOAT_DTYPES)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def classes_(self):
        """Alias for `.classes_` attribute of the underlying estimator."""
        return self.estimator.classes_
