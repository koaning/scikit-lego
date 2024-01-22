import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted, check_X_y


class OrdinalClassifier(MultiOutputMixin, ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    r"""The `OrdinalClassifier` allows to use a binary classifier to address an ordinal classification problem.

    Suppose we have N ordinal classes to predict, then the original binary classifier is fitted on N-1 by training sets,
    each of which represents the samples where `y <= y_label` for each `y_label` in `y` except `y.max()` (as every
    sample is smaller than the maximum value).

    The binary classifiers are then used to predict the probability of each sample to be in each _new_ class
    `y <= y_label`, and finally the probability of each sample is the difference between two consecutive classes is
    computed:

    $$ P(y = \text{class}_i) = P(\text{class}_{i-1} < y \leq \text{class}_i) = P(y \leq \text{class}_i) - P(y \leq \text{class}_{i-1}) $$

    !!! warning "About scikit-learn `predict_proba`s"

        As you can see from the formula above, it is of utmost importance to use _proper_ probabilities to compute the
        results. However, not every scikit-learn classifier `.predict_proba()` method outputs probabilities (they are
        more like a confidence score).

        We recommend to use `CalibratedClassifierCV` to calibrate the probabilities of the binary classifiers.
        This is enabled by default, but can be disabled by setting `use_calibration=False` and passing a calibrated
        classifier to the `OrdinalClassifier` constructor.

        More on this topic can be found in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/calibration.html).

    !!! warning "Computation time"

        The `OrdinalClassifier` is a meta-estimator that fits N-1 binary classifiers. This can be computationally
        expensive, especially when using a large number of samples and/or features or a complex classifier.

    Parameters
    ----------
    estimator : scikit-learn compatible classifier
        The estimator to be applied to the data, used as binary classifier.
    n_jobs : int, default=None
        The number of jobs to run in parallel. `None` means 1, `-1` means using all processors.
    use_calibration : bool, default=True
        Whether or not to calibrate the binary classifiers using `CalibratedClassifierCV`.

    Attributes
    ----------
    estimators_ : dict[int, scikit-learn compatible classifier]
        The fitted underlying binary classifiers.
    classes_ : np.ndarray of shape (n_classes,)
        The classes seen during `fit`.
    n_features_in_ : int
        The number of features seen during `fit`.

    Examples
    --------
    ```py
    from sklego.meta import OrdinalClassifier
    from sklearn.linear_model import LogisticRegression

    ...

    clf = OrdinalClassifier(LogisticRegression())
    _ = clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    ```

    Notes
    -----
    The implementation is based on the paper [A simple approach to ordinal classification](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf)
    by Eibe Frank and Mark Hall.

    """

    def __init__(self, estimator, *, n_jobs=None, use_calibration=True):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.use_calibration = use_calibration

    def fit(self, X, y):
        """Fit the `OrdinalClassifier` model on training data `X` and `y` by fitting its underlying estimators on
        N-1 datasets `X` and `y` for each class `y_label` in `y` except `y.max()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : OrdinalClassifier
            Fitted model.

        Raises
        ------
        ValueError
            If the estimator is not a classifier or if it does not implement `.predict_proba()`.
        """

        if not is_classifier(self.estimator):
            raise ValueError("The estimator must be a classifier.")

        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("The estimator must implement `.predict_proba()` method.")

        X, y = check_X_y(X, y, estimator=self)

        self.classes_ = np.sort(np.unique(y))
        self.n_features_in_ = X.shape[1]

        if self.n_jobs is None or self.n_jobs == 1:
            self.estimators_ = {y_label: self._fit_binary_estimator(X, y, y_label) for y_label in self.classes_[:-1]}
        else:
            self.estimators_ = dict(
                zip(
                    self.classes_[:-1],
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(self._fit_binary_estimator)(X, y, y_label) for y_label in self.classes_[:-1]
                    ),
                )
            )
        return self

    def predict_proba(self, X):
        """Predict class probabilities for samples in `X`. The class probabilities of a sample are computed as the
        difference between the probability of the sample to be in class `y_label` and the probability of the sample to
        be in class `y_label - 1`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The predicted class probabilities.

        Raises
        ------
        ValueError
            If `X` has a different number of features than the one seen during `fit`.
        """
        check_is_fitted(self, ["estimators_", "classes_"])

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_} features.")

        raw_proba = np.array([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_.values()]).T
        p_y_le = np.column_stack((np.zeros(X.shape[0]), raw_proba, np.ones(X.shape[0])))

        # Equivalent to (p_y_le[:, 1:] - p_y_le[:, :-1])
        return np.diff(p_y_le, n=1, axis=1)

    def predict(self, X):
        """Predict class labels for samples in `X` as the class with the highest probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _fit_binary_estimator(self, X, y, y_label):
        """Utility method to fit a binary classifier on the dataset where `y <= y_label`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features )
            The training data.
        y : array-like of shape (n_samples,)
            The target values.
        y_label : int
            The label of the class to predict.

        Returns
        -------
        fitted_model : scikit-learn compatible classifier
            The fitted binary classifier.
        """
        y_bin = (y <= y_label).astype(int)
        fitted_model = clone(self.estimator).fit(X, y_bin)
        if self.use_calibration:
            return CalibratedClassifierCV(fitted_model, cv="prefit").fit(X, y_bin)
        else:
            return fitted_model
