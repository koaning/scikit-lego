import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted
from sklearn_compat.utils.validation import validate_data


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

        You can enable this by setting `use_calibration=True` and passing an uncalibrated classifier to the
        `OrdinalClassifier` or by passing a calibrated classifier to the `OrdinalClassifier` constructor.

        More on this topic can be found in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/calibration.html).

    !!! warning "Computation time"

        The `OrdinalClassifier` is a meta-estimator that fits N-1 binary classifiers. This can be computationally
        expensive, especially when using a large number of samples and/or features or a complex classifier.

    Parameters
    ----------
    estimator : scikit-learn compatible classifier
        The estimator to be applied to the data, used as binary classifier.
    n_jobs : int, default=None
        The number of jobs to run in parallel. The same convention of [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
        holds:

        - `n_jobs = None`: interpreted as n_jobs=1.
        - `n_jobs > 0`: n_cpus=n_jobs are used.
        - `n_jobs < 0`: (n_cpus + 1 + n_jobs) are used.
    use_calibration : bool, default=False
        Whether or not to calibrate the binary classifiers using `CalibratedClassifierCV`.
    calibrarion_kwargs : dict | None, default=None
        Keyword arguments to the `CalibratedClassifierCV` class, used only if `use_calibration=True`.

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
    import pandas as pd

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from sklego.meta import OrdinalClassifier

    url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
    df = pd.read_stata(url).assign(apply_codes = lambda t: t["apply"].cat.codes)

    target = "apply_codes"
    features = [c for c in df.columns if c not in {target, "apply"}]

    X, y = df[features].to_numpy(), df[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = OrdinalClassifier(LogisticRegression(), n_jobs=-1)
    _ = clf.fit(X_train, y_train)
    clf.predict_proba(X_test)
    ```

    Notes
    -----
    The implementation is based on the paper [A simple approach to ordinal classification](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf)
    by Eibe Frank and Mark Hall.

    """

    is_multiclass = True

    def __init__(self, estimator, *, n_jobs=None, use_calibration=False, calibration_kwargs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.use_calibration = use_calibration
        self.calibration_kwargs = calibration_kwargs

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

        X, y = validate_data(self, X=X, y=y, ensure_min_samples=2, ensure_2d=True, reset=True)
        self.classes_ = np.sort(np.unique(y))

        if self.n_classes_ < 3:
            raise ValueError("`OrdinalClassifier` can't train when less than 3 classes are present.")

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
        X = validate_data(self, X=X, ensure_2d=True, reset=False)

        raw_proba = np.array([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_.values()]).T
        p_y_le = np.column_stack((np.zeros(X.shape[0]), raw_proba, np.ones(X.shape[0])))

        # Equivalent to (p_y_le[:, 1:] - p_y_le[:, :-1])
        return np.diff(p_y_le, n=1, axis=1)

    def predict(self, X):
        """Predict class labels for samples in `X` as the class with the highest probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, ["estimators_", "classes_"])
        X = validate_data(self, X=X, ensure_2d=True, reset=False)
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
        if self.use_calibration:
            return CalibratedClassifierCV(estimator=clone(self.estimator), **(self.calibration_kwargs or {})).fit(
                X, y_bin
            )
        else:
            return clone(self.estimator).fit(X, y_bin)

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)
