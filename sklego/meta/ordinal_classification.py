import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted


class OrdinalClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin, MultiOutputMixin):
    def __init__(self, estimator, n_jobs=None, calibrate=True):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.calibrate = calibrate

    def fit(self, X, y):
        # TODO: Add estimator checks:
        #  - estimator is a classifier
        #  - estimator is a binary classifier
        #  - estimator has a predict_proba method

        self.classes_ = np.sort(np.unique(y))
        self.n_binaries_ = len(self.classes_) - 1
        self.n_features_in_ = X.shape[1]

        if self.n_jobs is None or self.n_jobs == 1:
            self.estimators_ = {y_label: self._fit_binary_estimator(X, y, y_label) for y_label in self.classes_[:-1]}
        else:
            self.estimators_ = dict(
                zip(
                    self.classes_[:-1],
                    Parallel(n_jobs=min(self.n_binaries_, self.n_jobs))(
                        delayed(self._fit_binary_estimator)(X, y, y_label) for y_label in self.classes_[:-1]
                    ),
                )
            )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["estimators_", "classes_"])

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_} features.")

        raw_proba = np.array([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_.values()]).T
        p_y_le = np.column_stack((np.zeros(X.shape[0]), raw_proba, np.ones(X.shape[0])))

        return p_y_le[:, 1:] - p_y_le[:, :-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _fit_binary_estimator(self, X, y, y_label):
        y_bin = (y <= y_label).astype(int)
        fitted_model = clone(self.estimator).fit(X, y_bin)
        if self.calibrate:
            return CalibratedClassifierCV(fitted_model, cv="prefit").fit(X, y_bin)
        else:
            return fitted_model
