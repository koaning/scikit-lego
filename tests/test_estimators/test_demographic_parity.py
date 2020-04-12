import warnings

import pytest
import numpy as np
from cvxpy import SolverError
from sklearn.linear_model import LogisticRegression

from sklego.common import flatten
from sklego.linear_model import DemographicParityClassifier, FairClassifier
from sklego.metrics import p_percent_score
from tests.conftest import general_checks, nonmeta_checks, classifier_checks


@pytest.mark.parametrize(
    "test_fn", flatten([general_checks, nonmeta_checks, classifier_checks])
)
def test_standard_checks(test_fn):
    trf = DemographicParityClassifier(
        covariance_threshold=None,
        C=1,
        penalty="none",
        sensitive_cols=[0],
        train_sensitive_cols=True,
    )
    test_fn(DemographicParityClassifier.__name__, trf)


def _test_same(dataset):
    X, y = dataset
    if X.shape[1] == 1:
        # If we only have one column (which is also the sensitive one) we can't fit
        return True

    sensitive_cols = [0]
    X_without_sens = np.delete(X, sensitive_cols, axis=1)
    lr = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        multi_class="ovr",
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    )
    fair = DemographicParityClassifier(
        covariance_threshold=None, sensitive_cols=sensitive_cols, penalty="none"
    )
    try:
        fair.fit(X, y)
    except SolverError:
        pass
    else:
        lr.fit(X_without_sens, y)
        normal_pred = lr.predict_proba(X_without_sens)
        fair_pred = fair.predict_proba(X)
        np.testing.assert_almost_equal(normal_pred, fair_pred, decimal=3)
        assert np.sum(lr.predict(X_without_sens) != fair.predict(X)) / len(X) < 0.01


def test_same_logistic(random_xy_dataset_clf):
    """
    Tests whether the fair classifier performs similar to logistic regression
    for binary classification problems when we set a high threshold in the case where we set
    the covariance_threshold to None
    """
    _test_same(random_xy_dataset_clf)


def test_same_logistic_multiclass(random_xy_dataset_multiclf):
    """
    Tests whether the fair classifier performs similar to logistic regression
    for multiclass problems when we set a high threshold in the case where we set
    the covariance_threshold to None
    """
    _test_same(random_xy_dataset_multiclf)


def test_regularization(sensitive_classification_dataset):
    """Tests whether increasing regularization decreases the norm of the coefficient vector"""
    X, y = sensitive_classification_dataset

    prev_theta_norm = np.inf
    for C in [1, 0.5, 0.2, 0.1]:
        fair = DemographicParityClassifier(
            covariance_threshold=None, sensitive_cols=["x1"], C=C
        ).fit(X, y)
        theta_norm = np.abs(np.sum(fair.coef_))
        assert theta_norm < prev_theta_norm
        prev_theta_norm = theta_norm


def test_fairness(sensitive_classification_dataset):
    """tests whether fairness (measured by p percent score) increases as we decrease the covariance threshold"""
    X, y = sensitive_classification_dataset
    scorer = p_percent_score("x1")

    prev_fairness = -np.inf
    for cov_threshold in [None, 10, 0.5, 0.1]:
        fair = DemographicParityClassifier(
            covariance_threshold=cov_threshold,
            sensitive_cols=["x1"],
            penalty="none",
            train_sensitive_cols=False,
        ).fit(X, y)
        fairness = scorer(fair, X, y)
        assert fairness >= prev_fairness
        prev_fairness = fairness


def test_deprecation():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        FairClassifier(
            covariance_threshold=1,
            sensitive_cols=["x1"],
            penalty="none",
            train_sensitive_cols=False,
        )
        assert issubclass(w[-1].category, DeprecationWarning)
