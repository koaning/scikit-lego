import numpy as np
import pytest
from sklearn.multiclass import OneVsRestClassifier

try:
    from cvxpy import SolverError
except ImportError:
    pass
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.linear_model import DemographicParityClassifier
from sklego.metrics import p_percent_score

pytestmark = pytest.mark.cvxpy


@parametrize_with_checks(
    [
        DemographicParityClassifier(
            covariance_threshold=None,
            C=1,
            sensitive_cols=[0],
            penalty=penalty,
            train_sensitive_cols=train_sensitive_cols,
        )
        for train_sensitive_cols in [True, False]
        for penalty in ["l1", "l2", None]
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # It passes all steps until the last check: assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))
        # In there large numbers "lose" their relative ranking due to numerical issues, e.g.
        # y_d = [40, 50 ], y_p = expit(y_d) = [1., 1.] => rank(y_d) = [1, 2], rank(y_p) = [1.5, 1.5]
        "check_classifier_multioutput",
        # It passes all steps until score is checked, adding `{"poor_score": True}` doesn't seem to solve or bypass
        # the test
        "check_classifiers_train",
        "check_n_features_in",  # TODO: This should be fixable?!
        "check_n_features_in_after_fitting",  # same problem as above, new check in 1.6
    }:
        pytest.skip()

    # if check.func.__name__ not in {"check_classifier_multioutput"}: pytest.skip()
    check(estimator)


def _test_same(dataset):
    X, y = dataset
    if X.shape[1] == 1:
        # If we only have one column (which is also the sensitive one) we can't fit
        return True

    sensitive_cols = [0]
    X_without_sens = np.delete(X, sensitive_cols, axis=1)
    lr = OneVsRestClassifier(
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
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
    )
    fair = DemographicParityClassifier(covariance_threshold=None, sensitive_cols=sensitive_cols, penalty="none")
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


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_regularization(sensitive_classification_dataset, penalty):
    """Tests whether increasing regularization decreases the norm of the coefficient vector"""
    X, y = sensitive_classification_dataset

    prev_theta_norm = np.inf
    for C in [1, 0.5, 0.1, 0.05]:
        fair = DemographicParityClassifier(
            covariance_threshold=None,
            sensitive_cols=["x1"],
            C=C,
            penalty=penalty,
        ).fit(X, y)
        theta_norm = np.linalg.norm(fair.estimators_[0].coef_, ord=int(penalty[-1]))
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
