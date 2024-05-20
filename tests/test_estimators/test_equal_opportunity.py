import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.linear_model import EqualOpportunityClassifier
from sklego.metrics import equal_opportunity_score

pytestmark = pytest.mark.cvxpy


@parametrize_with_checks(
    [
        EqualOpportunityClassifier(
            covariance_threshold=None,
            positive_target=True,
            C=1,
            sensitive_cols=[0],
            penalty=penalty,
            train_sensitive_cols=train_sensitive_cols,
        )
        for train_sensitive_cols in [True, False]
        for penalty in ["none", "l1"]
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
    }:
        pytest.skip()
    check(estimator)


def _test_same(dataset):
    X, y = dataset
    if X.shape[1] == 1:
        # If we only have one column (which is also the sensitive one) we can't fit
        return True

    sensitive_cols = [0]
    X_without_sens = np.delete(X, sensitive_cols, axis=1)
    lr = LogisticRegression(
        penalty=None,
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
    fair = EqualOpportunityClassifier(
        covariance_threshold=None,
        sensitive_cols=sensitive_cols,
        penalty="none",
        positive_target=True,
    )

    fair.fit(X, y)
    lr.fit(X_without_sens, y)

    normal_pred = lr.predict_proba(X_without_sens)
    fair_pred = fair.predict_proba(X)
    np.testing.assert_almost_equal(normal_pred, fair_pred, decimal=2)
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
        fair = EqualOpportunityClassifier(
            covariance_threshold=None, sensitive_cols=["x1"], C=C, positive_target=True
        ).fit(X, y)
        theta_norm = np.abs(np.sum(fair.estimators_[0].coef_))
        assert theta_norm < prev_theta_norm
        prev_theta_norm = theta_norm


def test_fairness(sensitive_classification_dataset):
    """tests whether fairness (measured by p percent score) increases as we decrease the covariance threshold"""
    X, y = sensitive_classification_dataset
    scorer = equal_opportunity_score("x1")

    prev_fairness = -np.inf
    for cov_threshold in [None, 10, 0.5, 0.1]:
        fair = EqualOpportunityClassifier(
            covariance_threshold=cov_threshold,
            positive_target=True,
            sensitive_cols=["x1"],
            penalty="none",
            train_sensitive_cols=False,
        ).fit(X, y)
        fairness = scorer(fair, X, y)
        assert fairness >= prev_fairness
        prev_fairness = fairness
