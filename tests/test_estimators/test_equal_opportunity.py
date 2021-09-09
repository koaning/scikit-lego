import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklego.common import flatten
from sklego.linear_model import EqualOpportunityClassifier
from sklego.metrics import equal_opportunity_score
from tests.conftest import general_checks, classifier_checks, select_tests, nonmeta_checks


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, classifier_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series"
        ]
    )
)
@pytest.mark.cvxpy
def test_standard_checks(test_fn):
    trf = EqualOpportunityClassifier(
        covariance_threshold=None,
        positive_target=True,
        C=1,
        penalty="none",
        sensitive_cols=[0],
        train_sensitive_cols=True,
    )
    test_fn(EqualOpportunityClassifier.__name__, trf)


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


@pytest.mark.cvxpy
def test_same_logistic(random_xy_dataset_clf):
    """
    Tests whether the fair classifier performs similar to logistic regression
    for binary classification problems when we set a high threshold in the case where we set
    the covariance_threshold to None
    """
    _test_same(random_xy_dataset_clf)


@pytest.mark.cvxpy
def test_same_logistic_multiclass(random_xy_dataset_multiclf):
    """
    Tests whether the fair classifier performs similar to logistic regression
    for multiclass problems when we set a high threshold in the case where we set
    the covariance_threshold to None
    """
    _test_same(random_xy_dataset_multiclf)


@pytest.mark.cvxpy
def test_regularization(sensitive_classification_dataset):
    """Tests whether increasing regularization decreases the norm of the coefficient vector"""
    X, y = sensitive_classification_dataset

    prev_theta_norm = np.inf
    for C in [1, 0.5, 0.2, 0.1]:
        fair = EqualOpportunityClassifier(
            covariance_threshold=None, sensitive_cols=["x1"], C=C, positive_target=True
        ).fit(X, y)
        theta_norm = np.abs(np.sum(fair.coef_))
        assert theta_norm < prev_theta_norm
        prev_theta_norm = theta_norm


@pytest.mark.cvxpy
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
