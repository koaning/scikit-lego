import pytest
import numpy as np
from cvxpy import SolverError
from sklearn.linear_model import LogisticRegression

from sklego.common import flatten
from sklego.linear_model import FairClassifier
from tests.conftest import general_checks, nonmeta_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([
    general_checks,
    nonmeta_checks,
    classifier_checks,
]))
def test_standard_checks(test_fn):
    trf = FairClassifier(covariance_threshold=None, C=1, penalty='none', sensitive_cols=[0])
    test_fn(FairClassifier.__name__, trf)


def test_same_logistic(random_xy_dataset_clf):
    """Tests whether the fair classifier performs similar to logistic regression when we set a high threshold"""
    X, y = random_xy_dataset_clf

    lr = LogisticRegression(penalty='none', solver='lbfgs')
    fair = FairClassifier(covariance_threshold=None, sensitive_cols=[0], penalty='none')
    try:
        fair.fit(X, y)
    except SolverError:
        pass
    else:
        lr.fit(X, y)

        np.testing.assert_almost_equal(lr.predict_proba(X), fair.predict_proba(X), decimal=2)
        assert np.sum(lr.predict(X) != fair.predict(X)) / len(X) < 0.01


def test_same_logistic_multiclass(random_xy_dataset_multiclf):
    """Tests whether the fair classifier performs similar to logistic regression when we set a high threshold"""
    X, y = random_xy_dataset_multiclf

    lr = LogisticRegression(penalty='none', solver='lbfgs', multi_class='ovr')
    fair = FairClassifier(covariance_threshold=99999, sensitive_cols=[0], C=99999)
    try:
        fair.fit(X, y)
    except SolverError:
        pass
    else:
        lr.fit(X, y)

        np.testing.assert_almost_equal(lr.predict_proba(X), fair.predict_proba(X), decimal=2)
        assert np.sum(lr.predict(X) != fair.predict(X)) / len(X) < 0.01
