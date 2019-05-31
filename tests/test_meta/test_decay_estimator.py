import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklego.common import flatten
from sklego.meta import DecayEstimator
from tests.conftest import general_checks, classifier_checks, regressor_checks, nonmeta_checks


@pytest.mark.parametrize("test_fn", flatten([
    general_checks,
    nonmeta_checks,
    regressor_checks
]))
def test_estimator_checks_regression(test_fn):
    trf = DecayEstimator(LinearRegression())
    test_fn(DecayEstimator.__name__, trf)


@pytest.mark.parametrize("test_fn", flatten([
    general_checks,
    nonmeta_checks,
    classifier_checks
]))
def test_estimator_checks_classification(test_fn):
    trf = DecayEstimator(LogisticRegression())
    test_fn(DecayEstimator.__name__, trf)
