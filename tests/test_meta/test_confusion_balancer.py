import pytest
from sklearn.linear_model import LogisticRegression


from sklego.common import flatten
from sklego.meta import ConfusionBalancer
from tests.conftest import general_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([
    general_checks,
    classifier_checks
]))
def test_estimator_checks_classification(test_fn):
    trf = ConfusionBalancer(LogisticRegression(solver='lbfgs'))
    test_fn(ConfusionBalancer.__name__, trf)
