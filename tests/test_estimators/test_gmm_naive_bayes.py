import pytest
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.naive_bayes import GaussianMixtureNB
from sklego.testing import check_shape_remains_same_classifier
from tests.conftest import nonmeta_checks, general_checks, classifier_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    general_checks,
    classifier_checks,
    check_shape_remains_same_classifier
]))
def test_estimator_checks(test_fn):
    clf = GaussianMixtureNB()
    test_fn(GaussianMixtureNB.__name__, clf)