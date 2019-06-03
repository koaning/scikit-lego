import pytest
import numpy as np

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
    clf1 = GaussianMixtureNB()
    clf2 = GaussianMixtureNB(n_components=5)
    test_fn(GaussianMixtureNB.__name__, clf1)
    test_fn(GaussianMixtureNB.__name__ + "_components_5", clf2)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


@pytest.mark.parametrize('k', [1, 5, 10])
def test_obvious_usecase(k):
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GaussianMixtureNB(n_components=k).fit(X, y).predict(X) == y).all()


def test_value_error_threshold():
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    with pytest.raises(ValueError):
        GaussianMixtureNB(megatondinosaurhead=1).fit(X, y)
