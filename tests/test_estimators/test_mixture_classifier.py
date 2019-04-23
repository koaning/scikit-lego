import pytest
import numpy as np

from sklego.mixture import GMMClassifier
from sklego.testing import check_shape_remains_same_classifier


@pytest.mark.parametrize("test_fn", [
    check_shape_remains_same_classifier
])
def test_mixture_clf(test_fn):
    """Test for a whole bunch of standard checks."""
    clf = GMMClassifier()
    test_fn(GMMClassifier.__name__, clf)


def test_obvious_usecase():
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GMMClassifier().fit(X, y).predict(X) == y).all()


def test_value_error_threshold():
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    with pytest.raises(ValueError):
        GMMClassifier(megatondinosaurhead=1).fit(X, y)
