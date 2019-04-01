import pytest
import numpy as np
from sklego.mixture import GMMClassifier


def test_obvious_usecase():
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    assert (GMMClassifier().fit(X, y).predict(X) == y).all()


def test_value_error_threshold():
    X = np.concatenate([np.random.normal(-10, 1, (100, 2)), np.random.normal(10, 1, (100, 2))])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    with pytest.raises(ValueError):
        GMMClassifier(megatondinosaurhead=1).fit(X, y)
