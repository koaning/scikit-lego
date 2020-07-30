import numpy as np
import pytest

from sklego.impute import SVDImputer


@pytest.fixture
def X_test():
    np.random.seed(42)
    return np.random.normal(size=(100, 5))


def test_get_kth_approximation_shape(X_test):
    for k in range(X_test.shape[1]):
        svdi = SVDImputer(k)
        assert svdi._get_kth_approximation(X_test).shape == X_test.shape


def test_get_kth_approximation_value(X_test):
    svdi = SVDImputer(X_test.shape[1])
    assert np.allclose(svdi._get_kth_approximation(X_test), X_test)
