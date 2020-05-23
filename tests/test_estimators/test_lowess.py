import numpy as np
import pytest

from sklego.common import flatten
from sklego.linear_model import LowessRegression
from tests.conftest import nonmeta_checks, regressor_checks, general_checks


@pytest.mark.parametrize(
    "test_fn", flatten([nonmeta_checks, general_checks, regressor_checks])
)
def test_estimator_checks(test_fn):
    lowess = LowessRegression()
    test_fn(LowessRegression.__name__, lowess)


def test_obvious_usecase():
    x = np.linspace(0, 10, 100)
    X = x.reshape(-1, 1)
    y = np.ones(x.shape)
    y_pred = LowessRegression().fit(X, y).predict(X)
    assert np.isclose(y, y_pred).all()