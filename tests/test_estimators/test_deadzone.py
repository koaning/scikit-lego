import pytest
import numpy as np

from sklego.linear_model import DeadZoneRegressor
from sklego.testing import check_shape_remains_same_regressor


@pytest.fixture
def dataset():
    np.random.seed(42)
    n = 1000
    inputs = np.concatenate([np.ones((n, 1)), np.random.normal(0, 1, (n, 1))], axis=1)
    targets = 3.1 * inputs[:, 0] + 2.0 * inputs[:, 1] + np.random.normal(0, 1, (n, ))
    return inputs, targets


@pytest.fixture(scope="module", params=["linear", "quadratic"])
def mod(request):
    return DeadZoneRegressor(effect=request.param)


@pytest.mark.parametrize("test_fn", [
    check_shape_remains_same_regressor
])
def test_deadzone(test_fn):
    regr = DeadZoneRegressor()
    test_fn(DeadZoneRegressor.__name__, regr)


def test_values_uniform(dataset, mod):
    X, y = dataset
    coefs = mod.fit(X, y).coefs_
    assert coefs[0] == pytest.approx(3.1, abs=0.1)
    assert coefs[1] == pytest.approx(2.0, abs=0.1)
