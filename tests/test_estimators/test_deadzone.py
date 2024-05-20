import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.linear_model import DeadZoneRegressor
from sklego.testing import check_shape_remains_same_regressor


@parametrize_with_checks([DeadZoneRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    n = 100
    inputs = np.concatenate([np.ones((n, 1)), np.random.normal(0, 1, (n, 1))], axis=1)
    targets = 3.1 * inputs[:, 0] + 2.0 * inputs[:, 1]
    return inputs, targets


@pytest.fixture(scope="module", params=["constant", "linear", "quadratic"])
def mod(request):
    return DeadZoneRegressor(effect=request.param, threshold=0.3)


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_deadzone(test_fn):
    regr = DeadZoneRegressor()
    test_fn(DeadZoneRegressor.__name__, regr)


def test_values_uniform(dataset, mod):
    if mod.effect == "constant":
        pytest.skip("Constant effect")
    X, y = dataset
    coefs = mod.fit(X, y).coef_
    assert coefs[0] == pytest.approx(3.1, abs=0.2)
    assert coefs[1] == pytest.approx(2.0, abs=0.2)
