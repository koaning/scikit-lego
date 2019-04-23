import pytest
import numpy as np
from sklearn.utils import estimator_checks

from sklego.dummy import RandomRegressor
from sklego.testing import check_shape_remains_same_regressor


@pytest.mark.parametrize("test_fn", [
    estimator_checks.check_classifier_data_not_an_array,
    check_shape_remains_same_regressor
])
def test_random_adder(test_fn):
    regr = RandomRegressor()
    test_fn(RandomRegressor.__name__, regr)


def test_values_uniform(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod = RandomRegressor(strategy="uniform")
    predictions = mod.fit(X, y).predict(X)
    assert (predictions >= y.min()).all()
    assert (predictions <= y.max()).all()
    assert mod.min_ == pytest.approx(y.min(), abs=0.0001)
    assert mod.max_ == pytest.approx(y.max(), abs=0.0001)


def test_values_normal(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    mod = RandomRegressor(strategy="normal").fit(X, y)
    assert mod.mu_ == pytest.approx(np.mean(y), abs=0.001)
    assert mod.sigma_ == pytest.approx(np.std(y), abs=0.001)


def test_bad_values():
    np.random.seed(42)
    X = np.random.normal(0, 1, (10, 2))
    y = np.random.normal(0, 1, (10, 1))
    with pytest.raises(ValueError):
        RandomRegressor(strategy="foobar").fit(X, y)
