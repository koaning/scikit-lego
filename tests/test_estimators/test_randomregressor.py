import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.dummy import RandomRegressor


@parametrize_with_checks(
    [RandomRegressor(strategy="normal", random_state=42), RandomRegressor(strategy="uniform", random_state=42)]
)
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_methods_subset_invariance",
        "check_methods_sample_order_invariance",
        "check_pipeline_consistency",
    }:
        pytest.skip("RandomRegressor is not invariant")
    check(estimator)


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
