import numpy as np
import pytest

from sklego.common import flatten
from sklego.dummy import RandomRegressor
from tests.conftest import nonmeta_checks, regressor_checks, general_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, nonmeta_checks, regressor_checks]),
        exclude=[
            "check_sample_weights_invariance",
            "check_methods_subset_invariance",
            "check_regressors_train"
        ]
    )
)
def test_estimator_checks(test_fn):
    # Tests that are skipped:
    # 'check_methods_subset_invariance': Since we add noise, the method is not invariant on a subset
    # 'check_regressors_train': score is not always greater than 0.5 due to randomness
    regr_normal = RandomRegressor(strategy="normal")
    test_fn(RandomRegressor.__name__ + "_normal", regr_normal)

    regr_uniform = RandomRegressor(strategy="uniform")
    test_fn(RandomRegressor.__name__ + "_uniform", regr_uniform)


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
