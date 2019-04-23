import numpy as np
import pytest
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.dummy import RandomRegressor
from sklego.testing import check_shape_remains_same_regressor
from tests.conftest import nonmeta_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    check_shape_remains_same_regressor,
    # General checks
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters,
    # Regressor checks
    estimator_checks.check_regressor_data_not_an_array,
    estimator_checks.check_estimators_partial_fit_n_features,
    estimator_checks.check_regressors_no_decision_function,
    estimator_checks.check_supervised_y_2d,
    estimator_checks.check_supervised_y_no_nan,
    estimator_checks.check_regressors_int,
    estimator_checks.check_estimators_unfitted,
]))
def test_estimator_checks(test_fn):
    # Tests that are skipped:
    # 'check_methods_subset_invariance': Since we add noise, the method is not invariant on a subset
    # 'check_regressors_train': score is not always greater than 0.5 due to randomness
    regr_normal = RandomRegressor(strategy="normal")
    test_fn(RandomRegressor.__name__ + '_normal', regr_normal)

    regr_uniform = RandomRegressor(strategy="uniform")
    test_fn(RandomRegressor.__name__ + '_uniform', regr_uniform)


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
