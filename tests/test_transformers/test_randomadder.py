import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.utils import estimator_checks
from sklearn.utils.estimator_checks import check_transformers_unfitted

from sklego.common import flatten
from sklego.transformers import RandomAdder
from tests.conftest import nonmeta_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    # Transformer checks
    check_transformers_unfitted,
    # General checks
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_fit2d_1sample,
    estimator_checks.check_fit2d_1feature,
    estimator_checks.check_fit1d,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_dont_overwrite_parameters
]))
def test_estimator_checks(test_fn):
    # Tests that are skipped:
    # check_methods_subset_invariance: Since we add noise, the method is not invariant on a subset
    # check_transformer_data_not_an_array: tests with `NotAnArray` as X for which we don't have a hashing function
    # check_transformer_general: tests with lists as X for which we don't have a hashing function
    adder = RandomAdder()
    test_fn(RandomAdder.__name__, adder)


def test_dtype_regression(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float


def test_dtype_classification(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert RandomAdder().fit(X, y).transform(X).dtype == np.float


def test_only_transform_train(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_adder = RandomAdder()
    random_adder.fit(X_train, y_train)

    assert np.all(random_adder.transform(X_train) != X_train)
    assert np.all(random_adder.transform(X_test) == X_test)
