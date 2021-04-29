"""Test the ZeroInflatedRegressor."""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from sklego.common import flatten
from sklego.meta import ZeroInflatedRegressor
from sklego.testing import check_shape_remains_same_regressor
from tests.conftest import general_checks, select_tests, regressor_checks


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_zir(test_fn):
    regr = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(random_state=0),
        regressor=ExtraTreesRegressor(random_state=0)
    )
    test_fn(ZeroInflatedRegressor.__name__, regr)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks, regressor_checks]),
    )
)
def test_estimator_checks(test_fn):
    test_fn(
        ZeroInflatedRegressor.__name__,
        ZeroInflatedRegressor(
            classifier=ExtraTreesClassifier(random_state=0),
            regressor=ExtraTreesRegressor(random_state=0)
        )
    )


def test_zero_inflated_example():
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(random_state=0),
        regressor=ExtraTreesRegressor(random_state=0)
    )

    zir_score = cross_val_score(zir, X, y).mean()
    et_score = cross_val_score(ExtraTreesRegressor(), X, y).mean()

    assert zir_score > 0.85
    assert zir_score > et_score

def test_zero_inflated_with_sample_weights_example():
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(random_state=0),
        regressor=ExtraTreesRegressor(random_state=0)
    )

    zir_score = cross_val_score(zir, X, y, fit_params={'sample_weight': np.arange(len(y))}).mean()

    assert zir_score > 0.85

def test_wrong_estimators_exceptions():
    X = np.array([[0.]])
    y = np.array([0.])

    with pytest.raises(ValueError, match="`classifier` has to be a classifier."):
        zir = ZeroInflatedRegressor(ExtraTreesRegressor(), ExtraTreesRegressor())
        zir.fit(X, y)

    with pytest.raises(ValueError, match="`regressor` has to be a regressor."):
        zir = ZeroInflatedRegressor(ExtraTreesClassifier(), ExtraTreesClassifier())
        zir.fit(X, y)



