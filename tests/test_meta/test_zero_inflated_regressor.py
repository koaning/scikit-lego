"""Test the ZeroInflatedRegressor."""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import ZeroInflatedRegressor
from sklego.testing import check_shape_remains_same_regressor

params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=-1)


@pytest.mark.parametrize("test_fn", [check_shape_remains_same_regressor])
def test_zir(test_fn):
    regr = ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params))
    test_fn(ZeroInflatedRegressor.__name__, regr)


@parametrize_with_checks(
    [ZeroInflatedRegressor(classifier=ExtraTreesClassifier(**params), regressor=ExtraTreesRegressor(**params))]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_zero_inflated_example():
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(
        classifier=ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
        regressor=ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
    )

    zir_score = cross_val_score(zir, X, y).mean()
    et_score = cross_val_score(ExtraTreesRegressor(), X, y).mean()

    assert zir_score > 0.85
    assert zir_score > et_score


@pytest.mark.parametrize(
    "classifier,regressor,performance",
    [
        (
            ExtraTreesClassifier(max_depth=20, random_state=0, n_jobs=-1),
            ExtraTreesRegressor(max_depth=20, random_state=0, n_jobs=-1),
            0.85,
        ),
        (KNeighborsClassifier(), KNeighborsRegressor(), 0.55),
    ],
)
def test_zero_inflated_with_sample_weights_example(classifier, regressor, performance):
    from sklearn.model_selection import cross_val_score

    np.random.seed(0)
    X = np.random.randn(10_000, 4)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)  # many zeroes here, in about 75% of the cases.

    zir = ZeroInflatedRegressor(classifier=classifier, regressor=regressor)

    zir_score = cross_val_score(zir, X, y, fit_params={"sample_weight": np.arange(len(y))}).mean()
    # TODO: fit_params -> params in future versions

    assert zir_score > performance


def test_wrong_estimators_exceptions():
    X = np.array([[0.0]])
    y = np.array([0.0])

    with pytest.raises(ValueError, match="`classifier` has to be a classifier."):
        zir = ZeroInflatedRegressor(ExtraTreesRegressor(), ExtraTreesRegressor())
        zir.fit(X, y)

    with pytest.raises(ValueError, match="`regressor` has to be a regressor."):
        zir = ZeroInflatedRegressor(ExtraTreesClassifier(), ExtraTreesClassifier())
        zir.fit(X, y)
