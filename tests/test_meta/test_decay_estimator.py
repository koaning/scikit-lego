import numpy as np
import pytest
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklego.common import flatten
from sklego.meta import DecayEstimator
from tests.conftest import (
    classifier_checks,
    general_checks,
    regressor_checks,
)


@pytest.mark.parametrize("test_fn", flatten([general_checks, regressor_checks]))
def test_estimator_checks_regression(test_fn):
    trf = DecayEstimator(LinearRegression(), check_input=True)
    test_fn(DecayEstimator.__name__, trf)


@pytest.mark.parametrize("test_fn", flatten([general_checks, classifier_checks]))
def test_estimator_checks_classification(test_fn):
    trf = DecayEstimator(LogisticRegression(solver="lbfgs"), check_input=True)
    test_fn(DecayEstimator.__name__, trf)


@pytest.mark.parametrize(
    "mod, is_clf",
    [
        (LinearRegression(), False),
        (Ridge(), False),
        (DecisionTreeRegressor(), False),
        (DecisionTreeClassifier(), True),
        (LogisticRegression(solver="lbfgs"), True),
    ],
)
@pytest.mark.parametrize(
    "decay_func, decay_kwargs",
    [
        ("exponential", {"decay_rate": 0.999}),
        ("exponential", {"decay_rate": 0.99}),
        ("linear", {"min_value": 0.0, "max_value": 1.0}),
        ("linear", {"min_value": 0.5, "max_value": 1.0}),
        ("sigmoid", {"growth_rate": 0.1}),
        ("sigmoid", {"growth_rate": None}),
        ("stepwise", {"n_steps": 10}),
        ("stepwise", {"step_size": 2}),
    ],
)
def test_decay_weight(mod, is_clf, decay_func, decay_kwargs):
    X, y = np.random.normal(0, 1, (100, 100)), np.random.normal(0, 1, (100,))

    if is_clf:
        y = (y < 0).astype(int)

    mod = DecayEstimator(mod, decay_func=decay_func, **decay_kwargs).fit(X, y)

    assert np.logical_and(mod.weights_ >= 0, mod.weights_ <= 1).all()
    assert np.all(mod.weights_[:-1] <= mod.weights_[1:])


@pytest.mark.parametrize("mod", flatten([KNeighborsClassifier()]))
def test_throw_warning(mod):
    X, y = np.random.normal(0, 1, (100, 100)), np.random.normal(0, 1, (100,)) < 0
    with pytest.raises(TypeError) as e:
        DecayEstimator(mod, decay_rate=0.95).fit(X, y)
        assert "sample_weight" in str(e)
        assert type(mod).__name__ in str(e)


@pytest.mark.parametrize(
    "mod, is_regr",
    [
        (LinearRegression(), True),
        (Ridge(), True),
        (DecisionTreeRegressor(), True),
        (LogisticRegression(), False),
        (DecisionTreeClassifier(), False),
    ],
)
def test_estimator_type_regressor(mod, is_regr):
    mod = DecayEstimator(mod)
    assert mod._estimator_type == mod.model._estimator_type
    assert is_regressor(mod) == is_regr
    assert is_classifier(mod) == (not is_regr)
