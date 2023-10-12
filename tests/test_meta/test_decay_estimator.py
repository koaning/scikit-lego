import pytest
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import is_regressor, is_classifier


from sklego.common import flatten
from sklego.meta import DecayEstimator
from tests.conftest import (
    general_checks,
    classifier_checks,
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
    "mod", flatten([LinearRegression(), Ridge(), DecisionTreeRegressor()])
)
def test_decay_weight_regr(mod):
    X, y = np.random.normal(0, 1, (100, 100)), np.random.normal(0, 1, (100,))
    mod = DecayEstimator(mod, decay=0.95).fit(X, y)
    assert mod.weights_[0] == pytest.approx(0.95**100, abs=0.001)


@pytest.mark.parametrize(
    "mod", flatten([DecisionTreeClassifier(), LogisticRegression(solver="lbfgs")])
)
def test_decay_weight_clf(mod):
    X, y = (
        np.random.normal(0, 1, (100, 100)),
        (np.random.normal(0, 1, (100,)) < 0).astype(int),
    )
    mod = DecayEstimator(mod, decay=0.95).fit(X, y)
    assert mod.weights_[0] == pytest.approx(0.95**100, abs=0.001)


@pytest.mark.parametrize("mod", flatten([KNeighborsClassifier()]))
def test_throw_warning(mod):
    X, y = np.random.normal(0, 1, (100, 100)), np.random.normal(0, 1, (100,)) < 0
    with pytest.raises(TypeError) as e:
        DecayEstimator(mod, decay=0.95).fit(X, y)
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
