import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import RegressionOutlierDetector


@parametrize_with_checks([RegressionOutlierDetector(LinearRegression(), column=0)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_fit2d_1feature",  # custom message
    }:
        pytest.skip()

    check(estimator)


def test_obvious_example():
    # generate random data for illustrative example
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 1))
    y = 1 + np.sum(X, axis=1).reshape(-1, 1) + np.random.normal(0, 0.2, (100, 1))
    for i in [20, 25, 50, 80]:
        y[i] += 2
    X = np.concatenate([X, y], axis=1)

    # fit and plot
    mod = RegressionOutlierDetector(LinearRegression(), column=1)
    preds = mod.fit(X).predict(X)
    for i in [20, 25, 50, 80]:
        assert preds[i] == -1


def test_obvious_example_pandas():
    # generate random data for illustrative example
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 1 + x + np.random.normal(0, 0.2, 100)
    for i in [20, 25, 50, 80]:
        y[i] += 2
    X = pd.DataFrame({"x": x, "y": y})

    # fit and plot
    mod = RegressionOutlierDetector(LinearRegression(), column="y")
    preds = mod.fit(X).predict(X)
    for i in [20, 25, 50, 80]:
        assert preds[i] == -1


def test_raises_error():
    # generate random data for illustrative example
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = 1 + x + np.random.normal(0, 0.2, 100)
    for i in [20, 25, 50, 80]:
        y[i] += 2
    X = pd.DataFrame({"x": x, "y": y})

    with pytest.raises(ValueError):
        mod = RegressionOutlierDetector(LogisticRegression(), column="y")
        mod.fit(X)
