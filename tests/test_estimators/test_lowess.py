import re

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.linear_model import LowessRegression


@parametrize_with_checks([LowessRegression()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_obvious_usecase():
    x = np.linspace(0, 10, 100)
    X = x.reshape(-1, 1)
    y = np.ones(x.shape)
    y_pred = LowessRegression().fit(X, y).predict(X)
    assert np.isclose(y, y_pred).all()


def test_custom_error_for_zero_division():
    x = np.arange(0, 100)
    X = x.reshape(-1, 1)
    y = np.ones(x.shape)
    estimator = LowessRegression(sigma=1e-10).fit(X, y)

    with pytest.raises(
        ValueError, match=re.escape("Weights, resulting from `np.exp(-(distances**2) / self.sigma)`, are all zero.")
    ):
        estimator.predict(X[:10] + 0.5)
