import numpy as np
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
