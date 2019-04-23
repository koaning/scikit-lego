import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklego.dummy import RandomRegressor
from sklego.linear_model import DeadZoneRegressor
from sklego.mixture import GMMClassifier, GMMOutlierDetector
from tests.conftest import id_func


@pytest.mark.parametrize("estimator", [
    RandomRegressor(strategy="uniform"),
    RandomRegressor(strategy="normal"),
    DeadZoneRegressor(effect="linear", n_iter=100),
    DeadZoneRegressor(effect="quadratic", n_iter=100),
], ids=id_func)
def test_shape_regression(estimator, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert estimator.fit(X, y).predict(X).shape[0] == y.shape[0]
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', estimator)])
    assert pipe.fit(X, y).predict(X).shape[0] == y.shape[0]


@pytest.mark.parametrize("estimator", [
    GMMClassifier(),
    GMMOutlierDetector(threshold=0.999, method="quantile"),
    GMMOutlierDetector(threshold=2, method="stddev")
], ids=id_func)
def test_shape_classification(estimator, random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert estimator.fit(X, y).predict(X).shape[0] == y.shape[0]
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', estimator)])
    assert pipe.fit(X, y).predict(X).shape[0] == y.shape[0]
