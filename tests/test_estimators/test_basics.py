import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from sklego.dummy import RandomRegressor
from sklego.mixture import GMMClassifier, GMMOutlierDetector
from tests.conftest import id_func


regressors = [
    RandomRegressor(),
]

classifiers = [
    GMMClassifier(),
    GMMOutlierDetector(threshold=0.999, method="quantile"),
    GMMOutlierDetector(threshold=2, method="stddev")
]


# @pytest.mark.parametrize("estimator", regressors, ids=id_func)
# def test_sklearn_regression(estimator):
#     check_estimator(estimator)
#
#
# @pytest.mark.parametrize("estimator", classifiers, ids=id_func)
# def test_sklearn_classification(estimator):
#     check_estimator(estimator)


@pytest.mark.parametrize("estimator", regressors, ids=id_func)
def test_shape_regression(estimator, random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    assert estimator.fit(X, y).predict(X).shape[0] == y.shape[0]
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', estimator)])
    assert pipe.fit(X, y).predict(X).shape[0] == y.shape[0]


@pytest.mark.parametrize("estimator", classifiers, ids=id_func)
def test_shape_classification(estimator, random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    assert estimator.fit(X, y).predict(X).shape[0] == y.shape[0]
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('clf', estimator)])
    assert pipe.fit(X, y).predict(X).shape[0] == y.shape[0]
