import pytest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

from sklego.common import flatten
from sklego.mixture import GMMOutlierDetector
from sklego.meta import OutlierClassifier

from tests.conftest import general_checks, select_tests


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        flatten([general_checks]),
        exclude=[
            "check_sample_weights_invariance",
        ]
    )
)
def test_estimator_checks(test_fn):
    mod_quantile = GMMOutlierDetector(threshold=0.999, method="quantile")
    clf_quantile = OutlierClassifier(mod_quantile)
    test_fn('OutlierClassifier', clf_quantile)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.random.normal(0, 1, (2000, 2))


def test_obvious_usecase_quantile(dataset):
    mod_quantile = GMMOutlierDetector(threshold=0.999, method="quantile")
    clf_quantile = OutlierClassifier(mod_quantile)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    clf_quantile.fit(X, y)
    assert clf_quantile.predict([[10, 10]]) == np.array([1])
    assert clf_quantile.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(clf_quantile.predict_proba([[10, 10]]), np.array([[0, 1]]), decimal=4)
    np.testing.assert_array_almost_equal(clf_quantile.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=4)
    assert isinstance(clf_quantile.score(X, y), float)


def test_obvious_usecase_isolationforest(dataset):
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    outlier_model = IsolationForest(contamination=y.sum()/len(y))
    clf_model = OutlierClassifier(outlier_model)
    clf_model.fit(X, y)
    assert clf_model.predict([[10, 10]]) == np.array([1])
    assert clf_model.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(clf_model.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=4)
    outlier_proba = clf_model.predict_proba([[10, 10]])[0]
    assert outlier_proba[0]<0.2
    assert outlier_proba[1]>0.8
    assert isinstance(clf_model.score(X, y), float)


def test_obvious_usecase_oneclass_svm(dataset):
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    outlier_model = OneClassSVM(nu=0.05)
    clf_model = OutlierClassifier(outlier_model)
    clf_model.fit(X, y)
    assert clf_model.predict([[10, 10]]) == np.array([1])
    assert clf_model.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(clf_model.predict_proba([[10, 10]]), np.array([[0, 1]]), decimal=3)
    np.testing.assert_array_almost_equal(clf_model.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=3)
    assert isinstance(clf_model.score(X, y), float)


def test_raises_error(dataset):
    mod_quantile = LinearRegression()
    clf_quantile = OutlierClassifier(mod_quantile)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    with pytest.raises(ValueError):
        clf_quantile.fit(X, y)


def test_raises_error_no_decision_function(dataset):
    outlier_model = LocalOutlierFactor()
    clf_model = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    with pytest.raises(ValueError):
        clf_model.fit(X, y)