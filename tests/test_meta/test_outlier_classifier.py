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


def check_predict_proba(outlier_classifier):
    assert outlier_classifier.predict([[10, 10]]) == np.array([1])
    assert outlier_classifier.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(outlier_classifier.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=3)
    np.testing.assert_allclose(outlier_classifier.predict_proba([[10, 10]]), np.array([[0, 1]]), atol=0.2)


@pytest.mark.parametrize(
    "test_fn",
    [check_predict_proba]
)
def test_obvious_usecases(test_fn, dataset):
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    clf_isolation_forest = OutlierClassifier(IsolationForest(contamination=y.sum() / len(y))).fit(X, y)
    test_fn(clf_isolation_forest)
    clf_svm = OutlierClassifier(OneClassSVM(nu=0.05)).fit(X, y)
    test_fn(clf_svm)


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
