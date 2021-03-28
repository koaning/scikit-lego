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


@pytest.mark.parametrize('outlier_model', [GMMOutlierDetector(), OneClassSVM(nu=0.05), IsolationForest()])
def test_obvious_usecase(dataset, outlier_model):
    outlier_clf = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    outlier_clf.fit(X, y)
    assert outlier_clf.predict([[10, 10]]) == np.array([1])
    assert outlier_clf.predict([[0, 0]]) == np.array([0])
    np.testing.assert_array_almost_equal(outlier_clf.predict_proba([[0, 0]]), np.array([[1, 0]]), decimal=3)
    np.testing.assert_allclose(outlier_clf.predict_proba([[10, 10]]), np.array([[0, 1]]), atol=0.2)
    assert isinstance(outlier_clf.score(X, y), float)


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
