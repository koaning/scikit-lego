import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import OutlierClassifier
from sklego.mixture import GMMOutlierDetector


@parametrize_with_checks([OutlierClassifier(GMMOutlierDetector(threshold=0.1, method="quantile"))])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        # Since `OutlierClassifier` is a classifier (`ClassifierMixin`), parametrize_with_checks feeds a classification
        # dataset. However this is not how this classifier is supposed to be used.
        "check_classifiers_train",
        "check_classifiers_classes",
        # Similarly, the original dataset could also be regression task depending on the outlier detection algo
        "check_classifiers_regression_target",
    }:
        pytest.skip()

    check(estimator)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.random.normal(0, 1, (2000, 2))


@pytest.mark.parametrize("outlier_model", [GMMOutlierDetector(), OneClassSVM(nu=0.05), IsolationForest()])
def test_obvious_usecase(dataset, outlier_model):
    outlier_clf = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(int)
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
    y = (dataset.max(axis=1) > 3).astype(int)
    with pytest.raises(ValueError):
        clf_quantile.fit(X, y)


def test_raises_error_no_decision_function(dataset):
    outlier_model = LocalOutlierFactor()
    clf_model = OutlierClassifier(outlier_model)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(int)
    with pytest.raises(ValueError):
        clf_model.fit(X, y)
