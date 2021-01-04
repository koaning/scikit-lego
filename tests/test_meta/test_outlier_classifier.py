import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

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
    assert isinstance(clf_quantile.score(X, y), float)


def test_raises_error(dataset):
    mod_quantile = LinearRegression()
    clf_quantile = OutlierClassifier(mod_quantile)
    X = dataset
    y = (dataset.max(axis=1) > 3).astype(np.int)
    with pytest.raises(ValueError):
        clf_quantile.fit(X, y)
