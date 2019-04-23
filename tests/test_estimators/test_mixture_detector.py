import numpy as np
import pytest
from sklearn.utils import estimator_checks

from sklego.common import flatten
from sklego.mixture import GMMOutlierDetector
from tests.conftest import nonmeta_checks, general_checks


@pytest.mark.parametrize("test_fn", flatten([
    nonmeta_checks,
    general_checks,
    # outlier checks
    estimator_checks.check_outliers_fit_predict,
    estimator_checks.check_classifier_data_not_an_array,
    estimator_checks.check_estimators_unfitted,
]))
def test_estimator_checks(test_fn):
    clf_quantile = GMMOutlierDetector(threshold=0.999, method="quantile")
    test_fn(GMMOutlierDetector.__name__ + '_quantile', clf_quantile)

    clf_stddev = GMMOutlierDetector(threshold=2, method="stddev")
    test_fn(GMMOutlierDetector.__name__ + '_stddev', clf_stddev)


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


def test_obvious_usecase_quantile(dataset):
    mod = GMMOutlierDetector(n_components=2, threshold=0.999, method="quantile").fit(dataset)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert (mod.predict([[0, 0]]) == np.array([-1])).all()


def test_obvious_usecase_stddev(dataset):
    mod = GMMOutlierDetector(n_components=2, threshold=2, method="stddev").fit(dataset)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert (mod.predict([[0, 0]]) == np.array([-1])).all()


def test_value_error_threshold(dataset):
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=10).fit(dataset)
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=-10).fit(dataset)
    with pytest.raises(ValueError):
        GMMOutlierDetector(megatondinosaurhead=1).fit(dataset)
    with pytest.raises(ValueError):
        GMMOutlierDetector(method="dinosaurhead").fit(dataset)
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=-10, method="stddev").fit(dataset)


def test_thresh_effect_stddev(dataset):
    mod1 = GMMOutlierDetector(threshold=1, method="stddev").fit(dataset)
    mod2 = GMMOutlierDetector(threshold=2, method="stddev").fit(dataset)
    mod3 = GMMOutlierDetector(threshold=3, method="stddev").fit(dataset)
    assert mod1.predict(dataset).sum() > mod2.predict(dataset).sum()
    assert mod2.predict(dataset).sum() > mod3.predict(dataset).sum()


def test_thresh_effect_quantile(dataset):
    mod1 = GMMOutlierDetector(threshold=0.90, method="quantile").fit(dataset)
    mod2 = GMMOutlierDetector(threshold=0.95, method="quantile").fit(dataset)
    mod3 = GMMOutlierDetector(threshold=0.99, method="quantile").fit(dataset)
    assert mod1.predict(dataset).sum() > mod2.predict(dataset).sum()
    assert mod2.predict(dataset).sum() > mod3.predict(dataset).sum()
