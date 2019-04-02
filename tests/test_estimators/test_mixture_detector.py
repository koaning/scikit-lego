import pytest
import numpy as np
from sklego.mixture import GMMOutlierDetector


@pytest.fixture
def dataset():
    np.random.seed(42)
    return np.concatenate([np.random.normal(0, 1, (2000, 2))])


def test_obvious_usecase_quantile(dataset):
    mod = GMMOutlierDetector(n_components=2, threshold=0.999, method="quantile").fit(dataset)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert not mod.predict([[0, 0]]).all()


def test_obvious_usecase_stddev(dataset):
    mod = GMMOutlierDetector(n_components=2, threshold=2, method="stddev").fit(dataset)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert not mod.predict([[0, 0]]).all()


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
