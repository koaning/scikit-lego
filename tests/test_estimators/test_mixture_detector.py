import pytest
import numpy as np
from sklego.mixture import GMMOutlierDetector


def test_obvious_usecase_quantile():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    mod = GMMOutlierDetector(n_components=2, threshold=0.999, method="quantile").fit(X)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert not mod.predict([[0, 0]]).all()


def test_obvious_usecase_stddev():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    mod = GMMOutlierDetector(n_components=2, threshold=2, method="stddev").fit(X)
    assert mod.predict([[10, 10], [-10, -10]]).all()
    assert not mod.predict([[0, 0]]).all()


def test_value_error_threshold():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=10).fit(X)
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=-10).fit(X)
    with pytest.raises(ValueError):
        GMMOutlierDetector(megatondinosaurhead=1).fit(X)
    with pytest.raises(ValueError):
        GMMOutlierDetector(method="dinosaurhead").fit(X)
    with pytest.raises(ValueError):
        GMMOutlierDetector(threshold=-10, method="stddev").fit(X)


def test_thresh_effect_stddev():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    mod1 = GMMOutlierDetector(threshold=1, method="stddev").fit(X)
    mod2 = GMMOutlierDetector(threshold=2, method="stddev").fit(X)
    mod3 = GMMOutlierDetector(threshold=3, method="stddev").fit(X)
    assert mod1.predict(X).sum() > mod2.predict(X).sum()
    assert mod2.predict(X).sum() > mod3.predict(X).sum()


def test_thresh_effect_quantile():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    mod1 = GMMOutlierDetector(threshold=0.90, method="quantile").fit(X)
    mod2 = GMMOutlierDetector(threshold=0.95, method="quantile").fit(X)
    mod3 = GMMOutlierDetector(threshold=0.99, method="quantile").fit(X)
    assert mod1.predict(X).sum() > mod2.predict(X).sum()
    assert mod2.predict(X).sum() > mod3.predict(X).sum()

