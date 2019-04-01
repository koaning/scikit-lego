import pytest
import numpy as np
from sklego.mixture import GMMOutlierDetector


def test_obvious_usecase():
    X = np.concatenate([np.random.normal(0, 1, (2000, 2))])
    mod = GMMOutlierDetector(n_components=1, threshold=0.999).fit(X)
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
