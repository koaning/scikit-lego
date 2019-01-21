import numpy as np
from skblocks.transformers import RandomAdder


def test_shape_does_not_change():
    X = np.random.normal(0, 1, (10, 100))
    y = np.random.normal(0, 1, (1, 100))
    assert RandomAdder().fit(X, y).transform(X).shape == X.shape