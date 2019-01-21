import itertools as it

import numpy as np

import pytest


@pytest.fixture(scope="module", params=[_ for _ in it.product((1, 100, 1000, 10000), (1, 5, 10, 25,))])
def random_xy_dataset(request):
    n, k = request.param
    X = np.random.normal(0, 1, (n, k))
    y = np.random.normal(0, 1, (n, 1))
    return X, y
